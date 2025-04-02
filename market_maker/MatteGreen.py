import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import deque 
import matplotlib.pyplot as plt
import mplfinance as mpf
from market_maker.settings import settings
from market_maker.market_maker import OrderManager
from market_maker.utils.TeleLogBot import configure_logging, TelegramBot  # Assumes TeleLogBot is here
from market_maker import bitmex  # BitMEX API integration

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class MatteGreenOrderManager(OrderManager):
    def __init__(self):
        super().__init__()
        # MatteGreen-specific parameters
        self.timeframe = "5m"
        self.initial_capital = 10000
        self.current_balance = self.initial_capital
        self.risk_per_trade = 0.02
        self.rr_ratio = 1.25
        self.lookback_period = 20
        self.fvg_threshold = 0.003
        #BOT_TOKEN =   # Replace with actual token
        #CHAT_ID =      # Replace with actual chat ID
        self.telegram_token = settings.BOT_TOKEN
        self.telegram_chat_id = settings.CHAT_ID
        
        # Logging and Telegram setup
        self.logger, self.bot = configure_logging(self.telegram_token, self.telegram_chat_id)
        self.telegram_bot = TelegramBot(self.telegram_token, self.telegram_chat_id) if self.telegram_token and self.telegram_chat_id else None
        
        # Market data and state
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.current_trades = []  # List of (orderID, entry_price, direction, stop_loss, take_profit, size)
        self.trades = []  # Completed trades
        self.equity_curve = [self.initial_capital]
        self.market_bias = 'neutral'
        
        self.logger.info(f"🎉 MatteGreenOrderManager initialized for {self.exchange.symbol} on {self.timeframe}")

    def get_market_data(self):
        """Fetch candle data using BitMEX API."""
        try:
            # Use BitMEX API to fetch candles (assuming binSize matches timeframe)
            data = self.exchange.bitmex.candles(binSize=self.timeframe, count=self.lookback_period * 2)
            if not data or len(data) == 0:
                self.logger.error("🚨 No data from BitMEX API")
                return False
            # Convert to DataFrame and set timestamp index
            self.df = pd.DataFrame(data)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
            self.df['higher_high'] = False
            self.df['lower_low'] = False
            self.df['bos_up'] = False
            self.df['bos_down'] = False
            self.df['choch_up'] = False
            self.df['choch_down'] = False
            self.df['bullish_fvg'] = False
            self.df['bearish_fvg'] = False
            return True
        except Exception as e:
            self.logger.error(f"🚨 Market data fetch failed: {str(e)}")
            return False

    def identify_swing_points(self):
        """Identify swing highs and lows."""
        window = min(self.lookback_period // 2, 3)
        self.swing_highs = np.zeros(len(self.df))
        self.swing_lows = np.zeros(len(self.df))
        for i in range(window, len(self.df) - window):
            if all(self.df['high'].iloc[i] >= self.df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['high'].iloc[i] >= self.df['high'].iloc[i+j] for j in range(1, window+1)):
                self.swing_highs[i] = 1
            if all(self.df['low'].iloc[i] <= self.df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['low'].iloc[i] <= self.df['low'].iloc[i+j] for j in range(1, window+1)):
                self.swing_lows[i] = 1

    def detect_market_structure(self):
        """Detect CHoCH, BOS, and FVG areas."""
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        recent_highs = deque(maxlen=self.lookback_period)
        recent_lows = deque(maxlen=self.lookback_period)

        for i in range(self.lookback_period, len(self.df)):
            if self.swing_highs[i]:
                recent_highs.append((i, self.df['high'].iloc[i]))
            if self.swing_lows[i]:
                recent_lows.append((i, self.df['low'].iloc[i]))

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (self.market_bias in ['bullish', 'neutral']) and \
                   recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bearish'))
                    self.market_bias = 'bearish'
                elif (self.market_bias in ['bearish', 'neutral']) and \
                     recent_lows[-1][1] > recent_lows[-2][1] and recent_highs[-1][1] > recent_highs[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bullish'))
                    self.market_bias = 'bullish'

            if self.market_bias == 'bearish' and recent_highs and self.df['high'].iloc[i] > recent_highs[-1][1]:
                self.bos_points.append((i, self.df['high'].iloc[i], 'bullish'))
            elif self.market_bias == 'bullish' and recent_lows and self.df['low'].iloc[i] < recent_lows[-1][1]:
                self.bos_points.append((i, self.df['low'].iloc[i], 'bearish'))

            if i > 1:
                if (self.df['low'].iloc[i] - self.df['high'].iloc[i-2]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i-2], self.df['low'].iloc[i], 'bullish'))
                if (self.df['low'].iloc[i-2] - self.df['high'].iloc[i]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i], self.df['low'].iloc[i-2], 'bearish'))

    def execute_trades(self):
        """Generate trading signals based on market structure."""
        signals = []
        current_idx = len(self.df) - 1
        current_price = self.df['close'].iloc[current_idx]

        total_risk_amount = sum(abs(entry_price - stop_loss) * size for _, entry_price, _, stop_loss, _, size in self.current_trades)
        max_total_risk = self.current_balance * 0.20

        # Check existing trades for exits
        for trade in list(self.current_trades):
            order_id, entry_price, direction, stop_loss, take_profit, size = trade
            if (direction == 'long' and self.df['low'].iloc[current_idx] <= stop_loss) or \
               (direction == 'short' and self.df['high'].iloc[current_idx] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.current_balance += pl
                self.trades.append({'entry_price': entry_price, 'exit_price': stop_loss, 'direction': direction, 'pl': pl, 'result': 'loss'})
                signals.append({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'order_id': order_id})
                self.current_trades.remove(trade)
                self.logger.info(f"🚨 Exit: {direction} stopped out at {stop_loss}")
            elif (direction == 'long' and self.df['high'].iloc[current_idx] >= take_profit) or \
                 (direction == 'short' and self.df['low'].iloc[current_idx] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.current_balance += pl
                self.trades.append({'entry_price': entry_price, 'exit_price': take_profit, 'direction': direction, 'pl': pl, 'result': 'win'})
                signals.append({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'order_id': order_id})
                self.current_trades.remove(trade)
                self.logger.info(f"💰 Exit: {direction} took profit at {take_profit}")

        # Enter new trades if conditions are met
        if len(self.current_trades) < 3 and current_idx >= self.lookback_period:
            direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
            if direction:
                entry_price = current_price
                lookback_start = max(0, current_idx - self.lookback_period)
                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                            max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                stop_loss = entry_price - stop_dist * 0.5 if direction == 'long' else entry_price + stop_dist * 0.5
                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                risk_of_new_trade = abs(entry_price - stop_loss) * size

                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                    signals.append({'action': 'entry', 'side': direction, 'price': entry_price, 'stop_loss': stop_loss,
                                    'take_profit': take_profit, 'position_size': int(size)})
                    self.logger.info(f"📈 Entry: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        
        self.equity_curve.append(self.current_balance)
        return signals

    def place_orders(self):
        """Override place_orders to implement MatteGreen strategy."""
        if not self.get_market_data() or len(self.df) < self.lookback_period:
            self.logger.warning(f"⚠️ Insufficient data: {len(self.df)} candles")
            return

        self.identify_swing_points()
        self.detect_market_structure()
        signals = self.execute_trades()

        buy_orders = []
        sell_orders = []

        for signal in signals:
            if signal['action'] == 'entry':
                side = "Buy" if signal['side'] == 'long' else "Sell"
                order = {
                    'price': signal['price'],
                    'orderQty': max(2, int(signal['position_size'])),  # Ensure minimum quantity
                    'side': side,
                    'stopPx': signal['stop_loss'],  # For stop-loss order
                    'execInst': 'ParticipateDoNotInitiate'  # Post-only equivalent
                }
                if side == "Buy":
                    buy_orders.append(order)
                else:
                    sell_orders.append(order)
                # Place market order and set SL/TP
                if not self.exchange.dry_run:
                    try:
                        result = self.exchange.bitmex.create(side=side, orderQty=order['orderQty'], ordType="Market")
                        if result and 'orderID' in result:
                            order_id = result['orderID']
                            self.current_trades.append((order_id, signal['price'], signal['side'], signal['stop_loss'], signal['take_profit'], signal['position_size']))
                            # Set stop-loss and take-profit
                            self.exchange.bitmex.create(side="Sell" if side == "Buy" else "Buy", orderQty=order['orderQty'], 
                                                        stopPx=signal['stop_loss'], ordType="Stop", execInst="Close")
                            self.exchange.bitmex.create(side="Sell" if side == "Buy" else "Buy", orderQty=order['orderQty'], 
                                                        price=signal['take_profit'], ordType="Limit", execInst="Close")
                    except Exception as e:
                        self.logger.error(f"🚨 Order placement failed: {str(e)}")

            elif signal['action'] == 'exit':
                order_id = signal['order_id']
                if not self.exchange.dry_run:
                    try:
                        self.exchange.bitmex.cancel(order_id)  # Individual cancellation
                        self.logger.info(f"🛑 Cancelled order {order_id} due to {signal['reason']}")
                    except Exception as e:
                        self.logger.error(f"🚨 Failed to cancel order {order_id}: {str(e)}")

        # Bulk cancellation if too many trades are open
        if len(self.current_trades) > 3:
            self.exchange.cancel_all_orders()
            self.current_trades.clear()
            self.logger.info("🧹 Bulk cancelled all orders due to exceeding trade limit")

        self.converge_orders(buy_orders, sell_orders)
        
        # Visualization and Telegram update
        if self.telegram_bot:
            fig = self.visualize_results(start_idx=max(0, len(self.df) - 48))
            sast_now = get_sast_time()
            caption = (f"📸 Scan at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                       f"Balance: ${self.current_balance:.2f}\nPrice: {self.df['close'][-1]}")
            self.telegram_bot.send_photo(fig=fig, caption=caption)

    def visualize_results(self, start_idx=0, end_idx=None):
        """Visualize market data and trades."""
        if end_idx is None:
            end_idx = len(self.df)
        subset = self.df.iloc[start_idx:end_idx]

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        mpf.plot(subset, type='candle', style='charles', ax=ax1, ylabel='Price', show_nontrading=False,
                 datetime_format='%H:%M')

        swing_high_idx = [i - start_idx for i, val in enumerate(self.swing_highs[start_idx:end_idx]) if val]
        swing_low_idx = [i - start_idx for i, val in enumerate(self.swing_lows[start_idx:end_idx]) if val]
        ax1.plot(swing_high_idx, subset['high'].iloc[swing_high_idx], 'rv', label='Swing High')
        ax1.plot(swing_low_idx, subset['low'].iloc[swing_low_idx], 'g^', label='Swing Low')

        for idx, price, c_type in self.choch_points:
            if start_idx <= idx < end_idx:
                ax1.plot(idx - start_idx, price, 'mo', label='CHoCH' if idx == self.choch_points[0][0] else "")
        for idx, price, b_type in self.bos_points:
            if start_idx <= idx < end_idx:
                ax1.plot(idx - start_idx, price, 'co', label='BOS' if idx == self.bos_points[0][0] else "")

        for start, end, high, low, fvg_type in self.fvg_areas:
            if start_idx <= end < end_idx:
                color = 'green' if fvg_type == 'bullish' else 'red'
                ax1.fill_between(range(max(0, start - start_idx), min(end - start_idx + 1, len(subset))),
                                 high, low, color=color, alpha=0.2, label=f"{fvg_type.capitalize()} FVG" if start == self.fvg_areas[0][0] else "")

        ax1.set_title(f"{self.exchange.symbol} - SMC Analysis")
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range(len(self.equity_curve)), self.equity_curve, label='Equity', color='blue')
        ax2.set_title("Equity Curve")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

def run() -> None:
    order_manager = MatteGreenOrderManager()
    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        order_manager.exit()
        sys.exit()

if __name__ == "__main__":
    run()
