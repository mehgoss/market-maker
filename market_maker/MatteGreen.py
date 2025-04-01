import sys
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from market_maker.settings import settings
from market_maker.market_maker import OrderManager
from market_maker.utils.TeleLogBot import TelegramBot, configure_logging

class MatteGreen(OrderManager):
    def __init__(self, bot_token, chat_id, initial_capital=10000, risk_per_trade=0.02,
                 rr_ratio=2, lookback_period=20, fvg_threshold=0.003):
        super().__init__()
        # Telegram setup
        self.logger, self.tg_bot = configure_logging(bot_token, chat_id)
        self.start_time = datetime.now()
        self.total_runtime = 30 * 60  # 30 minutes total
        self.break_duration = 5 * 60  # 5-minute breaks
        self.run_duration = 5 * 60    # 5-minute runs
        self.cycle_count = 0
        self.max_cycles = 5          # 5 cycles

        # SMC parameters
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = rr_ratio
        self.lookback_period = lookback_period
        self.fvg_threshold = fvg_threshold

        # Real-time data storage
        self.close = []
        self.high = []
        self.low = []
        self.open = []
        self.dates = []

        # Structure tracking
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []  # (index, price, type)
        self.bos_points = []    # (index, price, type)
        self.fvg_areas = []     # (start_idx, end_idx, min_price, max_price, type)

        # Trade tracking
        self.trades = []
        self.current_trades = []
        self.equity_curve = [initial_capital]
        self.win_count = 0
        self.loss_count = 0

    def update_market_data(self):
        """Update price data from BitMEX"""
        ticker = self.get_ticker()
        instrument = self.get_instrument()
        
        self.close.append(ticker['mid'])
        self.high.append(ticker['sell'])  # Using sell as high proxy
        self.low.append(ticker['buy'])    # Using buy as low proxy
        self.open.append(self.close[-1] if not self.open else self.close[-2])
        self.dates.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Keep only lookback_period + buffer worth of data
        if len(self.close) > self.lookback_period + 10:
            self.close.pop(0)
            self.high.pop(0)
            self.low.pop(0)
            self.open.pop(0)
            self.dates.pop(0)
            self.swing_highs.pop(0)
            self.swing_lows.pop(0)

    def identify_swing_points(self):
        """Identify swing highs and lows in real-time"""
        if len(self.close) < 5:  # Need at least 5 candles
            self.swing_highs.append(0)
            self.swing_lows.append(0)
            return

        window = min(self.lookback_period // 2, 3)
        i = len(self.close) - 1
        
        is_swing_high = all(self.high[i] >= self.high[i-j] for j in range(1, min(window+1, i+1))) and \
                       all(self.high[i] >= self.high[i+j] for j in range(1, min(window+1, len(self.high)-i)))
        is_swing_low = all(self.low[i] <= self.low[i-j] for j in range(1, min(window+1, i+1))) and \
                      all(self.low[i] <= self.low[i+j] for j in range(1, min(window+1, len(self.low)-i)))
        
        self.swing_highs.append(1 if is_swing_high else 0)
        self.swing_lows.append(1 if is_swing_low else 0)

    def detect_market_structure(self):
        """Detect SMC patterns in real-time"""
        if len(self.close) < self.lookback_period:
            return

        i = len(self.close) - 1
        recent_highs = [(idx, h) for idx, h in enumerate(self.high) if self.swing_highs[idx]]
        recent_lows = [(idx, l) for idx, l in enumerate(self.low) if self.swing_lows[idx]]
        
        market_bias = 'neutral'
        if self.choch_points:
            market_bias = self.choch_points[-1][2]
        
        # CHoCH detection
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            if (market_bias in ['bullish', 'neutral'] and 
                recent_highs[-1][1] < recent_highs[-2][1] and 
                recent_lows[-1][1] < recent_lows[-2][1]):
                self.choch_points.append((i, self.close[i], 'bearish'))
            elif (market_bias in ['bearish', 'neutral'] and 
                  recent_lows[-1][1] > recent_lows[-2][1] and 
                  recent_highs[-1][1] > recent_highs[-2][1]):
                self.choch_points.append((i, self.close[i], 'bullish'))

        # BOS detection
        if market_bias == 'bearish' and recent_highs and self.high[i] > recent_highs[-1][1]:
            self.bos_points.append((i, self.high[i], 'bullish'))
        elif market_bias == 'bullish' and recent_lows and self.low[i] < recent_lows[-1][1]:
            self.bos_points.append((i, self.low[i], 'bearish'))

        # FVG detection
        if i > 1:
            gap_size_up = self.low[i] - self.high[i-2]
            if gap_size_up > self.fvg_threshold * self.close[i]:
                self.fvg_areas.append((i-2, i, self.high[i-2], self.low[i], 'bullish'))
            gap_size_down = self.low[i-2] - self.high[i]
            if gap_size_down > self.fvg_threshold * self.close[i]:
                self.fvg_areas.append((i-2, i, self.high[i], self.low[i-2], 'bearish'))

    def place_orders(self):
        """Place orders based on SMC patterns"""
        self.update_market_data()
        self.identify_swing_points()
        self.detect_market_structure()
        
        ticker = self.get_ticker()
        i = len(self.close) - 1
        
        # Check existing trades
        for trade in list(self.current_trades):
            idx, entry_price, direction, stop_loss, take_profit, size = trade
            if (direction == 'long' and ticker['buy'] <= stop_loss) or \
               (direction == 'short' and ticker['sell'] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.capital += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': i, 'entry_price': entry_price, 
                                  'exit_price': stop_loss, 'direction': direction, 'pl': pl, 'result': 'loss'})
                self.current_trades.remove(trade)
                self.loss_count += 1
            elif (direction == 'long' and ticker['sell'] >= take_profit) or \
                 (direction == 'short' and ticker['buy'] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.capital += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': i, 'entry_price': entry_price, 
                                  'exit_price': take_profit, 'direction': direction, 'pl': pl, 'result': 'win'})
                self.current_trades.remove(trade)
                self.win_count += 1

        # New trade setups
        buy_orders = []
        sell_orders = []
        potential_entries = []
        
        for idx, price, type_ in self.choch_points[-1:] + self.bos_points[-1:]:
            if i - idx <= 3:
                if type_ == 'bullish':
                    potential_entries.append((idx, price, 'long', 'CHoCH' if (idx, price, type_) in self.choch_points else 'BOS'))
                else:
                    potential_entries.append((idx, price, 'short', 'CHoCH' if (idx, price, type_) in self.choch_points else 'BOS'))

        if len(self.current_trades) < 3:
            for entry_idx, entry_price, direction, entry_type in potential_entries:
                fvg_confirmed = any(
                    (direction == 'long' and f_type == 'bullish') or 
                    (direction == 'short' and f_type == 'bearish') 
                    for _, _, _, _, f_type in self.fvg_areas[-5:]
                )
                
                if fvg_confirmed or entry_type == 'BOS':
                    stop_dist = (entry_price - min(self.low[-5:])) if direction == 'long' else (max(self.high[-5:]) - entry_price)
                    stop_loss = entry_price - stop_dist * 1.1 if direction == 'long' else entry_price + stop_dist * 1.1
                    take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                    
                    risk_amount = self.capital * self.risk_per_trade
                    size = risk_amount / abs(entry_price - stop_loss)
                    
                    order = {'price': entry_price, 'orderQty': int(size), 'side': 'Buy' if direction == 'long' else 'Sell'}
                    (buy_orders if direction == 'long' else sell_orders).append(order)
                    self.current_trades.append((i, entry_price, direction, stop_loss, take_profit, size))
        
        self.equity_curve.append(self.capital)
        self.converge_orders(buy_orders, sell_orders)

    def create_visualization(self):
        """Create SMC visualization with equity curve"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(self.close, color='black', label='Price')
        for i, h in enumerate(self.high):
            if self.swing_highs[i]:
                ax1.scatter(i, h, color='red', marker='^')
            if self.swing_lows[i]:
                ax1.scatter(i, self.low[i], color='green', marker='v')
        
        for idx, price, type_ in self.choch_points:
            ax1.scatter(idx, price, color='green' if type_ == 'bullish' else 'red', marker='o')
        for idx, price, type_ in self.bos_points:
            ax1.scatter(idx, price, color='green' if type_ == 'bullish' else 'red', marker='s')
        for start, end, min_p, max_p, type_ in self.fvg_areas:
            ax1.fill_between(range(start, end+1), min_p, max_p, 
                           color='lightgreen' if type_ == 'bullish' else 'lightcoral', alpha=0.3)
        
        # Equity curve
        ax2.plot(self.equity_curve, color='green', label='Equity')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Capital ($)')
        ax2.grid(True)
        
        ax1.set_title(f'MatteGreen SMC - Cycle {self.cycle_count + 1}')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend(['Price'])
        
        plt.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        return buffer, fig

    def send_status_update(self):
        """Send visualization via Telegram"""
        buffer, fig = self.create_visualization()
        if buffer:
            win_rate = self.win_count / (self.win_count + self.loss_count) * 100 if (self.win_count + self.loss_count) > 0 else 0
            caption = (f"📈 MatteGreen SMC - Cycle {self.cycle_count + 1}/{self.max_cycles}\n"
                      f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                      f"Balance: ${self.capital:.2f}\n"
                      f"Trades: {len(self.trades)} (Win Rate: {win_rate:.1f}%)")
            self.tg_bot.send_photo(buffer, caption)
            buffer.close()
            plt.close(fig)

    def run_loop(self):
        self.logger.info("🤖 MatteGreen SMC Bot starting...")
        
        while self.cycle_count < self.max_cycles:
            cycle_start = datetime.now()
            self.logger.info(f"▶️ Cycle {self.cycle_count + 1}/{self.max_cycles} starting")
            
            while (datetime.now() - cycle_start).total_seconds() < self.run_duration:
                if not self.check_connection():
                    self.logger.error("❌ Connection lost, restarting...")
                    self.restart()
                
                self.sanity_check()
                self.print_status()
                self.place_orders()
                time.sleep(self.settings.LOOP_INTERVAL)
            
            self.cycle_count += 1
            self.send_status_update()
            
            if self.cycle_count < self.max_cycles:
                self.logger.info(f"⏸️ Break time (Cycle {self.cycle_count}/{self.max_cycles})")
                self.exchange.cancel_all_orders()
                time.sleep(self.break_duration)
                self.logger.info("🔄 Resuming after break")
        
        self.logger.info("🏁 MatteGreen completed all cycles")
        self.send_status_update()
        self.exit()

def run():
    BOT_TOKEN = settings.BOT_TOKEN  # Replace with actual token
    CHAT_ID = settings.CHAT_ID     # Replace with actual chat ID
    
    bot = MatteGreen(BOT_TOKEN, CHAT_ID)
    try:
        bot.run_loop()
    except (KeyboardInterrupt, SystemExit):
        bot.logger.info("🛑 Bot manually stopped")
        bot.exit()
        sys.exit()

if __name__ == "__main__":
    run()
