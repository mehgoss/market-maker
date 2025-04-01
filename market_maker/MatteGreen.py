import sys
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import mplfinance as mpf
import pandas as pd
from market_maker.settings import settings
from market_maker.market_maker import OrderManager
from market_maker.utils.TeleLogBot import TelegramBot, configure_logging

class MatteGreen(OrderManager):
    def __init__(self, bot_token, chat_id, initial_capital=10000, risk_per_trade=0.02,
                 rr_ratio=2, lookback_period=20, fvg_threshold=0.003):
        # Initialize data storage BEFORE calling super().__init__()
        # This is crucial because OrderManager's __init__ calls reset() which calls place_orders()
        # which calls update_market_data() that uses these attributes
        self.close = []
        self.high = []
        self.low = []
        self.open = []
        self.dates = []
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        
        # Trade tracking
        self.trades = []
        self.current_trades = []
        self.win_count = 0
        self.loss_count = 0
        
        # Now it's safe to call super().__init__()
        super().__init__()
        
        # Telegram setup with error handling
        try:
            self.logger, self.tg_bot = configure_logging(bot_token, chat_id)
        except Exception as e:
            print(f"Warning: Telegram setup failed: {str(e)}")
            # Provide fallback logging
            import logging
            self.logger = logging.getLogger()
            self.tg_bot = None
        
        # Timing parameters
        self.start_time = datetime.now()
        self.total_runtime = 30 * 60
        self.break_duration = 5 * 60
        self.run_duration = 5 * 60
        self.cycle_count = 0
        self.max_cycles = 5

        # SMC parameters
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = rr_ratio
        self.lookback_period = lookback_period
        self.fvg_threshold = fvg_threshold
        
        # Initialize equity curve
        self.equity_curve = [initial_capital]
        
        # Log successful initialization
        self.log_info("MatteGreen initialization complete")

    def log_info(self, message):
        """Safe logging that won't crash if logger is unavailable"""
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.info(message)
            except Exception as e:
                print(f"Logging error: {str(e)}")
                print(f"INFO: {message}")
        else:
            print(f"INFO: {message}")
            
    def log_error(self, message):
        """Safe error logging that won't crash if logger is unavailable"""
        if hasattr(self, 'logger') and self.logger:
            try:
                self.logger.error(message)
            except Exception as e:
                print(f"Logging error: {str(e)}")
                print(f"ERROR: {message}")
        else:
            print(f"ERROR: {message}")

    def profile_market(self):
        """Profile market before trading using historical data"""
        self.log_info("📊 Profiling market before trading...")
        # Fetch historical data (simplified - in reality, use BitMEX API for more data)
        historical_data = []
        for _ in range(self.lookback_period * 2):  # Get 2x lookback period
            try:
                ticker = self.get_ticker()
                historical_data.append({
                    'Date': datetime.now(),
                    'Open': ticker['mid'],
                    'High': ticker['sell'],
                    'Low': ticker['buy'],
                    'Close': ticker['mid']
                })
                time.sleep(1)  # Simulate slower data collection
            except Exception as e:
                self.log_error(f"Error fetching ticker data: {str(e)}")
                time.sleep(2)  # Wait a bit longer on error
        
        if not historical_data:
            self.log_error("Failed to collect historical data")
            return 0, [self.initial_capital]
            
        df = pd.DataFrame(historical_data)
        df.set_index('Date', inplace=True)
        
        # Simulate trades on historical data
        temp_close = df['Close'].tolist()
        temp_high = df['High'].tolist()
        temp_low = df['Low'].tolist()
        temp_trades = []
        temp_equity = [self.initial_capital]
        temp_capital = self.initial_capital
        temp_wins = 0
        temp_losses = 0
        
        for i in range(self.lookback_period, len(temp_close)):
            recent_highs = [(idx, h) for idx, h in enumerate(temp_high[:i+1]) if h == max(temp_high[max(0, idx-3):idx+1])]
            recent_lows = [(idx, l) for idx, l in enumerate(temp_low[:i+1]) if l == min(temp_low[max(0, idx-3):idx+1])]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                entry_price = temp_close[i]
                direction = 'long' if temp_close[i] > temp_close[i-1] else 'short'
                stop_dist = (entry_price - min(temp_low[i-5:i+1])) if direction == 'long' else (max(temp_high[i-5:i+1]) - entry_price)
                stop_loss = entry_price - stop_dist * 1.1 if direction == 'long' else entry_price + stop_dist * 1.1
                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                risk_amount = temp_capital * self.risk_per_trade
                size = risk_amount / abs(entry_price - stop_loss)
                
                # Simulate trade outcome
                for j in range(i+1, len(temp_close)):
                    if direction == 'long':
                        if temp_low[j] <= stop_loss:
                            pl = (stop_loss - entry_price) * size
                            temp_losses += 1
                            break
                        elif temp_high[j] >= take_profit:
                            pl = (take_profit - entry_price) * size
                            temp_wins += 1
                            break
                    else:
                        if temp_high[j] >= stop_loss:
                            pl = (entry_price - stop_loss) * size
                            temp_losses += 1
                            break
                        elif temp_low[j] <= take_profit:
                            pl = (entry_price - take_profit) * size
                            temp_wins += 1
                            break
                else:
                    pl = (temp_close[-1] - entry_price) * size if direction == 'long' else (entry_price - temp_close[-1]) * size
                
                temp_capital += pl
                temp_trades.append({'pl': pl, 'result': 'win' if pl > 0 else 'loss'})
                temp_equity.append(temp_capital)
        
        win_rate = temp_wins / (temp_wins + temp_losses) * 100 if (temp_wins + temp_losses) > 0 else 0
        self.log_info(f"📈 Profile Results: Win Rate: {win_rate:.1f}%, Simulated Trades: {len(temp_trades)}")
        return win_rate, temp_equity

    def update_market_data(self):
        try:
            ticker = self.get_ticker()
            
            # Ensure all lists are initialized
            if not hasattr(self, 'close') or self.close is None:
                self.close = []
            if not hasattr(self, 'high') or self.high is None:
                self.high = []
            if not hasattr(self, 'low') or self.low is None:
                self.low = []
            if not hasattr(self, 'open') or self.open is None:
                self.open = []
            if not hasattr(self, 'dates') or self.dates is None:
                self.dates = []
            if not hasattr(self, 'swing_highs') or self.swing_highs is None:
                self.swing_highs = []
            if not hasattr(self, 'swing_lows') or self.swing_lows is None:
                self.swing_lows = []
                
            self.close.append(ticker['mid'])
            self.high.append(ticker['sell'])
            self.low.append(ticker['buy'])
            self.open.append(self.close[-1] if not self.open else self.close[-2])
            self.dates.append(datetime.now())
            
            if len(self.close) > self.lookback_period + 10:
                self.close.pop(0)
                self.high.pop(0)
                self.low.pop(0)
                self.open.pop(0)
                self.dates.pop(0)
                self.swing_highs.pop(0) if self.swing_highs else None
                self.swing_lows.pop(0) if self.swing_lows else None
        except Exception as e:
            self.log_error(f"Error in update_market_data: {str(e)}")
            # Add dummy data if needed to prevent further errors
            if not self.close:
                ticker = self.get_ticker()
                self.close = [ticker['mid']]
                self.high = [ticker['sell']]
                self.low = [ticker['buy']]
                self.open = [ticker['mid']]
                self.dates = [datetime.now()]
                self.swing_highs = [0]
                self.swing_lows = [0]

    def identify_swing_points(self):
        try:
            if len(self.close) < 5:
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
        except Exception as e:
            self.log_error(f"Error in identify_swing_points: {str(e)}")
            self.swing_highs.append(0)
            self.swing_lows.append(0)

    def detect_market_structure(self):
        if not hasattr(self, 'choch_points'):
            self.choch_points = []
        if not hasattr(self, 'bos_points'):
            self.bos_points = []
        if not hasattr(self, 'fvg_areas'):
            self.fvg_areas = []
            
        if len(self.close) < self.lookback_period:
            return
        
        try:
            i = len(self.close) - 1
            recent_highs = [(idx, h) for idx, h in enumerate(self.high) if idx < len(self.swing_highs) and self.swing_highs[idx]]
            recent_lows = [(idx, l) for idx, l in enumerate(self.low) if idx < len(self.swing_lows) and self.swing_lows[idx]]
            
            market_bias = 'neutral' if not self.choch_points else self.choch_points[-1][2]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (market_bias in ['bullish', 'neutral'] and 
                    recent_highs[-1][1] < recent_highs[-2][1] and 
                    recent_lows[-1][1] < recent_lows[-2][1]):
                    self.choch_points.append((i, self.close[i], 'bearish'))
                elif (market_bias in ['bearish', 'neutral'] and 
                      recent_lows[-1][1] > recent_lows[-2][1] and 
                      recent_highs[-1][1] > recent_highs[-2][1]):
                    self.choch_points.append((i, self.close[i], 'bullish'))
            
            if market_bias == 'bearish' and recent_highs and self.high[i] > recent_highs[-1][1]:
                self.bos_points.append((i, self.high[i], 'bullish'))
            elif market_bias == 'bullish' and recent_lows and self.low[i] < recent_lows[-1][1]:
                self.bos_points.append((i, self.low[i], 'bearish'))
            
            if i > 1:
                if self.low[i] - self.high[i-2] > self.fvg_threshold * self.close[i]:
                    self.fvg_areas.append((i-2, i, self.high[i-2], self.low[i], 'bullish'))
                if self.low[i-2] - self.high[i] > self.fvg_threshold * self.close[i]:
                    self.fvg_areas.append((i-2, i, self.high[i], self.low[i-2], 'bearish'))
        except Exception as e:
            self.log_error(f"Error in detect_market_structure: {str(e)}")

    def place_orders(self):
        try:
            self.update_market_data()
            self.identify_swing_points()
            self.detect_market_structure()
            
            ticker = self.get_ticker()
            i = len(self.close) - 1
            
            if not hasattr(self, 'current_trades') or self.current_trades is None:
                self.current_trades = []
            if not hasattr(self, 'trades') or self.trades is None:
                self.trades = []
            
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
            
            buy_orders = []
            sell_orders = []
            potential_entries = []
            
            # Check that we have choch_points and bos_points before accessing them
            if hasattr(self, 'choch_points') and self.choch_points and hasattr(self, 'bos_points') and self.bos_points:
                for idx, price, type_ in self.choch_points[-1:] + self.bos_points[-1:]:
                    if i - idx <= 3:
                        potential_entries.append((idx, price, 'long' if type_ == 'bullish' else 'short', 
                                               'CHoCH' if (idx, price, type_) in self.choch_points else 'BOS'))
            
            if len(self.current_trades) < 3:
                for entry_idx, entry_price, direction, entry_type in potential_entries:
                    fvg_confirmed = False
                    if hasattr(self, 'fvg_areas') and self.fvg_areas:
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
            
            # Ensure equity_curve is initialized
            if not hasattr(self, 'equity_curve') or self.equity_curve is None:
                self.equity_curve = [self.capital]
            else:
                self.equity_curve.append(self.capital)
                
            self.converge_orders(buy_orders, sell_orders)
        except Exception as e:
            self.log_error(f"Error in place_orders: {str(e)}")

    def create_visualization(self, profile_win_rate, profile_equity):
        """Create visualization with mplfinance candlestick and profile stats"""
        try:
            if not self.dates or not self.open or not self.high or not self.low or not self.close:
                self.log_error("Cannot create visualization: missing market data")
                return None, None
                
            df = pd.DataFrame({
                'Date': self.dates,
                'Open': self.open,
                'High': self.high,
                'Low': self.low,
                'Close': self.close
            }).set_index('Date')
            
            fig = plt.figure(figsize=(12, 10))
            ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
            
            # Candlestick chart
            mpf.plot(df, type='candle', ax=ax1, style='charles', 
                    addplot=[
                        mpf.make_addplot([h if sh else np.nan for h, sh in zip(self.high, self.swing_highs)], 
                                       type='scatter', markersize=50, marker='^', color='red'),
                        mpf.make_addplot([l if sl else np.nan for l, sl in zip(self.low, self.swing_lows)], 
                                       type='scatter', markersize=50, marker='v', color='green')
                    ])
            
            if hasattr(self, 'choch_points') and self.choch_points:
                for idx, price, type_ in self.choch_points:
                    if 0 <= idx < len(df):
                        ax1.scatter(idx, price, color='green' if type_ == 'bullish' else 'red', marker='o', s=100)
            
            if hasattr(self, 'bos_points') and self.bos_points:
                for idx, price, type_ in self.bos_points:
                    if 0 <= idx < len(df):
                        ax1.scatter(idx, price, color='green' if type_ == 'bullish' else 'red', marker='s', s=100)
            
            if hasattr(self, 'fvg_areas') and self.fvg_areas:
                for start, end, min_p, max_p, type_ in self.fvg_areas:
                    if 0 <= start < len(df) and 0 <= end < len(df):
                        ax1.fill_between(range(start, end+1), min_p, max_p, 
                                      color='lightgreen' if type_ == 'bullish' else 'lightcoral', alpha=0.3)
            
            # Equity curve
            ax2.plot(self.equity_curve, color='green', label='Live Equity')
            ax2.plot(range(len(profile_equity)), profile_equity, color='blue', label='Profile Equity', alpha=0.5)
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Capital ($)')
            ax2.grid(True)
            ax2.legend()
            
            # Add profile stats
            live_win_rate = self.win_count / (self.win_count + self.loss_count) * 100 if (self.win_count + self.loss_count) > 0 else 0
            plt.figtext(0.1, 0.05, 
                       f"Profile Win Rate: {profile_win_rate:.1f}%\n"
                       f"Live Win Rate: {live_win_rate:.1f}%\n"
                       f"Total Trades: {len(self.trades)}",
                       ha="left", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
            
            ax1.set_title(f'MatteGreen SMC - Cycle {self.cycle_count + 1}')
            plt.tight_layout()
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            return buffer, fig
        except Exception as e:
            self.log_error(f"Error creating visualization: {str(e)}")
            return None, None

    def send_status_update(self, profile_win_rate, profile_equity):
        try:
            buffer, fig = self.create_visualization(profile_win_rate, profile_equity)
            if buffer and hasattr(self, 'tg_bot') and self.tg_bot:
                win_rate = self.win_count / (self.win_count + self.loss_count) * 100 if (self.win_count + self.loss_count) > 0 else 0
                caption = (f"📈 MatteGreen SMC - Cycle {self.cycle_count + 1}/{self.max_cycles}\n"
                          f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                          f"Balance: ${self.capital:.2f}\n"
                          f"Trades: {len(self.trades)} (Live Win Rate: {win_rate:.1f}%)")
                try:
                    self.tg_bot.send_photo(buffer, caption)
                except Exception as e:
                    self.log_error(f"Telegram error sending photo: {str(e)}")
                    # Try text-only update as fallback
                    try:
                        self.tg_bot.send_message(caption)
                    except:
                        pass
                buffer.close()
                plt.close(fig)
            elif buffer:
                buffer.close()
                if fig:
                    plt.close(fig)
        except Exception as e:
            self.log_error(f"Error in send_status_update: {str(e)}")

    def run_loop(self):
        self.log_info("🤖 MatteGreen SMC Bot starting...")
        try:
            profile_win_rate, profile_equity = self.profile_market()
            
            while self.cycle_count < self.max_cycles:
                cycle_start = datetime.now()
                self.log_info(f"▶️ Cycle {self.cycle_count + 1}/{self.max_cycles} starting")
                
                while (datetime.now() - cycle_start).total_seconds() < self.run_duration:
                    try:
                        if not self.check_connection():
                            self.log_error("❌ Connection lost, restarting...")
                            try:
                                self.restart()
                            except Exception as e:
                                self.log_error(f"Error during restart: {str(e)}")
                            break
                        
                        self.sanity_check()
                        self.print_status()
                        self.place_orders()
                    except Exception as e:
                        self.log_error(f"Error in main loop: {str(e)}")
                    
                    try:
                        time.sleep(self.settings.LOOP_INTERVAL)
                    except (AttributeError, TypeError):
                        # Fallback if settings.LOOP_INTERVAL is not available
                        time.sleep(5)
                
                self.cycle_count += 1
                self.send_status_update(profile_win_rate, profile_equity)
                
                if self.cycle_count < self.max_cycles:
                    self.log_info(f"⏸️ Break time (Cycle {self.cycle_count}/{self.max_cycles})")
                    try:
                        self.exchange.cancel_all_orders()
                    except Exception as e:
                        self.log_error(f"Error cancelling orders: {str(e)}")
                    time.sleep(self.break_duration)
                    self.log_info("🔄 Resuming after break")
            
            self.log_info("🏁 MatteGreen completed all cycles")
            self.send_status_update(profile_win_rate, profile_equity)
        except Exception as e:
            self.log_error(f"Critical error in run_loop: {str(e)}")
        finally:
            try:
                self.exit()
            except Exception as e:
                self.log_error(f"Error during exit: {str(e)}")
                sys.exit()

def run():
    # Add error handling for settings
    try:
        BOT_TOKEN = settings.BOT_TOKEN  # Replace with actual token
        CHAT_ID = settings.CHAT_ID     # Replace with actual chat ID
    except AttributeError:
        print("Warning: BOT_TOKEN or CHAT_ID not found in settings. Using empty values.")
        BOT_TOKEN = ""
        CHAT_ID = ""
    
    try:
        bot = MatteGreen(BOT_TOKEN, CHAT_ID)
        bot.run_loop()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Bot manually stopped")
        try:
            bot.exit()
        except:
            pass
        sys.exit()
    except Exception as e:
        print(f"Critical error initializing bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run()
