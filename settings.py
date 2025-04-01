# settings.py

# BitMEX API credentials
API_KEY = "your_api_key_here"  # Get from https://testnet.bitmex.com/app/apiKeys or https://www.bitmex.com/app/apiKeys
API_SECRET = "your_api_secret_here"

# Base URL for BitMEX (use testnet for practice, live for real trading)
BASE_URL = "https://testnet.bitmex.com/api/v1/"  # Testnet URL
# BASE_URL = "https://www.bitmex.com/api/v1/"  # Uncomment for live trading

# Trading settings
SYMBOL = "XBTUSD"  # Trading pair, e.g., Bitcoin vs USD
ORDER_SIZE = 100  # Size of each order in contracts
INTERVAL = 1  # Time between orders in seconds
SPREAD = 0.5  # Spread to maintain around the market price

# Dry run mode (set to True to simulate trades without executing them)
DRY_RUN = True

# Logging settings (optional)
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Additional optional settings
MAX_POSITION = 1000  # Maximum position size in contracts
STOP_LOSS = 0.02  # Stop loss as a percentage (e.g., 2%)
TAKE_PROFIT = 0.05  # Take profit as a percentage (e.g., 5%)
