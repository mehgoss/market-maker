import logging
from market_maker.settings import settings
from market_maker.utils.TeleLogBot import configure_logging  # Adjust path as needed

loggers = {}

def setup_custom_logger(name, log_level=settings.LOG_LEVEL, log_file=None, 
                        telegram_token=None, telegram_chat_id=None):
    if loggers.get(name):
        return loggers[name]

    # Use configure_logging from TeleLogBot to set up the logger with Telegram and console
    logger, _ = configure_logging(telegram_token or settings.BOT_TOKEN, 
                                  telegram_chat_id or settings.CHAT_ID)
    
    # Rename the logger to match the requested name
    logger.name = name

    # Optionally adjust log level if specified
    if log_level != settings.LOG_LEVEL:
        logger.setLevel(log_level)

    # Optionally add a FileHandler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    loggers[name] = logger
    return logger
