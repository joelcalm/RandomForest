import logging

def mylogger(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    # output goes to console
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # otherwise root's handler is used => as if level was WARNING
    return logger
