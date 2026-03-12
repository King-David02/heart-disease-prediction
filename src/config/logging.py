import sys
from loguru import logger

def logging_setup():
    logger.remove()
    
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
        level="INFO",
        colorize=True
    )
    
    logger.add(
        "logs/{name}/training/{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
        level="DEBUG",
        colorize=True,
        rotation="10 MB",
        retention="30 days"
    )
    
    return logger