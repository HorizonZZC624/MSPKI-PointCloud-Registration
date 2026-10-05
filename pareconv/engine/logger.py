import logging
import os

try:
    import coloredlogs
except ImportError:
    coloredlogs = None


def create_logger(log_file=None):
    logger = logging.getLogger(f'pareconv.{id(log_file)}')
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'
    stream_handler = logging.StreamHandler()
    if coloredlogs is not None:
        stream_formatter = coloredlogs.ColoredFormatter(format_str)
    else:
        stream_formatter = logging.Formatter(
            format_str, datefmt='%Y-%m-%d %H:%M:%S'
        )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        parent = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(file_handler)
    return logger


class Logger:
    def __init__(self, log_file=None, local_rank=-1):
        self.logger = (
            create_logger(log_file=log_file)
            if local_rank in (0, -1)
            else None
        )

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message):
        if self.logger is not None:
            self.logger.critical(message)
