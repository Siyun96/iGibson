import logging
import os


def get_logger(log_dir, name):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log to file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create log format
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # Add handerlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
