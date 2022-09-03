"""
Logger class based on 'loguru'

Works as singleton
"""

# Authors: Sergey Zemskov

import sys

from loguru import logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    created = None

    def __init__(self, filename='log.log'):
        if not self.created:
            logger.remove()
            logger.add(sys.stderr, format="{time} | {level} | {message}")
            logger.add(
                f"{filename}",
                format="{time} | {level} | {message}",
                level='TRACE',
                rotation="100 MB",
            )
            self.created = True
