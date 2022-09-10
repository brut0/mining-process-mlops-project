"""
Logger class based on 'loguru'

Works as singleton
"""

# Authors: Sergey Zemskov

import sys

from loguru import logger
from utils.singleton import Singleton


class Logger(metaclass=Singleton):
    created = None

    def __init__(self, filename='log.log', level='TRACE'):
        if not self.created:
            logger.remove()
            logger.add(sys.stderr, format="{time} | {level} | {message}", level=level)
            logger.add(
                f"{filename}",
                format="{time} | {level} | {message}",
                level=level,
                rotation="100 MB",
            )
            self.created = True
