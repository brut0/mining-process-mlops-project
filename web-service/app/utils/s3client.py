"""
Class to access S3 using boto3

Works as singleton
"""

# Authors: Sergey Zemskov

import os

import boto3
from dotenv import load_dotenv

from utils.singleton import Singleton


class S3Client(metaclass=Singleton):
    '''S3 client with parameters from environment'''

    _created = None
    _s3 = None

    def __init__(self, host='yandex'):
        if not self._created or not self._s3:
            load_dotenv(override=True)  # load environment variables from .env
            self._s3 = self.__create_client(host=host)
            self._created = True

    def __create_client(self, host):
        if host == 'yandex':
            return self.__init_yandex()
        return None

    def __init_yandex(self):
        session = boto3.session.Session()
        return session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            region_name=os.environ['AWS_REGION'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        )

    def reinit(self, host):
        '''Reinitialise client'''
        if self._created:
            self._s3 = self.__create_client(host=host)

    @property
    def client(self):
        '''Get S3 client'''
        return self._s3

    @client.setter
    def client(self, value):
        self._s3 = value

    @client.deleter
    def client(self):
        self._created = None
        self._s3 = None
