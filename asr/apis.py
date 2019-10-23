import json
import os
import threading
import time
from functools import wraps

import speech_recognition as sr


class BaseCredentials:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


class GoogleCloudCredientials(BaseCredentials):
    def __init__(self, credentials=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)):
        super().__init__()

        self.credentials = credentials
        if self.credentials and os.path.isfile(self.credentials):
            with open(self.credentials, 'r') as f:
                self.credentials = json.dumps(json.load(f))

    def __call__(self):
        return {'credentials_json': self.credentials}

    @property
    def name(self):
        return 'Google Cloud Speech'


class MicrosoftBingCredientials(BaseCredentials):
    def __init__(self, key=os.environ.get('BING_KEY', None)):
        super().__init__()

        self.key = key

    def __call__(self):
        return {'key': self.key}

    @property
    def name(self):
        return 'Microsoft Bing Voice Recognition'


class IBMCredientials(BaseCredentials):
    def __init__(self, username=os.environ.get('IBM_USERNAME', None), password=os.environ.get('IBM_PASSWORD', None)):
        super().__init__()

        self.username = username
        self.password = password

    def __call__(self):
        return {'username': self.username, 'password': self.password}

    @property
    def name(self):
        return 'IBM Speech to Text'


def rate_limited(max_per_second):
    """Rate-limits the decorated function locally, for one process.
    from: https://gist.github.com/gregburek/1441055 """
    lock = threading.Lock()
    min_interval = 1.0 / max_per_second

    def decorate(func):
        last_time_called = time.perf_counter()

        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            lock.acquire()
            nonlocal last_time_called
            try:
                elapsed = time.perf_counter() - last_time_called
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)

                return func(*args, **kwargs)
            finally:
                last_time_called = time.perf_counter()
                lock.release()

        return rate_limited_function

    return decorate


class SpeechRecognitionAPI:
    def __init__(self, api='gcp', lang='pt-BR', **kwargs):
        self._r = sr.Recognizer()
        self.lang = lang

        if api == 'gcp':
            self.credentials = GoogleCloudCredientials(**kwargs)
            self._recognize = self._r.recognize_google_cloud
        elif api == 'bing':
            self.credentials = MicrosoftBingCredientials(**kwargs)
            self._recognize = self._r.recognize_bing
        elif api == 'ibm':
            self.credentials = IBMCredientials(**kwargs)
            self._recognize = self._r.recognize_ibm

    @rate_limited(5)
    def recognize(self, audio, safe=True):

        if not isinstance(audio, sr.AudioData):
            with sr.AudioFile(audio) as source:
                audio = self._r.record(source)
        try:
            return self._recognize(audio, language=self.lang, **self.credentials())
        except sr.UnknownValueError as e:
            if not safe:
                raise e
            return "{} could not understand audio".format(self.credentials.name)
        except sr.RequestError as e:
            if not safe:
                raise e
            return "Could not request results from {} service; {}".format(self.credentials.name, e)
