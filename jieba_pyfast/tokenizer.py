from hashlib import md5
import json
import marshal
from math import log
import os
from pathlib import Path
import tempfile
from threading import RLock
import time
import re
from os import PathLike
from typing import Generator, Self

from jieba_pyfast import finalseg

# chinese characters, letters, digits, and specific special characters
CHARACTER_PATTERN = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")

# whitespace characters
WHITESPACE_PATTERN = re.compile(r"(\r\n|\s)")












class Tokenizer:
    def __init__(self):
        with Path(__file__).parent.joinpath("dict.json").open("r", encoding="utf-8") as f:
            _data = json.load(f)
            self._freqs = _data["freqs"]
            self._total = _data["total"]

    @classmethod
    def from_file(cls, filepath: PathLike) -> Self:
        tokenizer = cls()
        
        with Path(filepath).open("r", encoding="utf-8") as f:
            for record in json.load(f):
                tokenizer.add_token(record["token"], record["freq"])

        return tokenizer

    def __cut_DAG(self, sentence):
        route = []
        _jieba_fast_functions._get_DAG_and_calc(self.FREQ, sentence, route, float(self.total))
        x = 0
        buf = ""
        N = len(sentence)
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ""
                    else:
                        if not self.FREQ.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ""
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def tokenize(self, text: str) -> list[str]:
        return list(self.cut(text))

    def cut(self, text: str) -> Generator[str, None, None]:
        for block in CHARACTER_PATTERN.split(text):
            if not block:
                continue

            if CHARACTER_PATTERN.match(block):
                for word in self.__cut_DAG(block):
                    yield word
            else:
                for token in WHITESPACE_PATTERN.split(block):
                    if WHITESPACE_PATTERN.match(token):
                        yield token
                    else:
                        for character in token:
                            yield character

    def add_token(self, token: str, freq: int, tag: str):
        self._freqs[token] = freq
        self._total += freq
        
        for i in range(len(token)):
            wfrag = token[:i+1]
            if wfrag not in self._freqs:
                self._freqs[wfrag] = 0
        
        if freq == 0:
            finalseg.add_force_split(token)
