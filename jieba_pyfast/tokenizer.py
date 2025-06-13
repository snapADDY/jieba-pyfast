import json
import re
from os import PathLike
from pathlib import Path
from typing import Generator, Self
from jieba_pyfast.probs import START_PROB, TRANS_PROB, EMIT_PROB
from jieba_pyfast import _fast

# chinese characters, letters, digits, and specific special characters
CHARACTER_PATTERN = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")

# whitespace characters
WHITESPACE_PATTERN = re.compile(r"(\r\n|\s)")

# regex patterns for Chinese characters and alphanumeric tokens
CHINESE_PATTERN = re.compile(r"([\u4E00-\u9FD5]+)")

# regex pattern to skip alphanumeric tokens with optional percentage sign
SKIP_PATTERN = re.compile(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")


class Tokenizer:
    def __init__(self):
        with Path(__file__).parent.joinpath("dict.json").open("r", encoding="utf-8") as f:
            _data = json.load(f)
            self._freqs = _data["freqs"]
            self._total = _data["total"]

        self._force_split = set()

    @classmethod
    def from_file(cls, filepath: PathLike) -> Self:
        tokenizer = cls()
        
        with Path(filepath).open("r", encoding="utf-8") as f:
            for record in json.load(f):
                tokenizer.add_token(record["token"], record["freq"])

        return tokenizer
    
    def add_token(self, token: str, freq: int, tag: str):
        self._freqs[token] = freq
        self._total += freq
        
        for i in range(len(token)):
            wfrag = token[:i+1]
            if wfrag not in self._freqs:
                self._freqs[wfrag] = 0
        
        if freq == 0:
            self._force_split.add(token)

    def tokenize(self, text: str) -> list[str]:
        return list(self.cut(text))

    def cut(self, text: str) -> Generator[str, None, None]:
        for block in CHARACTER_PATTERN.split(text):
            if not block:
                continue

            if CHARACTER_PATTERN.match(block):
                for word in self._cut_dag(block):
                    yield word
            else:
                for token in WHITESPACE_PATTERN.split(block):
                    if WHITESPACE_PATTERN.match(token):
                        yield token
                    else:
                        for character in token:
                            yield character

    def _cut_dag(self, text: str):
        route = []
        _fast.get_dag_and_calc(self._freqs, text, route, float(self._total))
        x = 0
        buf = ""
        N = len(text)
        while x < N:
            y = route[x] + 1
            l_word = text[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ""
                    else:
                        if not self._freqs.get(buf):
                            recognized = self._cut(buf)
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
            elif not self._freqs.get(buf):
                recognized = self._cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def _cut(self, text: str) -> Generator[str, None, None]:
        for blk in CHINESE_PATTERN.split(text):
            if CHINESE_PATTERN.match(blk):
                for word in self._cut_viterbi(blk):
                    if word not in self._force_split:
                        yield word
                    else:
                        for c in word:
                            yield c
            else:
                tmp = SKIP_PATTERN.split(blk)
                for x in tmp:
                    if x:
                        yield x

    def _cut_viterbi(self, sentence):
        _, pos_list = _fast.viterbi(sentence, "BMES", START_PROB, TRANS_PROB, EMIT_PROB)
        begin, nexti = 0, 0
        for i, char in enumerate(sentence):
            pos = pos_list[i]
            if pos == "B":
                begin = i
            elif pos == "E":
                yield sentence[begin : i + 1]
                nexti = i + 1
            elif pos == "S":
                yield char
                nexti = i + 1
        if nexti < len(sentence):
            yield sentence[nexti:]
