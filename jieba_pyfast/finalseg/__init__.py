import re
from collections.abc import Iterator

import _jieba_fast_functions_py3 as _jieba_fast_functions

from jieba_pyfast.finalseg.prob_emit import P as _emit_P
from jieba_pyfast.finalseg.prob_start import P as _start_P
from jieba_pyfast.finalseg.prob_trans import P as _trans_P

_force_split_words: set[str] = set()

_RE_HAN = re.compile(r"([\u4E00-\u9FD5]+)")
_RE_SKIP = re.compile(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")


def add_force_split(word: str) -> None:
    _force_split_words.add(word)


def _cut_block(sentence: str) -> Iterator[str]:
    _, pos_list = _jieba_fast_functions._viterbi(
        sentence, "BMES", _start_P, _trans_P, _emit_P
    )
    begin = 0
    nexti = 0
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


def cut(sentence: str) -> Iterator[str]:
    for blk in _RE_HAN.split(sentence):
        if _RE_HAN.match(blk):
            for word in _cut_block(blk):
                if word not in _force_split_words:
                    yield word
                else:
                    yield from word
        else:
            for x in _RE_SKIP.split(blk):
                if x:
                    yield x
