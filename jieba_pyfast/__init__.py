import importlib.resources
import marshal
import os
import re
import tempfile
import threading
from collections.abc import Generator, Iterator
from hashlib import md5
from pathlib import Path
from typing import IO, BinaryIO

import _jieba_fast_functions_py3 as _jieba_fast_functions

from jieba_pyfast import finalseg

_DEFAULT_DICT_NAME = "dict.txt"

_RE_USERDICT = re.compile(r"^(.+?)( [0-9]+)?( [a-z]+)?$", re.U)
_RE_ENG = re.compile(r"[a-zA-Z0-9]", re.U)
_RE_HAN = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
_RE_SKIP = re.compile(r"(\r\n|\s)", re.U)

_dict_writing: dict[str | None, threading.RLock] = {}


def _get_module_res(name: str) -> BinaryIO:
    return importlib.resources.files(__package__).joinpath(name).open("rb")  # type: ignore[return-value]


class Tokenizer:
    def __init__(self, dictionary: str | None = None) -> None:
        self.lock = threading.RLock()
        self.dictionary: str | None = (
            str(Path(dictionary).resolve()) if dictionary else None
        )
        self.FREQ: dict[str, int] = {}
        self.total: int = 0
        self.initialized: bool = False
        self.tmp_dir: str | None = None
        self.cache_file: str | None = None

    def __repr__(self) -> str:
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    def _gen_pfdict(self, f: BinaryIO) -> tuple[dict[str, int], int]:
        lfreq: dict[str, int] = {}
        ltotal = 0
        for lineno, raw in enumerate(f, 1):
            try:
                line = raw.strip().decode("utf-8")
                word, freq_s = line.split(" ")[:2]
                freq = int(freq_s)
                lfreq[word] = freq
                ltotal += freq
                for ch in range(len(word)):
                    wfrag = word[: ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    f"invalid dictionary entry in {getattr(f, 'name', repr(f))} "
                    f"at line {lineno}: {raw!r}"
                )
        f.close()
        return lfreq, ltotal

    def _initialize(self, dictionary: str | None = None) -> None:
        if dictionary:
            abs_path = str(Path(dictionary).resolve())
            if self.dictionary == abs_path and self.initialized:
                return
            self.dictionary = abs_path
            self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            if abs_path in _dict_writing:
                with _dict_writing[abs_path]:
                    pass
            if self.initialized:
                return

            if self.cache_file:
                cache_file = self.cache_file
            elif abs_path is None:
                cache_file = "jieba.cache"
            else:
                cache_file = "jieba.u%s.cache" % md5(
                    abs_path.encode("utf-8", "replace")
                ).hexdigest()

            cache_file = os.path.join(
                self.tmp_dir or tempfile.gettempdir(), cache_file
            )
            tmpdir = os.path.dirname(cache_file)

            loaded = False
            if os.path.isfile(cache_file) and (
                abs_path is None
                or os.path.getmtime(cache_file) > os.path.getmtime(abs_path)
            ):
                try:
                    with open(cache_file, "rb") as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    loaded = True
                except Exception:
                    loaded = False

            if not loaded:
                wlock = _dict_writing.setdefault(abs_path, threading.RLock())
                with wlock:
                    self.FREQ, self.total = self._gen_pfdict(self._get_dict_file())
                    try:
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, "wb") as temp_cache_file:
                            marshal.dump(
                                (self.FREQ, self.total), temp_cache_file
                            )
                        os.rename(fpath, cache_file)
                    except Exception:
                        print("Dump cache file failed.")

                _dict_writing.pop(abs_path, None)

            self.initialized = True

    def _ensure_initialized(self) -> None:
        if not self.initialized:
            self._initialize()

    def _get_dict_file(self) -> BinaryIO:
        if self.dictionary is None:
            return _get_module_res(_DEFAULT_DICT_NAME)
        return open(self.dictionary, "rb")

    def _cut_dag_no_hmm(self, sentence: str) -> Iterator[str]:
        self._ensure_initialized()
        route: list[int] = []
        _jieba_fast_functions._get_DAG_and_calc(
            self.FREQ, sentence, route, float(self.total)
        )
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if _RE_ENG.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ""
                yield l_word
                x = y
        if buf:
            yield buf

    def _cut_dag(self, sentence: str) -> Iterator[str]:
        self._ensure_initialized()
        route: list[int] = []
        _jieba_fast_functions._get_DAG_and_calc(
            self.FREQ, sentence, route, float(self.total)
        )
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
                    elif not self.FREQ.get(buf):
                        yield from finalseg.cut(buf)
                    else:
                        yield from buf
                    buf = ""
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                yield from finalseg.cut(buf)
            else:
                yield from buf

    def cut(self, sentence: str | bytes, *, HMM: bool = True) -> Generator[str]:
        """Segment a sentence containing Chinese characters into words.

        Args:
            sentence: The string to be segmented.
            HMM: Whether to use the Hidden Markov Model.

        Yields:
            Individual word segments.
        """
        if isinstance(sentence, bytes):
            try:
                sentence = sentence.decode("utf-8")
            except UnicodeDecodeError:
                sentence = sentence.decode("gbk", "ignore")

        cut_block = self._cut_dag if HMM else self._cut_dag_no_hmm
        for blk in _RE_HAN.split(sentence):
            if not blk:
                continue
            if _RE_HAN.match(blk):
                yield from cut_block(blk)
            else:
                for x in _RE_SKIP.split(blk):
                    if _RE_SKIP.match(x):
                        yield x
                    else:
                        yield from x

    def load_userdict(self, f: str | os.PathLike[str] | IO[bytes]) -> None:
        """Load a user dictionary to improve segmentation.

        Args:
            f: Path to a UTF-8 dictionary file, or a binary file-like object.
               Each line: ``word [freq [word_type]]``
        """
        self._ensure_initialized()
        if isinstance(f, (str, os.PathLike)):
            f_name = str(f)
            f = open(f, "rb")  # noqa: SIM115
        else:
            f_name = getattr(f, "name", repr(f))
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8").lstrip("\ufeff")
                except UnicodeDecodeError:
                    raise ValueError(
                        f"dictionary file {f_name} must be utf-8"
                    ) from None
            if not line:
                continue
            m = _RE_USERDICT.match(line)
            if not m:
                continue
            word, freq_s, _ = m.groups()
            freq = freq_s.strip() if freq_s is not None else None
            self._add_word(word, freq)

    def _add_word(self, word: str, freq: str | int | None = None) -> None:
        self._ensure_initialized()
        if isinstance(word, bytes):
            word = word.decode("utf-8")
        resolved_freq = int(freq) if freq is not None else self._suggest_freq(word)
        self.FREQ[word] = resolved_freq
        self.total += resolved_freq
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if resolved_freq == 0:
            finalseg.add_force_split(word)

    def _suggest_freq(self, word: str) -> int:
        self._ensure_initialized()
        ftotal = float(self.total)
        freq = 1.0
        for seg in self.cut(word, HMM=False):
            freq *= self.FREQ.get(seg, 1) / ftotal
        return max(int(freq * self.total) + 1, self.FREQ.get(word, 1))


# default Tokenizer instance
_dt = Tokenizer()

# public API
cut = _dt.cut
load_userdict = _dt.load_userdict
