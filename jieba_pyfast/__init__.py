__version__ = "3.14.0"
__license__ = "MIT"

import logging
import marshal
import os
import re
import sys
import tempfile
import threading
import time
from hashlib import md5

import _jieba_fast_functions_py3 as _jieba_fast_functions

from . import finalseg
from ._compat import get_module_res, resolve_filename, strdecode

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

re_userdict = re.compile(r"^(.+?)( [0-9]+)?( [a-z]+)?$", re.U)

re_eng = re.compile(r"[a-zA-Z0-9]", re.U)

re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
re_skip_default = re.compile(r"(\r\n|\s)", re.U)


class Tokenizer:
    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return "<Tokenizer dictionary=%r>" % self.dictionary

    def gen_pfdict(self, f):
        lfreq = {}
        ltotal = 0
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                word, freq = line.split(" ")[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in range(len(word)):
                    wfrag = word[: ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    "invalid dictionary entry in %s at Line %s: %s"
                    % (f_name, lineno, line)
                )
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            try:
                with DICT_WRITING[abs_path]:
                    pass
            except KeyError:
                pass
            if self.initialized:
                return

            default_logger.debug(
                "Building prefix dict from %s ..."
                % (abs_path or "the default dictionary")
            )
            t1 = time.time()
            if self.cache_file:
                cache_file = self.cache_file
            elif abs_path == DEFAULT_DICT:
                cache_file = "jieba.cache"
            else:
                cache_file = (
                    "jieba.u%s.cache"
                    % md5(abs_path.encode("utf-8", "replace")).hexdigest()
                )
            cache_file = os.path.join(self.tmp_dir or tempfile.gettempdir(), cache_file)
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            if os.path.isfile(cache_file) and (
                abs_path == DEFAULT_DICT
                or os.path.getmtime(cache_file) > os.path.getmtime(abs_path)
            ):
                default_logger.debug("Loading model from cache %s" % cache_file)
                try:
                    with open(cache_file, "rb") as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
                    default_logger.debug("Dumping model to file cache %s" % cache_file)
                    try:
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, "wb") as temp_cache_file:
                            marshal.dump((self.FREQ, self.total), temp_cache_file)
                        os.rename(fpath, cache_file)
                    except Exception:
                        default_logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1)
            )
            default_logger.debug("Prefix dict has been built successfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        else:
            return open(self.dictionary, "rb")

    def __cut_DAG_NO_HMM(self, sentence):
        self.check_initialized()
        route = []
        _jieba_fast_functions._get_DAG_and_calc(
            self.FREQ, sentence, route, float(self.total)
        )
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
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

    def __cut_DAG(self, sentence):
        self.check_initialized()
        route = []
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

    def cut(self, sentence, HMM=True):
        """
        Segment a sentence containing Chinese characters into words.

        Parameter:
            - sentence: The str to be segmented.
            - HMM: Whether to use the Hidden Markov Model.
        """
        sentence = strdecode(sentence)

        re_han = re_han_default
        re_skip = re_skip_default
        if HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in cut_block(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    else:
                        for xx in x:
                            yield xx

    def load_userdict(self, f):
        """
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their occurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        """
        self.check_initialized()
        if isinstance(f, str):
            f_name = f
            f = open(f, "rb")
        else:
            f_name = resolve_filename(f)
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, str):
                try:
                    line = line.decode("utf-8").lstrip("\ufeff")
                except UnicodeDecodeError:
                    raise ValueError("dictionary file %s must be utf-8" % f_name)
            if not line:
                continue
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            self.add_word(word, freq)

    def add_word(self, word, freq=None):
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self._suggest_freq(word)
        self.FREQ[word] = freq
        self.total += freq
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        if freq == 0:
            finalseg.add_force_split(word)

    def _suggest_freq(self, word):
        self.check_initialized()
        ftotal = float(self.total)
        freq = 1
        for seg in self.cut(word, HMM=False):
            freq *= self.FREQ.get(seg, 1) / ftotal
        freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        return freq


# default Tokenizer instance
dt = Tokenizer()

# public API
cut = dt.cut
load_userdict = dt.load_userdict
