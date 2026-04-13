from pathlib import Path

import jieba_pyfast
from jieba_pyfast import Tokenizer

_USERDICT = Path(__file__).parent / "userdict.txt"


def _fresh_tokenizer() -> Tokenizer:
    t = Tokenizer()
    t.load_userdict(_USERDICT)
    return t


def test_load_userdict_recognises_custom_word() -> None:
    t = _fresh_tokenizer()
    words = list(t.cut("云计算方面的专家"))
    assert "云计算" in words


def test_load_userdict_english_word() -> None:
    t = _fresh_tokenizer()
    words = list(t.cut("easy_install is great"))
    assert "easy_install" in words


def test_load_userdict_name() -> None:
    t = _fresh_tokenizer()
    words = list(t.cut("李小福是创新办主任"))
    assert "李小福" in words
    assert "创新办" in words


def test_load_userdict_with_pathlike() -> None:
    t = Tokenizer()
    t.load_userdict(_USERDICT)
    words = list(t.cut("韩玉赏鉴是好的"))
    assert "韩玉赏鉴" in words


def test_load_userdict_reconstructs_input() -> None:
    t = _fresh_tokenizer()
    sentence = "李小福是创新办主任也是云计算方面的专家"
    assert "".join(t.cut(sentence)) == sentence


def test_module_level_load_userdict() -> None:
    jieba_pyfast.load_userdict(_USERDICT)
    words = list(jieba_pyfast.cut("云计算方面的专家"))
    assert "云计算" in words
