from setuptools import Extension

def build(setup_kwargs: dict):
    setup_kwargs["ext_modules"] = [
        Extension('_jieba_fast_functions_py3',
        sources=['jieba_fast/source/jieba_fast_functions_wrap_py3_wrap.c'],
        )
    ]
