from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "_jieba_fast_functions_py3",
            sources=["jieba_pyfast/source/jieba_fast_functions_wrap_py3_wrap.c"],
        )
    ],
    package_data={
        "jieba_pyfast": ["dict.txt"],
    },
)
