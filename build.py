from setuptools import Extension


def build(setup_kwargs: dict):
    setup_kwargs["ext_modules"] = [
        Extension(
            "_fast",
            sources=["jieba_pyfast/source/_fast.c"],
            extra_compile_args=["-std=c99"]
        )
    ]
