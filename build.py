from setuptools import Extension


def build(setup_kwargs: dict):
    setup_kwargs["ext_modules"] = [
        Extension(
            "jieba_pyfast._fast",
            sources=["jieba_pyfast/_core/fast.c"],
            extra_compile_args=["-std=c99"],
        )
    ]
