# jieba_pyfast

A Chinese text segmentation module based on `jieba_fast`, with wheels for Python 3.11, 3.12 and 3.13.

## Installation

You can install the latest stable version via:

```
$ pip install jieba_pyfast
```

## Main Functions

For details, see https://github.com/fxsjy/jieba

## Usage

```python
>>> from jieba_pyfast as jieba
>>> jieba.lcut('下雨天留客天留我不留')
['下雨天', '留客', '天留', '我', '不留']
```
