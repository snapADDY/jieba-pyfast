# jieba_pyfast

A Chinese text segmentation module with C extensions for speed.

## Installation

```
pip install jieba_pyfast
```

## Usage

```python
import jieba_pyfast as jieba

# Basic segmentation
list(jieba.cut('下雨天留客天留我不留'))
# ['下雨天', '留客', '天留', '我', '不留']

# Load custom dictionary
jieba.load_userdict('userdict.txt')
```
