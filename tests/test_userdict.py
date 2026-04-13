#encoding=utf-8
import os
import sys
sys.path.append("../")
import jieba_pyfast as jieba

_dir = os.path.dirname(os.path.abspath(__file__))

test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带\u201c韩玉赏鉴\u201d的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)

jieba.load_userdict(os.path.join(_dir, "userdict.txt"))

words = jieba.cut(test_sent)
print('/'.join(words))

print("="*40)

terms = jieba.cut('easy_install is great')
print('/'.join(terms))
terms = jieba.cut('python 的正则表达式是好用的')
print('/'.join(terms))

print("="*40)
