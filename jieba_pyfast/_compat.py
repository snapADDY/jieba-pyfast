# -*- coding: utf-8 -*-
import os
import sys
import importlib.resources


def get_module_res(*res):
    package = __name__.split('.')[0]
    resource_path = os.path.join(*res)
    return importlib.resources.files(package).joinpath(resource_path).open('rb')


def strdecode(sentence):
    if not isinstance(sentence, str):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)
