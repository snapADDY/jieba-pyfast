# -*- coding: utf-8 -*-
import os
import sys
import importlib.resources

def get_module_res(*res):
    package = __name__.split('.')[0]  # Get the top-level package name
    resource_path = os.path.join(*res)
    try:
        # For Python 3.9+
        return importlib.resources.files(package).joinpath(resource_path).open('rb')
    except (AttributeError, ImportError):
        # Fallback for older Python versions
        try:
            return importlib.resources.open_binary(package, resource_path)
        except (AttributeError, ImportError):
            # Final fallback using file system
            return open(
                os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), *res)),
                "rb",
            )


default_encoding = sys.getfilesystemencoding()

text_type = str
string_types = (str,)
xrange = range

iterkeys = lambda d: iter(d.keys())
itervalues = lambda d: iter(d.values())
iteritems = lambda d: iter(d.items())


def strdecode(sentence):
    if not isinstance(sentence, text_type):
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
