import inspect


def klass_fullname(klass):
    if not inspect.isclass(klass):
        klass = klass.__class__

    module = klass.__module__
    if module is None or module == str.__module__:
        return klass.__name__
    else:
        return module + '.' + klass.__name__