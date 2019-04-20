import sys


int_types = (int,)
if sys.version_info < (3,):
    int_types += (long,)  # noqa


def extend_with_original_traceback(exc, trace):
    args = exc.args
    arg0 = args[0] if args else ""
    arg0 = "%s\n\nOriginally defined at:\n%s" % (arg0, "".join(trace))
    exc.args = (arg0,) + args[1:]
    return exc


class SimpleObject(object):

    def __init__(self, kwargs):
        self.__dict__.update(kwargs)


def convert_to_str(s):
    if sys.version_info >= (3,):
        if isinstance(s, bytes):
            return s.decode("utf-8")
    return s
