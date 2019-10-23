import json
import logging
import os

import requests
from tqdm import tqdm

import sys
from asr.common.tee_logger import TeeLogger

logger = logging.getLogger(__name__)


def prepare_global_logging(serialization_dir, local_rank, verbosity=0):
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Params:
        serializezation_dir : ``str``, required.
            The directory to stream logs to.
    """

    std_out_file = os.path.join(serialization_dir, f"stdout.{local_rank}.log")

    sys.stdout = TeeLogger(
        std_out_file,  # type: ignore
        sys.stdout,
        local_rank=local_rank)
    sys.stderr = TeeLogger(
        os.path.join(serialization_dir,
                     f"stderr.{local_rank}.log"),  # type: ignore
        sys.stderr,
        local_rank=local_rank)

    fmt = '%(levelname).1s: %(message)s'
    datefmt = '%Y-%m-%d %I:%M:%S.%p'

    verbosity = 1

    if local_rank != 0:
        verbosity -= 1

    level = logging.WARNING
    if verbosity > 0:
        level = logging.INFO
    if verbosity > 1:
        level = logging.DEBUG

    logging.basicConfig(format=fmt, datefmt=datefmt, level=level)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(stdout_handler)


def write_labels(labels, filepath):
    """ Write labels to file
    """
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(labels, f, indent=4)


def expand_values(obj, **kwargs):

    if isinstance(obj, str):
        return obj.format(**kwargs)

    if isinstance(obj, (list, tuple)):
        for i, l in enumerate(obj):
            obj[i] = expand_values(l, **kwargs)

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = expand_values(v, **kwargs)

    return obj


def _reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""

    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        with open(path, "wb") as file:
            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=1,
                      desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if 'drive.google.com' not in url:
        response = requests.get(url,
                                headers={'User-Agent': 'Mozilla/5.0'},
                                stream=True)
        process_response(response)
        return

    logger.info('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)


# Expand paths containing shell variable substitutions.
# The following rules apply:
#       - no expansion within single quotes
#       - '$$' is translated into '$'
#       - '%%' is translated into '%' if '%%' are not seen in %var1%%var2%
#       - ${varname} is accepted.
#       - $varname is accepted.
#       - %varname% is accepted.
#       - varnames can be made out of letters, digits and the characters '_-'
#         (though is not verified in the ${varname} and %varname% cases)
# XXX With COMMAND.COM you can use any characters in a variable name,
# XXX except '^|<>='.
def expandvars(path, environ=None):
    """Expand shell variables of the forms $var, ${var} and %var%.
    Unknown variables are left unchanged."""
    path = os.fspath(path)
    if isinstance(path, bytes):
        if b'$' not in path and b'%' not in path:
            return path
        import string
        varchars = bytes(string.ascii_letters + string.digits + '_-', 'ascii')
        quote = b'\''
        percent = b'%'
        brace = b'{'
        rbrace = b'}'
        dollar = b'$'
        environ = environ or getattr(os, 'environb', None)
    else:
        if '$' not in path and '%' not in path:
            return path
        import string
        varchars = string.ascii_letters + string.digits + '_-'
        quote = '\''
        percent = '%'
        brace = '{'
        rbrace = '}'
        dollar = '$'
        environ = environ or os.environ
    res = path[:0]
    index = 0
    pathlen = len(path)
    while index < pathlen:
        c = path[index:index + 1]
        if c == quote:  # no expansion within single quotes
            path = path[index + 1:]
            pathlen = len(path)
            try:
                index = path.index(c)
                res += c + path[:index + 1]
            except ValueError:
                res += c + path
                index = pathlen - 1
        elif c == percent:  # variable or '%'
            if path[index + 1:index + 2] == percent:
                res += c
                index += 1
            else:
                path = path[index + 1:]
                pathlen = len(path)
                try:
                    index = path.index(percent)
                except ValueError:
                    res += percent + path
                    index = pathlen - 1
                else:
                    var = path[:index]
                    try:
                        value = environ[var]
                    except KeyError:
                        value = percent + var + percent
                    res += value
        elif c == dollar:  # variable or '$$'
            if path[index + 1:index + 2] == dollar:
                res += c
                index += 1
            elif path[index + 1:index + 2] == brace:
                path = path[index + 2:]
                pathlen = len(path)
                try:
                    index = path.index(rbrace)
                except ValueError:
                    res += dollar + brace + path
                    index = pathlen - 1
                else:
                    var = path[:index]
                    try:
                        value = environ[var]
                    except KeyError:
                        value = dollar + brace + var + rbrace
                    res += value
            else:
                var = path[:0]
                index += 1
                c = path[index:index + 1]
                while c and c in varchars:
                    var += c
                    index += 1
                    c = path[index:index + 1]
                try:
                    value = environ[var]
                except KeyError:
                    value = dollar + var
                res += value
                if c:
                    index -= 1
        else:
            res += c
        index += 1
    return res


def dump_metrics(file_path, metrics, log=False):
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)
