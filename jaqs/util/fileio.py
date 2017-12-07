# encoding: utf-8
import json
import os
import errno
try:
    import cPickle as pickle
except ImportError:
    import pickle
import codecs

from .. import SOURCE_ROOT_DIR


def create_dir(filename):
    """
    Create dir if directory of filename does not exist.

    Parameters
    ----------
    filename : str

    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def read_json(fp):
    """
    Read JSON file to dict. Return None if file not found.

    Parameters
    ----------
    fp : str
        Path of the JSON file.

    Returns
    -------
    dict

    """
    content = dict()
    try:
        with codecs.open(fp, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except IOError as e:
        if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
            raise
    return content


def save_json(serializable, file_name):
    """
    Save an serializable object to JSON file.

    Parameters
    ----------
    serializable : object
    file_name : str

    """
    fn = os.path.abspath(file_name)
    create_dir(fn)
    
    with codecs.open(fn, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, separators=(',\n', ': '))


def load_pickle(fp):
    """
    Read Pickle file. Return None if file not found.

    Parameters
    ----------
    fp : str
        Path of the Pickle file.

    Returns
    -------
    object or None

    """
    content = None
    try:
        with open(fp, 'rb') as f:
            content = pickle.load(f)
    except IOError as e:
        if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
            raise
    return content


def save_pickle(obj, file_name):
    """
    Save an object to Pickle file.

    Parameters
    ----------
    obj : object
    file_name : str

    """
    fn = os.path.abspath(file_name)
    create_dir(fn)

    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def join_relative_path(*paths):
    """Get absolute path using paths that are relative to project root."""
    return os.path.abspath(os.path.join(SOURCE_ROOT_DIR, *paths))


def fig2base64(fig, format='png'):
    """
    
    Parameters
    ----------
    fig : matplotlib.fig.Figure
    format : str
        Eg. png, jpg

    Returns
    -------

    """
    try:
        import BytesIO as io
    except ImportError:
        # from io import StringIO as StringIO
        import io
    import base64
    bytes_io = io.BytesIO()
    fig.savefig(bytes_io, format=format)
    bytes_io.seek(0)
    s = bytes_io.read()
    res = base64.b64encode(s)
    return res
