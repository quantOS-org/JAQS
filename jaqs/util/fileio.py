# encoding: utf-8
import json
import os
import errno

from jaqs import SOURCE_ROOT_DIR


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
    dict or None

    """
    content = None
    try:
        with open(fp, 'r') as f:
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
    create_dir(file_name)
    
    with open(file_name, 'w') as f:
        json.dump(serializable, f)


def join_relative_path(*paths):
    """Get absolute path using paths that are relative to project root."""
    return os.path.abspath(os.path.join(SOURCE_ROOT_DIR, *paths))
