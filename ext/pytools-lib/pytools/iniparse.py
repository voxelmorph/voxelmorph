"""
very simple ini parser and tools

tested on python 3.6

contact: adalca at csail.mit.edu

TODO: see 
  from collections import namedtuple
  instead of Struct
"""

# built-in modules
# we'll need python's ini parser: 'configparser'
import configparser

def ini_to_struct(file):
    """
    very simple ini parser that expands on configparser
    tries to cast values from string whereever possible
    parsed data ini can be accessed with

    data = ini_to_struct(file)
    value = data.section.key

    does not support hierarchical sections

    Parameters:
        file: string full filename of the ini file.

    Returns:
        stuct: a Struct that allows ini data to be access in the manner of data.section.key
    """

    # read the file via config.
    conf = configparser.ConfigParser()
    confout = conf.read(file)
    assert len(confout) > 0, 'Cannot read file %s ' % file

    # prepare the Struct
    strct = Struct()

    # go through the sections in the ini file
    for sec in conf.sections():

        # each section is its own struct
        secstrct = Struct()

        # go through the keys
        for key in conf[sec]:
            val = conf[sec][key]

            # try to cast the data
            ret, done = str_convert_single(val)

            # if couldn't cast, try a comma/whitespace separated list
            if not done:
                lst = str_to_list(val)

                # if the size of the list is 1, we didn't achieve anything
                if len(lst) == 1:
                    ret = lst[0]  # still not done

                # if we actually get a list, only keep it if we can cast its elements to something
                # otherwise keep the entry as an entire string
                else:
                    # make sure all elements in the list convert to something
                    done = all([str_convert_single(v)[1] for v in lst])
                    if done:
                        ret = [str_convert_single(v)[0] for v in lst]

            # defeated, accept the entry as just a simple string...
            if not done:
                ret = val  # accept string

            # assign secstrct.key = ret
            setattr(secstrct, key, ret)

        # assign strct.sec = secstrct
        setattr(strct, sec, secstrct)

    return strct


class Struct():
    """
    a simple struct class to allow for the following syntax:
    data = Struct()
    data.foo = 'bar'
    """

    def __str__(self):
        return self.__dict__.__str__()


def str_to_none(val):
    """
    cast a string to a None

    Parameters:
        val: the string to cast

    Returns:
        (casted_val, success)
        casted val: the casted value if successful, or None
        success: None if casting was successful
    """
    if val == 'None':
        return (None, True)
    else:
        return (None, False)


def str_to_type(val, ctype):
    """
    cast a string to a type (e.g. int('8')), with try/except
    do *not* use for bool casting, instead see str_to_bull

    Parameters:
        val: the string to cast

    Returns:
        (casted_val, success)
        casted val: the casted value if successful, or None
        success: bool if casting was successful
    """
    assert ctype is not bool, 'use str_to_bull() for casting to bool'

    ret = None
    success = True
    try:
        ret = ctype(val)
    except ValueError:
        success = False
    return (ret, success)


def str_to_bool(val):
    """
    cast a string to a bool

    Parameters:
        val: the string to cast

    Returns:
        (casted_val, success)
        casted val: the casted value if successful, or None
        success: bool if casting was successful
    """
    if val == 'True':
        return (True, True)
    elif val == 'False':
        return (False, True)
    else:
        return (None, False)


def str_to_list(val):
    """
    Split a string to a list of elements, where elements are separated by whitespace or commas
    Leading/ending parantheses are stripped.

    Returns:
        val: the string to split

    Returns:
        casted_dst: the casted list
    """
    val = val.replace('[', '')
    val = val.replace('(', '')
    val = val.replace(']', '')
    val = val.replace(')', '')

    if ',' in val:
        lst = val.split(',')
    else:
        lst = val.split()

    return lst


def str_convert_single(val):
    """
    try to cast a string to an int, float or bool (in that order)

    Parameters:
        val: the string to cast

    Returns:
        (casted_val, success)
        casted val: the casted value if successful, or None
        success: bool if casting was successful
    """
    val = val.strip()
    # try int
    ret, done = str_to_type(val, int)

    # try float
    if not done:
        ret, done = str_to_type(val, float)

    # try bool
    if not done:
        ret, done = str_to_bool(val)

    # try None
    if not done:
        ret, done = str_to_none(val)

    return (ret, done)

