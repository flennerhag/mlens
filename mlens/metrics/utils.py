"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Utility functions for constructing metrics
"""
from __future__ import division

import warnings
import numpy as np

from ..utils.exceptions import MetricWarning
try:
    from collections import OrderedDict as _dict
except ImportError:
    _dict = dict


def _get_string(obj, dec):
    """Stringify object"""
    try:
        return '{0:.{dec}f}'.format(obj, dec=dec)
    except TypeError:
        return obj.__str__()


def _get_partitions(obj):
    """Check if any entry has partitions"""
    for name, _ in obj:
        if int(name.split('.')[-2]) > 0:
            return True
    return False


def _split(f, s, a_p='', a_s='', b_p='', b_s='', reverse=False):
    """Split string on a symbol and return two string, first possible empty"""
    splitted = f.split(s)
    if len(splitted) == 1:
        a, b = '', splitted[0]
        if reverse:
            b, a = a, b
    else:
        a, b = splitted

    if a:
        a = '%s%s%s' % (a_p, a, a_s)
    if b:
        b = '%s%s%s' % (b_p, b, b_s)

    return a, b


class Data(_dict):

    """Wrapper class around dict to get pretty prints
    """
    def __init__(self, data=None, padding=2, decimals=2):
        if isinstance(data, list):
            data = assemble_data(data)
        super(Data, self).__init__(data)
        self.__padding__ = padding
        self.__decimals__ = decimals

    def __repr__(self):
        return assemble_table(self, self.__padding__, self.__decimals__)


def assemble_table(data, padding=2, decimals=2):
    """Construct data table from input dict"""
    buffer = 0
    row_glossary = ['layer', 'case', 'est', 'part']

    cols = list()
    rows = list()
    row_keys = list()
    max_col_len = dict()
    max_row_len = {r: 0 for r in row_glossary}

    # First, measure the maximum length of each column in table
    for key, val in data.items():
        cols.append(key)
        max_col_len[key] = len(key)

        # dat_key is the estimators. Number of columns is not fixed so need
        # to assume all exist and purge empty columns
        for dat_key, v in sorted(val.items()):
            if not v:
                # Safety: no data
                continue

            v_ = len(_get_string(v, decimals))
            if v_ > max_col_len[key]:
                max_col_len[key] = v_

            if dat_key in row_keys:
                # Already mapped row entry name
                continue

            layer, k = _split(dat_key, '/')
            case, k = _split(k, '.')
            est, part = _split(k, '--', reverse=True)

            # Header space before column headings
            items = [i for i in [layer, case, est, part] if i != '']
            buffer = max(buffer, len('  '.join(items)))

            for k, v in zip(row_glossary, [layer, case, est, part]):
                v_ = len(v)
                if v_ > max_row_len[k]:
                    max_row_len[k] = v_

            dat = _dict()
            dat['layer'] = layer
            dat['case'] = case
            dat['est'] = est
            dat['part'] = part
            row_keys.append(dat_key)
            rows.append(dat)

    # Check which row name columns we can drop (ex partition number)
    drop = list()
    for k, v in max_row_len.items():
        if v == 0:
            drop.append(k)

    # Header
    out = " " * (buffer + padding)
    for col in cols:
        adj = max_col_len[col] - len(col) + padding
        out += " " * adj + col
    out += "\n"

    # Entries
    for dat_key, dat in zip(row_keys, rows):
        # Estimator name
        for key, val in dat.items():
            if key in drop:
                continue
            adj = max_row_len[key] - len(val) + padding
            out += val + " " * adj

        # Data
        for col in cols:
            item = data[col][dat_key]
            if not item and item != 0:
                out += " " * (max_col_len[col] + padding)
                continue
            item_ = _get_string(item, decimals)
            adj = max_col_len[col] - len(item_) + padding
            out += " " * adj + item_
        out += "\n"
    return out


def assemble_data(data_list):
    """Build a data dictionary out of a list of datum"""
    data = _dict()
    tmp = _dict()

    partitions = _get_partitions(data_list)

    # Collect scores per preprocessing case and estimator(s)
    for name, data_dict in data_list:
        if not data_dict:
            continue

        prefix, name = _split(name, '/', a_s='/')

        # Names are either est.i.j or case.est.i.j
        splitted = name.split('.')
        if partitions:
            name = tuple(splitted[:-1])

            if len(name) == 3:
                name = '%s.%s--%s' % name
            else:
                name = '%s--%s' % name
        else:
            name = '.'.join(splitted[:-2])

        name = '%s%s' % (prefix, name)

        if name not in tmp:
            # Set up data struct for name
            tmp[name] = _dict()
            for k in data_dict.keys():
                tmp[name][k] = list()
                if '%s-m' % k not in data:
                    data['%s-m' % k] = _dict()
                    data['%s-s' % k] = _dict()
                data['%s-m' % k][name] = list()
                data['%s-s' % k][name] = list()

        # collect all data dicts belonging to name
        for k, v in data_dict.items():
            tmp[name][k].append(v)

    # Aggregate to get mean and std
    for name, data_dict in tmp.items():
        for k, v in data_dict.items():
            if not v:
                continue
            try:
                # Purge None values from the main est due to no predict times
                v = [i for i in v if i is not None]
                if v:
                    data['%s-m' % k][name] = np.mean(v)
                    data['%s-s' % k][name] = np.std(v)
            except Exception as exc:
                warnings.warn(
                    "Aggregating data for %s failed. Raw data:\n%r\n"
                    "Details: %r" % (k, v, exc), MetricWarning)

    # Check if there are empty columns
    discard = list()
    for key, data_dict in data.items():
        empty = True
        for val in data_dict.values():
            if val or val == 0:
                empty = False
        if empty:
            discard.append(key)
    for key in discard:
        data.pop(key)
    return data
