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
        return str(np.round(obj, dec))
    except TypeError:
        return obj.__str__()


def _get_partitions(obj):
    """Check if any entry has partitions"""
    for name, _ in obj:
        if int(name.split('__')[-2]) > 0:
            return True
    return False


class Data(_dict):

    """Wrapper class around dict to get pretty prints
    """
    def __init__(self, data=None):
        if isinstance(data, list):
            data = assemble_data(data)
        super(Data, self).__init__(data)

    def __repr__(self):
        return assemble_table(self)


def assemble_table(data, padding=2, decimals=2):
    """Construct data table from input dict"""
    db_ = 2  # two if case - est else 1

    # Construct column and row list (rows : case est)
    # Measure table entry lengths
    case = list()
    ests = list()
    cols = list()
    parts = list()
    rows = list()
    max_col_len = dict()
    max_case_len = 0
    max_part_len = 0
    max_est_len = 0

    for key, val in data.items():
        cols.append(key)
        max_col_len[key] = len(key)

        for k, v in val.items():
            # Update longest column entry for column 'key'
            if not v:
                continue

            v_ = len(_get_string(v, decimals))
            if v_ > max_col_len[key]:
                max_col_len[key] = v_

            if isinstance(k, tuple):
                db_ = len(k)
                if db_ == 2:
                    c, e = k
                    try:
                        int(e)
                        p = e
                        e = c
                        c = ''
                    except Exception:
                        p = ''
                elif db_ == 3:
                    c, e, p = k

                c_, p_ = len(c), len(p)
                if c_ > max_case_len:
                    max_case_len = c_
                if p_ > max_part_len:
                    max_part_len = p_
            else:
                # Treat p and c as empty
                p, c, e = '', '', k
                db_ = 1

            e_ = len(e)
            if e_ > max_est_len:
                max_est_len = e_

            if (c, e, p) not in rows:
                rows.append((c, e, p))
            if c not in case:
                case.append(c)
            if e not in ests:
                ests.append(e)
            if p not in parts:
                parts.append(p)

    # Header
    out = " " * (max_case_len + max_est_len + max_part_len + db_ * padding)
    for col in cols:
        adj = max_col_len[col] - len(col) + padding
        out += " " * adj + col
    out += "\n"

    for c in sorted(case):
        for e in sorted(ests):
            for p in sorted(parts):
                # Row entries
                if (c, e, p) not in rows:
                    continue

                # Format row
                k = [e]

                if c:
                    # First row entry
                    k = [c] + k
                    adj = max_case_len - len(c) + padding
                    out += " " * adj + c

                # Always est entry
                adj = max_est_len - len(e) + padding
                out += " " * adj + e

                if p:
                    # Partition entry
                    k.append(p)
                    adj = max_part_len - len(p) + padding
                    out += " " * adj + p

                if len(k) == 1:
                    k = e
                else:
                    k = tuple(k)

                # Table contents
                for col in cols:
                    item = data[col][k]
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

        # Names are either est__i__j or case__est__i__j
        splitted = name.split('__')

        if partitions:
            name = tuple(splitted[:-1])
        else:
            name = tuple(splitted[:-2])

        if len(name) == 1:
            name = name[0]

        try:
            tmp[name]
        except KeyError:
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
        if not data_dict:
            continue

        for k, v in data_dict.items():
            if not v:
                continue
            try:
                # Purge None values from the main learner
                # I.e. no prediction time during fit call
                v = [i for i in v if i is not None]
                if v:
                    data['%s-m' % k][name] = np.mean(v)
                    data['%s-s' % k][name] = np.std(v)
            except Exception as exc:
                warnings.warn(
                    "Aggregating data for %s failed. Raw data:\n%r\n"
                    "Details: %r" % (k, v, exc), MetricWarning)

    # Drop empty entries
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
