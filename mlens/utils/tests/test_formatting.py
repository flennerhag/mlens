"""ML-ENSEMBLE

"""

from mlens.utils.dummy import OLS
from mlens.utils.formatting import (_check_format, _assert_format,
                                    _format_instances, _check_instances)


def test_check_format():
    """[Utils] _check_format: test proper instance list passes test."""

    instances = [('ols-1', OLS()), ('ols-2', OLS(offset=1))]
    assert _check_format(instances)


def test_check_format_duplicate():
    """[Utils] _check_format: test duplicate names does not pass."""

    instances = [('ols-1', OLS()), ('ols-1', OLS(offset=1))]
    assert not _check_format(instances)


def test_check_format_no_name():
    """[Utils] _check_format: test non-named instance does not pass."""

    instances = [('ols-1', OLS()), OLS()]
    assert not _check_format(instances)

def test_check_format_iterable():
    """[Utils] _check_format: test tuple iterable does not pass."""

    instances = (('ols-1', OLS()), ('ols-2', OLS(offset=1)))
    assert not _check_format(instances)


def test_check_format_non_est():
    """[Utils] _check_format: test non-est instance does not pass."""

    instances = (('ols-1', dict()), ('ols-2', OLS(offset=1)))
    assert not _check_format(instances)


def test_check_format_empty():
    """[Utils] _check_format: test empty list passes."""
    assert _check_format([])


def test_assert_format_list():
    """[Utils] _assert_format: test correct list passes."""
    instances = [('ols-1', OLS()), ('ols-2', OLS(offset=1))]
    assert _assert_format(instances)


def test_assert_format_list_bad():
    """[Utils] _assert_format: test correct list passes."""
    instances = [('ols-1', OLS()), OLS(offset=1)]
    assert not _assert_format(instances)


def test_assert_format_dict():
    """[Utils] _assert_format: test correct dict passes."""
    instances = {'a': [('ols-a1', OLS()), ('ols-a2', OLS(offset=1))],
                 'b': [('ols-b1', OLS()), ('ols-b2', OLS(offset=1))]}
    assert _assert_format(instances)


def test_assert_format():
    """[Utils] _assert_format: test incorrect dict does not pass."""
    instances = {'a': [('ols-1', OLS()), ('ols-2', OLS(offset=1))],
                 'b': [OLS(), ('ols-2', OLS(offset=1))]}
    assert not _assert_format(instances)


def test_formatting_list():
    """[Utils] _format_instances: test correct formatting of list."""

    instances = [OLS(), ('ols', OLS()), ('ols', OLS()), ['list', OLS()]]

    formatted = _format_instances(instances, False)

    strings = []
    for i in formatted:
        assert isinstance(i, tuple)
        assert isinstance(i[0], str)
        assert isinstance(i[1], OLS)
        assert i[0] not in strings
        strings.append(i[0])


def test_check_instances_list_same():
    """[Utils] check_instances: test correct list is returned as is."""

    instances = [('ols-1', OLS()), ('ols-2', OLS(offset=1))]
    out = _check_instances(instances)

    assert id(out) == id(instances)
    for i in range(2):
        for j in range(2):
            assert id(out[i][j]) == id(instances[i][j])


def test_check_instances_list_formatting():
    """[Utils] check_instances: test formatting of list."""

    instances = [OLS(), ('ols', OLS()), ('ols', OLS()), ['list', OLS()]]
    formatted = _check_instances(instances)

    strings = []
    for i in formatted:
        assert isinstance(i, tuple)
        assert isinstance(i[0], str)
        assert isinstance(i[1], OLS)
        assert i[0] not in strings
        strings.append(i[0])


def test_check_instances_dict():
    """[Utils] check_instances: test correct dict is returned as is."""

    instances = {'a': [('ols-a1', OLS()), ('ols-a2', OLS(offset=1))],
                 'b': [('ols-b1', OLS()), ('ols-b2', OLS(offset=1))],
                 }
    out = _check_instances(instances)

    assert id(out) == id(instances)
    for k in out:
        ou = out[k]
        it = instances[k]
        for i in range(2):
            for j in range(2):
                assert id(ou[i][j]) == id(it[i][j])


def test_check_instances_dict_formatting():
    """[Utils] check_instances: test formatting of dict."""
    instances = {'a': [OLS(), ('ols', OLS()), ('ols', OLS()), ['list', OLS()]],
                 'b': [],
                 'c': [OLS(), ('ols', OLS())]}

    formatted = _check_instances(instances)

    for k, v in formatted.items():
        if k == 'b':
            assert len(v) == 0
        else:
            for i in v:
                strings = []
                assert isinstance(i, tuple)
                assert isinstance(i[0], str)
                assert isinstance(i[1], OLS)
                assert i[0] not in strings
                strings.append(i[0])
