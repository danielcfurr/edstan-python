import numpy as np
import pandas as pd
import pytest

from edstan import model


def test_unique_unsorted():
    x = np.array(["dog", "cat", "dog", "pony", "owl"])
    result = model._unique_unsorted(x)
    expected = np.array(["dog", "cat", "pony", "owl"])
    np.testing.assert_array_equal(result, expected)


def test_map_to_unique_ids():
    x = np.array(["dog", "cat", "dog", "pony", "owl"])
    result_ints, result_labels = model._map_to_unique_ids(x)
    expected_ints = np.array([1, 2, 1, 3, 4])
    expected_labels = np.array(["dog", "cat", "pony", "owl"])
    np.testing.assert_array_equal(result_ints, expected_ints)
    np.testing.assert_array_equal(result_labels, expected_labels)


def test_validate_numpy_matrix():
    x = np.zeros((2, 2))
    result = model._validate_numpy_matrix(x)
    np.testing.assert_array_equal(result, x)


def test_validate_numpy_matrix__errors():
    with pytest.raises(ValueError):
        model._validate_numpy_matrix(np.zeros(3))
    with pytest.raises(ValueError):
        model._validate_numpy_matrix(np.zeros((3, 3, 3)))


def test_validate_pandas_matrix():
    x = np.zeros((2, 2))
    df = pd.DataFrame(x)
    result = model._validate_pandas_matrix(df)
    np.testing.assert_array_equal(result, x)


def test_validate_pandas_matrix__errors():
    df = pd.DataFrame(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        x = df.copy()
        x.columns = ["a", "a"]
        model._validate_pandas_matrix(x)
    with pytest.raises(ValueError):
        x = df.copy()
        x.index = ["a", "a"]
        model._validate_pandas_matrix(x)
    with pytest.raises(ValueError):
        x = df.copy()
        x.index = pd.MultiIndex.from_product([["a"], ["b", "c"]])
        model._validate_pandas_matrix(x)
    with pytest.raises(ValueError):
        x = df.copy()
        x.columns = pd.MultiIndex.from_product([["a"], ["b", "c"]])
        model._validate_pandas_matrix(x)


def test_validate_responses_by_item__warnings():
    ii_ints = np.array([1, 1, 2, 2])
    with pytest.warns(UserWarning):
        # No variation in second item
        model._validate_responses_by_item(np.array([0, 1, 1, 1]), ii_ints, ii_ints)
    with pytest.warns(UserWarning):
        # No variation in second item
        model._validate_responses_by_item(np.array([0, 1, 0, 0]), ii_ints, ii_ints)
    with pytest.warns(UserWarning):
        # Missing zero in second item
        model._validate_responses_by_item(np.array([0, 1, 1, 2]), ii_ints, ii_ints)
    with pytest.warns(UserWarning):
        # Missing category in second item
        model._validate_responses_by_item(np.array([0, 1, 0, 2]), ii_ints, ii_ints)
