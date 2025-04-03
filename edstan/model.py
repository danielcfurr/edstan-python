import os
from typing import Union, Dict
from warnings import warn
import numpy as np
from cmdstanpy import CmdStanModel
from numpy.typing import NDArray
from pandas import DataFrame
from .mcmc import EdStanMCMC


class EdStanModel(CmdStanModel):
    def __init__(self, model: str, **kwargs):
        if not isinstance(model, str):
            raise ValueError(f"Invalid value for 'model': {model}. Expected a string.")

        directory = os.path.join(os.path.dirname(__file__), 'data')
        matching_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.stan') and filename.startswith(model.lower()):
                matching_files.append(os.path.join(directory, filename))

        if len(matching_files) == 0:
            raise ValueError(f"Invalid value for 'model': {model}. No matching edstan model found.")
        if len(matching_files) > 1:
            raise ValueError(f"Invalid value for 'model': {model}. More than one matching edstan model found.")

        self.model = matching_files[0]
        super().__init__(stan_file=matching_files[0], **kwargs)

    def sample_from_long(self,
                         ii: NDArray[int],
                         jj: NDArray[int],
                         y: NDArray[int],
                         integerize: bool = True, **kwargs
                         ) -> EdStanMCMC:
        ii = _validate_vector(ii, label='ii')
        jj = _validate_vector(jj, label='jj')
        y = _validate_vector(y, label='y')
        data = data_from_long(ii=ii, jj=jj, y=y, integerize=integerize, extended=True)
        return self.sample_from_dict(data, **kwargs)

    def sample_from_wide(self, response_matrix: Union[NDArray, DataFrame], **kwargs) -> EdStanMCMC:
        data = data_from_wide(response_matrix=response_matrix, extended=True)
        return self.sample_from_dict(data, **kwargs)

    def sample_from_dict(self, data: Dict, **kwargs) -> EdStanMCMC:
        ii_labels = data.pop('ii_labels')
        jj_labels = data.pop('jj_labels')
        max_per_item = data.pop('max_per_item')
        mcmc = super().sample(data=data, **kwargs)
        return EdStanMCMC(mcmc, jj_labels=jj_labels, ii_labels=ii_labels, max_per_item=max_per_item)


def data_from_long(ii: NDArray[int],
                   jj: NDArray[int],
                   y: NDArray[int],
                   integerize: bool = True,
                   extended: bool = False,
                   ) -> Dict:
    ii = _validate_vector(ii, label='ii')
    jj = _validate_vector(jj, label='jj')
    y = _validate_vector(y, label='y')

    if not len(ii) == len(jj) == len(y):
        raise ValueError("'ii', 'jj', and 'y' must all have the same length.")

    if integerize:
        ii_ints, ii_labels = _map_to_unique_ids(ii)
        jj_ints, jj_labels = _map_to_unique_ids(jj)
    else:
        ii_ints, ii_labels = ii, _unique_unsorted(ii)
        jj_ints, jj_labels = jj, _unique_unsorted(jj)

    max_per_item = _validate_responses_by_item(y, ii_ints, ii_labels)

    data = {
        "I": max(ii_ints),
        "J": max(jj_ints),
        "N": len(y),
        "ii": ii_ints,
        "jj": jj_ints,
        "y": y,
        "K": 1,
        "W": [[1]] * max(jj_ints),
    }

    if extended:
        data.update({
            "ii_labels": ii_labels,
            "jj_labels": jj_labels,
            "max_per_item": max_per_item
        })

    return data


def data_from_wide(response_matrix: Union[NDArray, DataFrame], extended: bool = False) -> Dict:
    if isinstance(response_matrix, DataFrame):
        mat = _validate_pandas_matrix(response_matrix)
        ii = np.tile(response_matrix.columns, mat.shape[0])
        jj = np.repeat(response_matrix.index, mat.shape[1])
    else:
        mat = _validate_numpy_matrix(response_matrix)
        ii = np.tile(np.arange(mat.shape[1]) + 1, mat.shape[0])
        jj = np.repeat(np.arange(mat.shape[0]) + 1, mat.shape[1])

    y = mat.flatten()

    return data_from_long(ii=ii, jj=jj, y=y, extended=extended, integerize=True)


def _unique_unsorted(arr: NDArray):
    return np.array([x for i, x in enumerate(arr) if x not in arr[:i]])


def _map_to_unique_ids(arr: NDArray):
    unique_values = _unique_unsorted(arr)
    unique_values_list = unique_values.tolist()
    indices = np.array([unique_values_list.index(x) + 1 for x in arr])
    return indices, unique_values


def _validate_pandas_matrix(response_matrix: Union[NDArray, DataFrame]) -> NDArray:
    if response_matrix.shape[0] != len(np.unique(response_matrix.index)):
        raise ValueError("The pandas dataframe must not have duplicate index values.")

    if response_matrix.shape[1] != len(np.unique(response_matrix.columns)):
        raise ValueError("The pandas dataframe must not have duplicate column names.")

    if response_matrix.index.nlevels != 1:
        raise ValueError("The pandas dataframe must not have a multi-index along the rows.")

    if response_matrix.columns.nlevels != 1:
        raise ValueError("The pandas dataframe must not have a multi-index along the columns.")

    return _validate_numpy_matrix(response_matrix)


def _validate_numpy_matrix(response_matrix: Union[NDArray, DataFrame]) -> NDArray:
    try:
        mat = np.asarray(response_matrix)
    except Exception as exc:
        raise ValueError(
            "'response_matrix' must be a 2-dimensional numpy array or an object convertable to the same.") from exc

    if mat.ndim != 2:
        raise ValueError(f"'response_matrix' has {mat.ndim} dimensions, but must have two.")
    return mat


def _validate_vector(arr: NDArray, label: str) -> NDArray:
    try:
        arr = np.array(arr)
    except Exception as exc:
        raise ValueError(
            f"'{label}' must be a 1-dimensional numpy array or an object convertable to the same.") from exc

    if arr.ndim != 1:
        raise ValueError(f"'{label}' must be a 1-dimensional numpy array or an object convertable to the same.")

    return arr


def _validate_responses_by_item(y: NDArray, ii_ints: NDArray, ii_labels: NDArray) -> NDArray:
    max_per_item = np.zeros(max(ii_ints))

    for u in np.unique(ii_ints):
        responses = y[ii_ints == u]
        label = ii_labels[u - 1]

        mn = min(responses)
        mx = max(responses)

        if mn != 0:
            warn(f"Item {label} does not have a minimum response value of zero.")

        if len(np.unique(responses)) == 1:
            warn(f"Item {label} only has response values of {responses[0]}.")

        if len(np.unique(responses)) != (mx - mn + 1):
            warn(f"Item {label} has missing response categories.")

        max_per_item[u - 1] = mx

    return max_per_item
