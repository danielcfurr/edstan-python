import numpy as np
import pandas as pd
from edstan import EdStanModel, EdStanMCMC

rng = np.random.default_rng(42)

# Preset number of items and persons for data generation
items = 5
persons = 100

# Dichotomous response matrix as array
dich_arr = rng.binomial(1, p=.5, size=(persons, items))

# Dichotomous response matrix as data frame
dich_df = pd.DataFrame(dich_arr)
dich_df.columns = [f'Item {i}' for i in range(items)]
dich_df.index = [f'Person {i}' for i in range(persons)]

# Polytomous response matrix as array
poly_arr = rng.binomial(2, p=.5, size=(persons, items))

# Polytomous response matrix as data frame
poly_df = pd.DataFrame(poly_arr)
poly_df.columns = dich_df.columns
poly_df.index = dich_df.index

# Dich/polytomous responses as long-format arrays
dich_y = dich_arr.flatten()
poly_y = poly_arr.flatten()
ii = np.tile(dich_df.columns, reps=persons)
jj = np.repeat(dich_df.index, repeats=items)


def test_long():
    result = EdStanModel('rasch').sample_from_long(
        ii=ii, jj=jj, y=dich_y,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert isinstance(result, EdStanMCMC)


def test_wide_numpy():
    response_matrix = rng.binomial(n=1, p=.5, size=(100, 5))
    result = EdStanModel('rasch').sample_from_wide(
        dich_arr,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert isinstance(result, EdStanMCMC)


def test_wide_pandas():
    result = EdStanModel('rasch').sample_from_wide(
        dich_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert isinstance(result, EdStanMCMC)


def test_wide_polytomous():
    result = EdStanModel('rsm').sample_from_wide(
        poly_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert isinstance(result, EdStanMCMC)


def test_summary_rasch():
    fit = EdStanModel('rasch').sample_from_wide(
        dich_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 1 + 2
    assert fit.person_summary().shape[0] == persons


def test_summary_2pl():
    fit = EdStanModel('2pl').sample_from_wide(
        dich_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 2 + 1
    assert fit.person_summary().shape[0] == persons


def test_summary_rsm():
    fit = EdStanModel('rsm').sample_from_wide(
        poly_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 1 + 4
    assert fit.person_summary().shape[0] == persons


def test_summary_grsm():
    fit = EdStanModel('grsm').sample_from_wide(
        poly_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 2 + 3
    assert fit.person_summary().shape[0] == persons


def test_summary_pcm():
    fit = EdStanModel('pcm').sample_from_wide(
        poly_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 2 + 2
    assert fit.person_summary().shape[0] == persons


def test_summary_gpcm():
    fit = EdStanModel('gpcm').sample_from_wide(
        poly_df,
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert fit.item_summary().shape[0] == items * 3 + 1
    assert fit.person_summary().shape[0] == persons
