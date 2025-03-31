import numpy as np
import pandas as pd
from edstan import EdStanModel, EdStanMCMC

rng = np.random.default_rng()


def test_long():
    items = 5
    persons = 100
    result = EdStanModel('rasch').sample_from_long(
        ii=np.tile(range(items), reps=persons),
        jj=np.repeat(range(persons), repeats=items),
        y=rng.binomial(n=1, p=.5, size=items * persons),
        iter_warmup=100, iter_sampling=100, chains=1
    )
    assert isinstance(result, EdStanMCMC)


def test_wide_numpy():
    response_matrix = rng.binomial(n=1, p=.5, size=(100, 5))
    result = EdStanModel('rasch').sample_from_wide(response_matrix, iter_warmup=100, iter_sampling=100, chains=1)
    assert isinstance(result, EdStanMCMC)


def test_wide_pandas():
    response_df = pd.DataFrame(rng.binomial(n=1, p=.5, size=(100, 5)))
    result = EdStanModel('rasch').sample_from_wide(response_df, iter_warmup=100, iter_sampling=100, chains=1)
    assert isinstance(result, EdStanMCMC)


def test_wide_polytomous():
    response_matrix = rng.binomial(n=3, p=.5, size=(100, 5))
    result = EdStanModel('rsm').sample_from_wide(response_matrix, iter_warmup=100, iter_sampling=100, chains=1)
    assert isinstance(result, EdStanMCMC)
