# edstan for Python

A python module that simplifies the fitting of common Bayesian item response theory models using Stan. It is compatible
with and extends the functionality of **pystan**.


Features

- Streamlined interface to common Bayesian item response models using Stan
- Models include: Rasch, two-parameter logistic, (generalized) rating scale, and (generalized) partial credit
- Posterior summaries tailored to item response models


Installation
------------

**edstan** depends on the successful installation of **pystan**, so please see the
[pystan installation guide](https://pystan.readthedocs.io/en/latest/installation.html).
Note that compatibility with Windows OS
[may be limited](https://pystan.readthedocs.io/en/latest/faq.html).

**edstan** may subsequently be installed with **pip**:

```bash
pip install edstan
```


## Quickstart

Here is an example of running a model using data from a response matrix:

```python
from edstan import EdStanModel
import numpy as np
import pandas as pd

# Simulate a "wide format" data frame of item responses for
# 5 items and 100 persons. Responses are scored 0 or 1.
rng = np.random.default_rng(seed=42)
data = pd.DataFrame(rng.binomial(1, p=.5, size=(100, 5)))
data.columns = [f"Question {i}" for i in range(5)]
data.index = [f"Respondent {i}" for i in range(100)]

# Instantiate the model, selecting the Rasch model
model = EdStanModel("rasch")

# Sample from the model by MCMC
fit = model.sample_from_wide(data)

# View a posterior summary of the item (and person distribution)
# parameters
print(fit.item_summary())

# View a posterior summary of the person parameters
print(fit.person_summary())
```

Alternatively, this is an example of using long format data:

```python
from edstan import EdStanModel
import numpy as np
import pandas as pd

# Simulate a "long format" data frame of item responses for
# 5 items and 100 persons. Responses are scored 0, 1, or 2.
rng = np.random.default_rng(seed=42)
data = pd.DataFrame(
  {
     "person": [f"Person {j}" for j in range(100) for i in range(5)],
     "item": [f"Item {i}" for j in range(100) for i in range(5)],
     "response": rng.binomial(2, p=.5, size=5*100),
  }
)

# Instantiate the model, choosing the generalized partial
# credit model
model = EdStanModel("gpcm")

# Sample from the model by MCMC
fit = model.sample_from_long(
  ii=data['item'],
  jj=data['person'],
  y=data['response'],
)

# View a posterior summary of the item (and person distribution)
# parameters
print(fit.item_summary())

# View a posterior summary of the person parameters
print(fit.person_summary())
```
