---
title: "The edstan package for Python"
author: "Daniel C. Furr"
---

# Purpose

The edstan package for Python provides convenience functions and pre-programmed Stan models related to item response theory. Its purpose is to make fitting common IRT models using Stan easy. 

The following table lists the models provided in edstan along with links to case studies that document the models. These case studies use R rather than Python, but the Stan models are identical.

| Model | Stan file |
|-----------------------------------------------------------------------------------------------------|-------------------------|
| [Rasch](http://mc-stan.org/users/documentation/case-studies/rasch_and_2pl.html)                     | *rasch_latent_reg.stan* |
| [Partical credit](http://mc-stan.org/users/documentation/case-studies/pcm_and_gpcm.html)            | *pcm_latent_reg.stan*   |
| [Rating scale](http://mc-stan.org/users/documentation/case-studies/rsm_and_grsm.html)               | *rsm_latent_reg.stan*   |
| [Two-parameter logistic](http://mc-stan.org/users/documentation/case-studies/rasch_and_2pl.html)    | *2pl_latent_reg.stan*   |
| [Generalized partial credit](http://mc-stan.org/users/documentation/case-studies/pcm_and_gpcm.html) | *gpcm_latent_reg.stan*  |
| [Generalized rating scale](http://mc-stan.org/users/documentation/case-studies/rsm_and_grsm.html)   | *grsm_latent_reg.stan*  |


# Relation to R package

I developed a very similar package for R by the same name. While the R version is entirely function-driven, the Python version is based heavily around the `EdstanData` class and its methods.

# Installation

This package relies on the pystan package, which has installation instructions
[here](https://pystan.readthedocs.io/en/latest/installation_beginner.html).
The other dependencies should install automatically with edstan.

# Vignette 1

Respondents were asked to spell four words: 'infidelity', 'panoramic', 'succumb', and 'girder'. The file 'spelling.csv' provides a matrix with one column per item and one row per person, and each element equals 1 if correct and 0 otherwise. A fifth column contains a dummy variable for whether the respondent was male.
        
```{}
import edstan
import pandas

# Import data as a pandas data frame
spelling = pandas.read_csv('spelling.csv')
words = ['infidelity', 'panoramic', 'succumb', 'girder']

# Use the response matrix to create an EdstanData instance
ed_1 = edstan.EdstanData(response_matrix = spelling[words])

# Fit the Rasch model
fit_1 = ed_1.fit_model('rasch', iter = 200, chains = 4)

# Print results
ed_1.print_from_fit(fit_1)
```

A latent regression on gender may be added when creating the `EdstanData` instance. To do this, the person covariates and a formula are provided.

```{}
# Include a latent regression while creating an EdstanData instance
ed_2 = edstan.EdstanData(response_matrix = spelling[words],
                         person_data = spelling['male'],
                                     formula = '~male')

# Fit the 2pl model
fit_2 = ed_2.fit_model('2pl', iter = 200, chains = 4)

# Print results
ed_2.print_from_fit(fit_2)
```

Both `fit_1` and `fit_2` are `StanFit4model` instances of the same type created by pystan.


# Vignette 2

In the prior example, the data were arranged in a response matrix. Alternatively, data may be arrange such that each row represents one person's response to one item. The file 'aggression.csv' is an example. In it, 'poly' is a scored response (0, 1, 2), 'person' is an index for which person the response is associated , and 'item' is an index for which item the response is associated. Also, 'male' and 'anger' are person-related coavariates.

```{}
import edstan
import pandas

# Import data as a pandas data frame
aggression = pandas.read_csv('aggression.csv')

# Create EdstanData instance from a long response vector
ed_3 = edstan.EdstanData(item_id = aggression['item'], 
                         person_id = aggression['person'],
                         y = aggression['poly'],
                         person_data = aggression[['male', 'anger']],
                         formula = '~ male + anger')

# Fit the partial credit model
fit_3 = ed_3.fit_model('pcm', iter = 300, chains = 4)

# Print results
ed_2.print_from_fit(fit_2)
```
