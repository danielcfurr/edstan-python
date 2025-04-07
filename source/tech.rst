Technical Notes
===============

Users will be able to fit the :mod:`edstan` models without full knowledge of
the technical details, though these are provided in this section. All
that is really needed for interpreting results is to know the meanings
assigned to the Greek letters.

Notation
--------

Variables and parameters are similar across :mod:`edstan` models. The variables
used are:

- :math:`i = 1 \ldots I` indexes items.
- :math:`j = 1 \ldots J` indexes persons.
- :math:`m_i` is simultaneously the maximum score and the number of step
  difficulty parameters for item $i$ for partial credit models.
  Alternatively, :math:`m` is the same across all items for rating scale
  models.
- :math:`s = 1 \ldots m_i` or :math:`s = 1 \ldots m` indexes steps within items.
- :math:`y_{ij}` is the scored response of person :math:`j` to item :math:`i`. The lowest
  score for items must be zero (except for rating scale models).

The parameters used are:

- For the Rasch and 2PL models, :math:`\beta_i` is the difficulty for item
  :math:`i`. For the rating scale models, :math:`\beta_i` is the mean difficulty for
  item :math:`i`. For partial credit models, :math:`\beta_{is}` is the difficulty
  for step :math:`s` of item :math:`i`.
- :math:`\kappa_s` is a step difficulty for the (generalized) rating scale
  model.
- :math:`\alpha_i` is the discrimination parameter for item :math:`i` (when
  applicable).
- :math:`\theta_j` is the ability for person :math:`j`.
- :math:`\lambda` is mean of the ability distribution.
- :math:`\sigma` is standard deviation for the ability distribution..

The *.stan* files and the notation for the models below closely adhere
to these conventions.

Rasch family models
-------------------

Rasch model
^^^^^^^^^^^

*rasch_latent_reg.stan*

.. math::

    \mathrm{logit} [ \Pr(y_{ij} = 1 | \theta_j, \beta_i) ] =
      \theta_j - \beta_i

Partial credit model
^^^^^^^^^^^^^^^^^^^^

*pcm_latent_reg.stan*

.. math::
    \Pr(Y_{ij} = y,~y > 0 | \theta_j, \beta_i) =
    \frac{\exp \sum_{s=1}^y (\theta_j - \beta_{is})}
         {1 + \sum_{k=1}^{m_i} \exp \sum_{s=1}^k (\theta_j - \beta_{is})}

.. math::
    \Pr(Y_{ij} = y,~y = 0 | \theta_j, \beta_i) =
    \frac{1}
         {1 + \sum_{k=1}^{m_i} \exp \sum_{s=1}^k (\theta_j - \beta_{is})}

Rating scale model
^^^^^^^^^^^^^^^^^^

*rsm_latent_reg.stan*

.. math::
    \Pr(Y_{ij} = y,~y > 0 | \theta_j, \beta_i, \kappa_s) =
    \frac{\exp \sum_{s=1}^y (\theta_j - \beta_i - \kappa_s)}
         {1 + \sum_{k=1}^{m} \exp \sum_{s=1}^k (\theta_j - \beta_i - \kappa_s)}

.. math::
    \Pr(Y_{ij} = y,~y = 0 | \theta_j, \beta_i, \kappa_s) =
    \frac{1}
         {1 + \sum_{k=1}^{m} \exp \sum_{s=1}^k (\theta_j - \beta_i - \kappa_s)}

Models featuring discrimination parameters
------------------------------------------

Two-parameter logistic model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*2pl_latent_reg.stan*

.. math::
  \mathrm{logit} [ \Pr(y_{ij} = 1 | \alpha_i, \beta_i, \theta_j) ] =
  \alpha_i \theta_j - \beta_i

Generalized partial credit model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*gpcm_latent_reg.stan*

.. math::
    \Pr(Y_{ij} = y,~y > 0 | \theta_j, \alpha_i, \beta_i) =
    \frac{\exp \sum_{s=1}^y (\alpha_i  \theta_j - \beta_{is})}
         {1 + \sum_{k=1}^{m_i} \exp \sum_{s=1}^k
           (\alpha_i \theta_j - \beta_{is})}

.. math::
    \Pr(Y_{ij} = y,~y = 0 | \theta_j, \alpha_i, \beta_i) =
    \frac{1}
         {1 + \sum_{k=1}^{m_i} \exp \sum_{s=1}^k
           (\alpha_i \theta_j + w_{j}' \lambda - \beta_{is})}

Generalized rating scale model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*grsm_latent_reg.stan*

.. math::
    \Pr(Y_{ij} = y,~y > 0 | \theta_j, \lambda, \alpha_i, \beta_i, \kappa_s) =
    \frac{\exp \sum_{s=1}^y
           (\alpha_i \theta_j - \beta_i - \kappa_s)}
         {1 + \sum_{k=1}^{m} \exp \sum_{s=1}^k
           (\alpha_i \theta_j - \beta_i - \kappa_s)}

.. math::
    \Pr(Y_{ij} = y,~y = 0 | \theta_j, \lambda, \alpha_i, \beta_i, \kappa_s) =
    \frac{1}
         {1 + \sum_{k=1}^{m} \exp \sum_{s=1}^k
           (\alpha_i \theta_j - \beta_i - \kappa_s)}


Prior distributions
-------------------

For Rasch family models, the prior distributions for the person-related
parameters are

- :math:`\theta_j \sim \mathrm{N}(\lambda, \sigma^2)`
- :math:`\lambda \sim t_7(0, 2.5)`
- :math:`\sigma \sim \mathrm{gamma}(2, 1)`

For models with discrimination parameters, the priors are

- :math:`\theta_j \sim \mathrm{N}(\lambda, 1)`
- :math:`\lambda \sim t_7(0, 2.5)`

The priors for the item parameters are

- :math:`\alpha \sim \mathrm{lognormal}(.5, 1)`
- :math:`\beta \sim \mathrm{N}(0, 9)`
- :math:`\kappa \sim \mathrm{N}(0, 9)`
