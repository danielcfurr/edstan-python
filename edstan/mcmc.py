from cmdstanpy import CmdStanMCMC
from numpy.typing import NDArray
import numpy as np
import pandas as pd


class EdStanMCMC():
    def __init__(self, mcmc: CmdStanMCMC, ii_labels: NDArray, jj_labels: NDArray, max_per_item: NDArray):
        self.mcmc = mcmc
        self.ii_labels = ii_labels
        self.jj_labels = jj_labels
        self.max_per_item = max_per_item

    def __getattr__(self, name):
        return getattr(self.mcmc, name)

    def item_summary(self, **kwargs):
        summary = self.mcmc.summary(**kwargs)
        summary.index.name = 'parameter'

        expected = _get_expected_parameters_by_group(
            item_labels=self.ii_labels,
            max_per_item=self.max_per_item,
            rasch_family='sigma' in summary.index,
            ratings_model='kappa[1]' in summary.index
        )

        return expected.merge(summary.reset_index(), on='parameter').set_index(['parameter group', 'parameter'])

    def person_summary(self, **kwargs):
        summary = self.mcmc.summary(**kwargs)
        summary = summary.loc[summary.index.str.match('theta')]
        summary['person'] = self.jj_labels
        summary.index.name = 'parameter'
        return summary.reset_index().set_index(['person', 'parameter'])


def _get_expected_parameters_by_group(item_labels, max_per_item, rasch_family: bool, ratings_model: bool):
    holder = []

    if ratings_model:
        betas_per_item = np.ones(len(max_per_item), dtype=int)
    else:
        betas_per_item = np.array(max_per_item, dtype=int)

    beta_counter = 0
    for item, item_max in enumerate(betas_per_item):
        if not rasch_family:
            holder.append((item_labels[item], f'alpha[{item + 1}]'))
        for step in range(item_max):
            beta_counter += 1
            holder.append((item_labels[item], f'beta[{beta_counter}]'))

    if ratings_model:
        for j in range(int(max(max_per_item))):
            holder.append(('Rating scale steps', f'kappa[{j + 1}]'))

    holder.append(('Ability distribution', 'lambda[1]'))

    if rasch_family:
        holder.append(('Ability distribution', 'sigma'))

    df = pd.DataFrame(holder)
    df.columns = ['parameter group', 'parameter']

    return df
