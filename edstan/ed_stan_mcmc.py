from cmdstanpy import CmdStanMCMC
from numpy.typing import NDArray


class EdStanMCMC():
    def __init__(self, mcmc: CmdStanMCMC, ii_labels: NDArray, jj_labels: NDArray):
        self.mcmc = mcmc
        self.ii_labels = ii_labels
        self.jj_labels = jj_labels

    def __getattr__(self, name):
        return getattr(self.mcmc, name)

    def hello(self):
        return "hello"
