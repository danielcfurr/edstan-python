import os
from cmdstanpy import CmdStanModel, CmdStanMCMC
from typing import List


class EdStanModel(CmdStanModel):
    def __init__(self, prefix: str, **kwargs):
        if not isinstance(prefix, str):
            raise ValueError(f"Invalid value for 'prefix': {prefix}. Expected a string.")

        directory = os.path.join(os.path.dirname(__file__), 'data')
        stan_file = None
        for filename in os.listdir(directory):
            if filename.endswith('.stan') and filename.startswith(prefix.lower()):
                stan_file = os.path.join(directory, filename)

        if not stan_file:
            raise ValueError(f"Invalid value for 'prefix': {prefix}. Expected a match for an edstan model file name.")

        super().__init__(stan_file=stan_file, **kwargs)


    def sample_from_long(self, ii: List, jj: List, y: List, **kwargs) -> CmdStanMCMC:
        data = dict(
            I=max(ii),
            J=max(jj),
            N=len(y),
            ii=ii,
            jj=jj,
            y=y,
            K=1,
            W=[[1]] * max(jj)
        )

        return super().sample(data=data, **kwargs)


    def sample_from_wide(self, response_matrix: List, **kwargs) -> CmdStanMCMC:
        y = [response for row in response_matrix for response in row]
        I = len(response_matrix[0])
        J = len(response_matrix)
        ii = list(range(1, I+1)) * J
        jj = [j for j in range(1, J + 1) for _ in range(I)]

        return self.sample_from_long(ii=ii, jj=jj, y=y, **kwargs)
