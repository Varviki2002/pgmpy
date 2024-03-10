from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
# from pgmpy.sampling import GibbsSampling
from pgmpy.models import MarkovNetwork, MarkovChain
from pgmpy.factors.discrete import DiscreteFactor
import numpy as np
import itertools

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.models import BayesianNetwork, MarkovChain, MarkovNetwork
from pgmpy.sampling import BayesianModelInference, _return_samples
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps

class GibbsSampling(MarkovChain):
    """
    Class for performing Gibbs sampling.

    Parameters
    ----------
    model: BayesianNetwork or MarkovNetwork
        Model from which variables are inherited and transition probabilities computed.

    Examples
    --------
    Initialization from a BayesianNetwork object:

    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.models import BayesianNetwork
    >>> intel_cpd = TabularCPD('intel', 2, [[0.7], [0.3]])
    >>> sat_cpd = TabularCPD('sat', 2, [[0.95, 0.2], [0.05, 0.8]], evidence=['intel'], evidence_card=[2])
    >>> student = BayesianNetwork()
    >>> student.add_nodes_from(['intel', 'sat'])
    >>> student.add_edge('intel', 'sat')
    >>> student.add_cpds(intel_cpd, sat_cpd)
    >>> from pgmpy.sampling import GibbsSampling
    >>> gibbs_chain = GibbsSampling(student)
    >>> gibbs_chain.sample(size=3)
       intel  sat
    0      0    0
    1      0    0
    2      1    1
    """

    def __init__(self, model=None):
        super(GibbsSampling, self).__init__()
        if isinstance(model, BayesianNetwork):
            self._get_kernel_from_bayesian_model(model)
        elif isinstance(model, MarkovNetwork):
            self._get_kernel_from_markov_model(model)

    def _get_kernel_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: BayesianNetwork
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.latents = model.latents
        self.cardinalities = {
            var: model.get_cpds(var).variable_card for var in self.variables
        }

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            factors = [cpd.to_factor() for cpd in model.cpds if var in cpd.scope()]
            factor = factor_product(*factors)
            scope = set(factor.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [State(v, s) for v, s in zip(other_vars, tup) if v in scope]
                reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = reduced_factor.values / sum(reduced_factor.values)
            self.transition_models[var] = kernel

    def _get_kernel_from_markov_model(self, model):
        """
        Computes the Gibbs transition models from a Markov Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: MarkovNetwork
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.latents = model.latents
        factors_dict = {var: [] for var in self.variables}
        for factor in model.get_factors():
            for var in factor.scope():
                factors_dict[var].append(factor)

        # Take factor product
        factors_dict = {
            var: factor_product(*factors) if len(factors) > 1 else factors[0]
            for var, factors in factors_dict.items()
        }
        self.cardinalities = {
            var: factors_dict[var].get_cardinality([var])[var] for var in self.variables
        }

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            factor = factors_dict[var]
            scope = set(factor.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [
                    State(first_var, s)
                    for first_var, s in zip(other_vars, tup)
                    if first_var in scope
                ]
                reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = reduced_factor.values / sum(reduced_factor.values)
            self.transition_models[var] = kernel

    def sample(self, start_state=None, size=1, seed=None, include_latents=False):
        """
        Sample from the Markov Chain.

        Parameters
        ----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.

        size: int
            Number of samples to be generated.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import MarkovNetwork
        >>> model = MarkovNetwork([('A', 'B'), ('C', 'B')])
        >>> factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(['C', 'B'], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gibbs.sample(size=4, return_tupe='dataframe')
           A  B  C
        0  0  1  1
        1  1  0  0
        2  1  1  0
        3  1  1  1
        """
        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        if seed is not None:
            np.random.seed(seed)

        types = [(str(var_name), "int") for var_name in self.variables]
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        sampled[0] = tuple(st for var, st in self.state)
        for i in tqdm(range(size - 1)):
            for j, (var, st) in enumerate(self.state):
                other_st = tuple(st for v, st in self.state if var != v)
                next_st = sample_discrete(
                    list(range(self.cardinalities[var])),
                    self.transition_models[var][other_st],
                )[0]
                self.state[j] = State(var, next_st)
            sampled[i + 1] = tuple(st for var, st in self.state)

        samples_df = _return_samples(sampled)
        if not include_latents:
            samples_df.drop(self.latents, axis=1, inplace=True)
        return samples_df

    def generate_sample(
        self, start_state=None, size=1, include_latents=False, seed=None
    ):
        """
        Generator version of self.sample

        Returns
        -------
        List of State namedtuples, representing the assignment to all variables of the model.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import MarkovNetwork
        >>> model = MarkovNetwork([('A', 'B'), ('C', 'B')])
        >>> factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(['C', 'B'], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gen = gibbs.generate_sample(size=2)
        >>> [sample for sample in gen]
        [[State(var='C', state=1), State(var='B', state=1), State(var='A', state=0)],
         [State(var='C', state=0), State(var='B', state=1), State(var='A', state=1)]]
        """
        if seed is not None:
            np.random.seed(seed)

        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        for i in range(size):
            for j, (var, st) in enumerate(self.state):
                other_st = tuple(st for v, st in self.state if var != v)
                next_st = sample_discrete(
                    list(range(self.cardinalities[var])),
                    self.transition_models[var][other_st],
                )[0]
                self.state[j] = State(var, next_st)
            if include_latents:
                yield self.state[:]
            else:
                yield [s for s in self.state if i not in self.latents]



def main():
    student = BayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
    cpd_diff = TabularCPD('diff', 2, [[0.6], [0.4]])
    cpd_intel = TabularCPD('intel', 2, [[0.7], [0.3]])
    cpd_grade = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7], [0.9, 0.1, 0.8, 0.3]], ['intel', 'diff'], [2, 2])
    stud = student.add_cpds(cpd_diff, cpd_intel, cpd_grade)
    cpdd = student.get_cpds("grade").variable_card

    gibbs_chain = GibbsSampling(student)
    samples = gibbs_chain.sample(size=3)
    print(samples)

    model = MarkovNetwork([('A', 'B'), ('A', 'C'), ('A', 'D'), ('C', 'B'), ('B', 'D'), ('C', 'D')])
    factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [1, 2, 3, 4])
    factor_ac = DiscreteFactor(['A', 'C'], [2, 2], [9, 10, 11, 12])
    factor_ad = DiscreteFactor(['A', 'D'], [2, 2], [13, 14, 15, 16])
    factor_cb = DiscreteFactor(['C', 'B'], [2, 2], [5, 6, 7, 8])
    factor_bd = DiscreteFactor(['B', 'D'], [2, 2], [1, 2, 3, 4])
    factor_cd = DiscreteFactor(['C', 'D'], [2, 2], [22, 34, 44, 56])
    model.add_factors(factor_ab, factor_cb, factor_ac, factor_ad, factor_bd, factor_cd)
    gibbs = GibbsSampling(model)
    gen = gibbs.sample(size=5)

    print(gen)


if __name__ == "__main__":
    main()