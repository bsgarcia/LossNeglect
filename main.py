#!/usr/bin/python3.6
import argparse
import os
import numpy as np
import multiprocessing as mp
import tqdm
import pickle
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt

import fit.env
from analysis import analysis
from fit.parameters import params, cond
from simulation.models import (
    QLearningAgent,
    AsymmetricQLearningAgent,
    PerseverationQLearningAgent,
    PriorQLearningAgent,
    FullQLearningAgent)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Globals:

    """
    class used in order to
    declare global variables
    accessible from the whole script.
    Instantiated after if __name__ == '__main__'.
    """

    # Continuous parameters bounds
    # --------------------------------------------------------------------- #
    alpha = (0, 1)
    alpha0 = (0, 1)
    alpha1 = (0, 1)
    beta = (1/1000, 1000)
    phi = (-10, 10)
    q = (-1, 1)
    # --------------------------------------------------------------------- #

    qlearning_params = {
            'model': [QLearningAgent, ],
            'labels': ['alpha', 'beta'],
            'guesses': np.array([0.2, 50]),
            'bounds':  [alpha, beta]
    }

    asymmetric_params = {
            'model':  [AsymmetricQLearningAgent, ],
            'labels': ['alpha0', 'alpha1', 'beta'],
            'guesses': np.array([0.3, 0.6, 50]),
            'bounds':  [alpha, alpha, beta]
    }

    perseveration_params = {
            'model':  [PerseverationQLearningAgent, ],
            'labels': ['alpha', 'beta', 'phi'],
            'guesses': np.array([0.5, 50, 5]),
            'bounds':  [alpha, beta, phi]
    }

    prior_params = {
            'model': [PriorQLearningAgent, ],
            'labels': ['alpha', 'beta', 'q'],
            'guesses': np.array([0.5, 50, 0]),
            'bounds':  [alpha, beta, q]
    }

    full_params = {
            'model':  [FullQLearningAgent, ],
            'labels': ['alpha0', 'alpha1', 'beta', 'phi,' 'q'],
            'guesses': np.array([0.5, 0.5, 50, 5, 0]),
            'bounds':  [alpha, alpha, beta, phi, q]
    }

    def __init__(
        self,
        data_path='fit/data/experiment_2/',
        save_path='fit/data/experiment_2_fit/',
        max_evals=10000
    ):

        self.data_path = os.path.abspath(data_path)
        self.save_path = os.path.abspath(save_path)
        self.f_names = sorted([f for f in os.listdir(self.data_path)])
        self.n = len(self.f_names)

        self.max_evals = max_evals
        self.trials_pbar = None

    def init_pbar(self):

        self.trials_pbar = tqdm.tqdm(total=self.max_evals, desc="Optimizing")


def run_fit(*args):

    # --------------------------------------------------------------------- #
    model = args[1][0]
    f_name = args[1][1]
    cog_values = args[0]
    cog_label = args[1][2:]
    cognitive_params = {k: v for k, v in zip(cog_label, cog_values)}
    # print()
    # print(cognitive_params)

    if 'alpha0' in cognitive_params:
        cognitive_params['alpha'] = np.array([
            cognitive_params['alpha0'],
            cognitive_params['alpha1'],
        ])

    p = params.copy()

    # --------------------------------------------------------------------- #

    p['data'] = scipy.io.loadmat(f"{g.data_path}/{f_name}")['data'][0][0]
    p['cognitive_params'] = cognitive_params.copy()
    p['model'] = model

    p['t_max'] = len(p['data'][:, 0])
    p['conds'] = p['data'][:, 2].flatten().astype(int) - 1
    p['dic_conds'] = [cond[int(i) - 1] for i in p['data'][:, 2]]

    # --------------------------------------------------------------------- #

    e = fit.env.Environment(pbar=None, **p)
    results = e.run_fit()

    return results


def comparisons(model, f_name, cognitive_params):

    # --------------------------------------------------------------------- #
    if 'alpha0' in cognitive_params:
        cognitive_params['alpha'] = np.array([
            cognitive_params['alpha0'],
            cognitive_params['alpha1'],
        ])

    p = params.copy()

    # --------------------------------------------------------------------- #

    p['data'] = scipy.io.loadmat(f"{g.data_path}/{f_name}")['data'][0][0]
    p['cognitive_params'] = cognitive_params.copy()
    p['model'] = model

    p['t_max'] = len(p['data'][:, 0])
    p['conds'] = p['data'][:, 2].flatten().astype(int) - 1
    p['dic_conds'] = [cond[int(i) - 1] for i in p['data'][:, 2]]

    # --------------------------------------------------------------------- #

    e = fit.env.Environment(**p)
    e.run()


def run_subject(
        file,
        qlearning_params=Globals.qlearning_params,
        asymmetric_params=Globals.asymmetric_params,
        perseveration_params=Globals.perseveration_params,
        prior_params=Globals.prior_params,
        full_params=Globals.full_params):

    # if os.path.exists(f'{g.save_path}/{file}'):
    #     print('File already exists.')
    #     return

    f_name = [file, ]

    # fit Qlearning
    qlearning_best = scipy.optimize.basinhopping(
        run_fit,
        x0=qlearning_params['guesses'],
        # bounds=qlearning_params['bounds'],
        minimizer_kwargs=dict(
            args=qlearning_params['model'] + f_name + qlearning_params['labels'],
            bounds=qlearning_params['bounds'],
            method="L-BFGS-B",
            options={'maxiter': 20000}
        ))

    print(qlearning_best)

    g.trials_pbar.update()

    # # fit AsymmetricQLearning
    # asymmetric_trials = Trials()
    # asymmetric_best = hp.fmin(
    #     fn=run_fit,
    #     params=asymmetric_params + f_name,
    #     algo=hp.tpe.suggest,
    #     max_evals=g.max_evals,
    #     trials=asymmetric_trials
    # )
    # asymmetric_best['likelihood'] = min(asymmetric_trials.losses())
    # g.trials_pbar.update()
    #
    # # fit PerseverationQLearning
    # perseveration_trials = Trials()
    # perseveration_best = hp.fmin(
    #     fn=run_fit,
    #     params=perseveration_params + f_name,
    #     algo=hp.tpe.suggest,
    #     max_evals=g.max_evals,
    #     trials=perseveration_trials
    # )
    # perseveration_best['likelihood'] = min(perseveration_trials.losses())
    # g.trials_pbar.update()
    #
    # # fit PriorQLearningAgent
    # prior_trials = Trials()
    # prior_best = hp.fmin(
    #     fn=run_fit,
    #     params=prior_params + f_name,
    #     algo=hp.tpe.suggest,
    #     max_evals=g.max_evals,
    #     trials=prior_trials
    # )
    # prior_best['likelihood'] = min(prior_trials.losses())
    # g.trials_pbar.update()
    #
    # # fit FullQLearningAgent
    # full_trials = Trials()
    # full_best = hp.fmin(
    #     fn=run_fit,
    #     params=full_params + f_name,
    #     algo=hp.tpe.suggest,
    #     max_evals=g.max_evals,
    #     trials=full_trials
    # )
    #
    # full_best['likelihood'] = min(full_trials.losses())
    # g.trials_pbar.update()

    with open(f'{g.save_path}/{file}'.replace('.mat', '.p'), 'wb') as f:
        pickle.dump(dict(
            qlearning={k: v for k, v in zip(qlearning_params['labels'], qlearning_best.x)},
            # asymmetric=asymmetric_best,
            # perseveration=perseveration_best,
            # prior=prior_best,
            # full=full_best
        ), file=f)

    scipy.io.savemat(f'{g.save_path}/{file}', dict(
        qlearning={k: v for k, v in zip(qlearning_params['labels'], qlearning_best.x)},
        # asymmetric=asymmetric_best,
        # perseveration=perseveration_best,
        # prior=prior_best,
        # full=full_best
    ))


def fitting():

    # run_subject(file=g.f_names[0])
    # with mp.Pool() as p:
    #     for _ in p.imap_unordered(run_subject, g.f_names):
    #         pass
    g.init_pbar()
    run_subject(file=g.f_names[0])


def run_analysis():

    tqdm.tqdm.write('Generating graphs...')

    analysis.run()


if __name__ == '__main__':

    g = Globals()

    # Parse cli arguments
    # ------------------------------------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", action="store_true",
                        help="run simulations")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="run analysis and display figures")
    parser.add_argument("-o", "--optimize", action="store_true",
                        help="optimize parameters")
    p_args = parser.parse_args()

    is_running_in_pycharm = "PYCHARM_HOSTED" in os.environ

    # if args.simulation:
    #     run_simulation()

    # comparisons(
    #     model=QLearningAgent,
    #         f_name=g.f_names[2],
    #         cognitive_params=pickle.load(
    #             open(f'fit/data/experiment_2_fit/{g.f_names[2].replace(".mat", ".p")}', 'rb'))['qlearning']
    # )
    if p_args.optimize:
        fitting()

    if p_args.analysis:
        run_analysis()

    if not p_args.simulation and not p_args.analysis and not is_running_in_pycharm:
        parser.print_help()

    if is_running_in_pycharm:
        run_analysis()
