#!/usr/bin/python3.6
import argparse
import os
import numpy as np
import hyperopt as hp
from hyperopt import Trials
import multiprocessing as mp
import tqdm
import pickle
import scipy.io as sp

import fit.env
from analysis import analysis
from fit.parameters import params, cond
from simulation.models import (
    QLearningAgent,
    AsymmetricQLearningAgent,
    PerseverationQLearningAgent,
    PriorQLearningAgent,
    FullQLearningAgent)


class Globals:

    """
    class used in order to
    declare global variables
    accessible from the whole script.
    Instantiated after if __name__ == '__main__'.
    """

    def __init__(
        self,
        data_path='fit/data/experiment_1/',
        save_path='fit/data/experiment_1_fit/',
        max_evals=1500
    ):

        self.data_path = os.path.abspath(data_path)
        self.save_path = os.path.abspath(save_path)
        self.f_names = sorted([f for f in os.listdir(self.data_path)])
        self.n = len(self.f_names)

        models = [
            QLearningAgent,
            AsymmetricQLearningAgent,
            PerseverationQLearningAgent,
            PriorQLearningAgent,
            FullQLearningAgent,
        ]

        self.max_evals = max_evals

        self.trials_pbar = tqdm.tqdm(
            total=self.max_evals*len(models), desc="Evals"
        )


def run_fit(*args):

    # --------------------------------------------------------------------- #

    model = args[0][0]['model']
    f_name = args[0][2]['f_name']
    cognitive_params = args[0][1]['params']

    p = params.copy()

    # --------------------------------------------------------------------- #

    p['data'] = sp.loadmat(f'{g.data_path}/{f_name}')['data']
    p['cognitive_params'] = cognitive_params.copy()
    p['model'] = model

    p['t_max'] = len(p['data'][:, 0])
    p['conds'] = [cond[int(i) - 1] for i in p['data'][:, 2]]

    # --------------------------------------------------------------------- #

    e = fit.env.Environment(pbar=None, **p)
    results = e.run()

    g.trials_pbar.update()

    return results


def run_subject(
        file,
        qlearning_space,
        asymmetric_space,
        perseveration_space,
        prior_space,
        full_space):

    f_name = [{'f_name': file}]

    # fit Qlearning
    qlearning_trials = Trials()
    qlearning_best = hp.fmin(
        fn=run_fit,
        space=qlearning_space + f_name,
        algo=hp.tpe.suggest,
        max_evals=g.max_evals,
        trials=qlearning_trials
    )
    qlearning_best['likelihood'] = min(qlearning_trials.losses())

    # fit AsymmetricQLearning
    asymmetric_trials = Trials()
    asymmetric_best = hp.fmin(
        fn=run_fit,
        space=asymmetric_space + f_name,
        algo=hp.tpe.suggest,
        max_evals=g.max_evals,
        trials=asymmetric_trials
    )
    asymmetric_best['likelihood'] = min(asymmetric_trials.losses())

    # fit PerseverationQLearning
    perseveration_trials = Trials()
    perseveration_best = hp.fmin(
        fn=run_fit,
        space=perseveration_space + f_name,
        algo=hp.tpe.suggest,
        max_evals=g.max_evals,
        trials=perseveration_trials
    )
    perseveration_best['likelihood'] = min(perseveration_trials.losses())

    # fit PriorQLearningAgent
    prior_trials = Trials()
    prior_best = hp.fmin(
        fn=run_fit,
        space=prior_space + f_name,
        algo=hp.tpe.suggest,
        max_evals=g.max_evals,
        trials=prior_trials
    )
    prior_best['likelihood'] = min(prior_trials.losses())

    # fit FullQLearningAgent
    full_trials = Trials()
    full_best = hp.fmin(
        fn=run_fit,
        space=full_space + f_name,
        algo=hp.tpe.suggest,
        max_evals=g.max_evals,
        trials=full_trials
    )

    full_best['likelihood'] = min(full_trials.losses())

    with open(f'{g.save_path}/{i}{file}'.replace('.mat', '.p'), 'wb') as f:
        pickle.dump(dict(
            qlearning=qlearning_best,
            asymmetric=asymmetric_best,
            perseveration=perseveration_best,
            prior=prior_best,
            full=full_best
        ), file=f)

    sp.savemat(f'{g.save_path}/{i}{file}', dict(
        qlearning=qlearning_best,
        asymmetric=asymmetric_best,
        perseveration=perseveration_best,
        prior=prior_best,
        full=full_best
    ))


def fitting():

    # Discrete parameter space
    # --------------------------------------------------------------------- #
    alpha_space = list(np.around(np.linspace(0.05, 1, 20), decimals=2))
    alpha0_space = list(np.around(np.linspace(0.05, 1, 20), decimals=2))
    alpha1_space = list(np.around(np.linspace(0.05, 1, 20), decimals=2))
    phi_space = list(np.around(np.linspace(0.5, 10, 20), decimals=2))
    q_space = list(np.around(np.linspace(0, 1, 20), decimals=2))

    alpha = hp.hp.choice('alpha', alpha_space)
    alpha0 = hp.hp.choice('alpha0', alpha0_space)
    alpha1 = hp.hp.choice('alpha1', alpha1_space)
    beta = hp.hp.quniform('beta', 0.5, 300, 5)
    phi = hp.hp.choice('phi', phi_space)
    q0 = hp.hp.choice('q0', q_space)
    q1 = hp.hp.choice('q1', q_space)

    # Continuous parameters space
    # --------------------------------------------------------------------- #
    alpha = hp.hp.uniform('alpha', 0, 1)
    alpha0 = hp.hp.uniform('alpha0', 0, 1)
    alpha1 = hp.hp.uniform('alpha1', 0, 1)
    beta = hp.hp.uniform('beta', 1/1000, 400)
    phi = hp.hp.uniform('phi', -10, 10)
    q0 = hp.hp.uniform('q0', -1, 1)
    q1 = hp.hp.uniform('q1', -1, 1)

    # --------------------------------------------------------------------- #

    qlearning_space = [
        {'model': QLearningAgent},
        {'params':
            {
                'alpha': alpha,
                'beta': beta,
            }
        }
    ]

    asymmetric_space = [
        {'model': AsymmetricQLearningAgent},
        {'params': {
            'alpha': [alpha0, alpha1],
            'beta': beta
        }}
    ]

    perseveration_space = [
        {'model': PerseverationQLearningAgent},
        {'params': {
            'alpha': alpha,
            'beta': beta,
            'phi': phi
        }}
    ]

    prior_space = [
        {'model': PriorQLearningAgent},
        {'params': {
            'alpha': alpha,
            'beta': beta,
            'q': [q0, q1]
        }}
    ]

    full_space = [
        {'model': FullQLearningAgent},
        {'params': {
            'alpha': [alpha0, alpha1],
            'beta': beta,
            'phi': phi,
            'q': [q0, q1]
        }}
    ]

    p = mp.Pool(processes=g.f_names)




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

    if p_args.optimize:
        fitting()

    if p_args.analysis:
        run_analysis()

    if not p_args.simulation and not p_args.analysis and not is_running_in_pycharm:
        parser.print_help()

    if is_running_in_pycharm:
        run_analysis()
