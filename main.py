#!/usr/bin/python3.6
import argparse
import os
import numpy as np
import hyperopt as hp
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
    PriorQLearningAgent)


# --------------------------------------------------------- #
data_path = os.path.abspath('fit/data/experiment_2/')
save_path = os.path.abspath('fit/data/experiment_2_fit/')

f_names = sorted([f for f in os.listdir(data_path)])
n = len(f_names)
data_gen = iter(f_names)
models = iter(
    [
        QLearningAgent,
        AsymmetricQLearningAgent,
        PerseverationQLearningAgent,
        PriorQLearningAgent,
    ] * n
)

model = None
data = None
trials_pbar = tqdm.tqdm(total=1000, desc="Evals")
# --------------------------------------------------------- #


# def run(*args):
#
#     n_reversal, t_max = args[0][:]
#
#     risk_positive['t_max'] = int(t_max)
#
#     risk_positive['t_when_reversal_occurs'] = \
#             np.arange(
#                 0,
#                 t_max + 1,
#                 t_max // (n_reversal + 1),
#             )[1:-1]
#
#     e = simulation.env.Environment(pbar=pbar, **risk_positive)
#     a = e.run()
#     pbar.update()
#
#     return a


def run_fit(*args):

    # --------------------------------------------------------------------- #

    args = args[0]

    p = params.copy()

    cog_params = p['cognitive_params'][model.__name__]

    # --------------------------------------------------------------------- #

    if model == QLearningAgent:

        cog_params['alpha'], cog_params['beta'] = args[:]

    elif model == AsymmetricQLearningAgent:

        cog_params['alpha'], cog_params['beta'] = \
            np.array([args[0], args[1]]), args[2]

    elif model == PerseverationQLearningAgent:

        cog_params['alpha'], cog_params['beta'], cog_params['phi'] = \
            args[:]

    else:

        cog_params['alpha'], cog_params['beta'], cog_params['q'] = \
            args[0], args[1], np.array([args[2], args[3]])

    # --------------------------------------------------------------------- #

    p['data'] = sp.loadmat(f'{data_path}/{data}')['data'][0][0]
    p['t_max'] = len(p['data'][:, 0])
    p['model'] = model
    p['conds'] = [cond[int(i) - 1] for i in p['data'][:, 2]]

    # --------------------------------------------------------------------- #

    e = fit.env.Environment(pbar=None, **p)
    results = e.run()

    trials_pbar.update()

    return results


def fitting():

    global model, data

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
    beta = hp.hp.quniform('beta', 0.5, 300, 5)
    phi = hp.hp.uniform('phi', 0, 10)
    q0 = hp.hp.uniform('q0', -1, 1)
    q1 = hp.hp.uniform('q1', -1, 1)

    # --------------------------------------------------------------------- #

    qlearning_labels = ['alpha', 'beta']
    asymmetric_labels = ['alpha0', 'alpha1', 'beta']
    perseveration_labels = ['alpha', 'beta', 'phi']
    prior_labels = ['alpha', 'beta', 'q0', 'q1']

    # --------------------------------------------------------------------- #

    qlearning_space = [
        alpha,
        beta
    ]

    asymmetric_space = [
        alpha0,
        alpha1,
        beta
    ]

    perseveration_space = [
        alpha,
        beta,
        phi
    ]

    prior_space = [
        alpha,
        beta,
        q0,
        q1
    ]

    # --------------------------------------------------------------------- #

    model = next(models)

    max_evals = 2000

    data = next(data_gen)

    for i, j in tqdm.tqdm(enumerate([f_names[0], ] * 3), desc="subjects"):

        trials_pbar.update(0)

        # fit Qlearning
        qlearning = hp.fmin(
            fn=run_fit,
            space=qlearning_space,
            algo=hp.tpe.suggest,
            max_evals=max_evals
        )

        # model = next(models)

        # fit AsymmetricQLearning
        # asymmetric = hp.fmin(
        #     fn=run_fit,
        #     space=asymmetric_space,
        #     algo=hp.tpe.suggest,
        #     max_evals=max_evals
        # )
        #
        # model = next(models)

        # fit PerseverationQLearning
        # perseveration = hp.fmin(
        #     fn=run_fit,
        #     space=perseveration_space,
        #     algo=hp.tpe.suggest,
        #     max_evals=max_evals
        # )
        #
        # model = next(models)

        # fit PerseverationQLearning
        # prior = hp.fmin(
        #     fn=run_fit,
        #     space=prior_space,
        #     algo=hp.tpe.suggest,
        #     max_evals=max_evals
        # )

        qlearning = {
            k: v for k, v in zip(qlearning_labels, hp.space_eval(qlearning_space, qlearning))
        }

        # asymmetric = {
        #     k: v for k, v in zip(asymmetric_labels, hp.space_eval(asymmetric_space, asymmetric))
        # }
        # perseveration = {
        #     k: v for k, v in zip(perseveration_labels, hp.space_eval(perseveration_space, perseveration))
        # }
        #
        # prior = {
        #     k: v for k, v in zip(prior_labels, hp.space_eval(prior_space, prior))
        # }

        pickle.dump(dict(
            qlearning=qlearning,
            # asymmetric=asymmetric,
            # perseveration=perseveration,
            # prior=prior
        ), file=open(f'{save_path}/{i}{j}.p', 'wb'))


def run_analysis():

    tqdm.tqdm.write('Generating graphs...')

    analysis.run()


if __name__ == '__main__':

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
