#!/usr/bin/python3.6
import pickle
import argparse
import os
import numpy as np
import tqdm
import scipy.io
import pyfmincon
import pyfmincon.opt

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
    """

    # Continuous parameters bounds
    # --------------------------------------------------------------------- #
    alpha = (1/1000, 1)
    beta = (1/1000, 1000)
    phi = (-10, 10)
    q = (-1, 1)

    qlearning_params = {
        'model': QLearningAgent,
        'labels': ['alpha', 'beta'],
        'guesses': np.array([0.5, 50]),
        'bounds': np.array([alpha, beta])
    }

    asymmetric_params = {
        'model': AsymmetricQLearningAgent,
        'labels': ['alpha0', 'alpha1', 'beta'],
        'guesses': np.array([0.5, 0.5, 1]),
        'bounds': np.array([alpha, alpha, beta])
    }

    perseveration_params = {
        'model': PerseverationQLearningAgent,
        'labels': ['alpha', 'beta', 'phi'],
        'guesses': np.array([0.5, 50, 5]),
        'bounds': np.array([alpha, beta, phi])
    }

    prior_params = {
        'model': PriorQLearningAgent,
        'labels': ['alpha', 'beta', 'q'],
        'guesses': np.array([0.5, 50, 0]),
        'bounds': np.array([alpha, beta, q])
    }

    full_params = {
        'model': FullQLearningAgent,
        'labels': ['alpha0', 'alpha1', 'beta', 'phi', 'q'],
        'guesses': np.array([0.5, 0.5, 50, 5, 0]),
        'bounds': np.array([alpha, alpha, beta, phi, q])
    }

    model_params = [
        qlearning_params,
        asymmetric_params,
        perseveration_params,
        prior_params,
        full_params
    ]

    data_path = os.path.abspath('fit/data/experiment_1/')
    save_path = os.path.abspath('fit/data/experiment_1_fit/')

    file = os.path.abspath('fit/data/pooled/experiment1.p')

    with open(file, 'rb') as f:
        data = pickle.load(f)

    n_subjects = len(data)

    max_evals = 10000

    subject_ids = data.keys()

    options = {
        'AlwaysHonorConstraints': 'bounds',
        # 'display': 'iter-detailed',
        'MaxIter': max_evals,
        'MaxFunEvals': max_evals,
        'display': 'off',
        # 'Diagnostics': 'on'
    }


def run_fit(x0, optional_args):

    # --------------------------------------------------------------------- #
    subject_id = optional_args[0]
    model_id = optional_args[1]
    # --------------------------------------------------------------------- #
    model_param = Globals.model_params[model_id]
    cognitive_params = {k: v for k, v in zip(model_param['labels'], x0)}
    if 'alpha0' in cognitive_params:
        cognitive_params['alpha'] = np.array([
            cognitive_params['alpha0'],
            cognitive_params['alpha1']
        ])
    # --------------------------------------------------------------------- #
    p = params.copy()
    p['data'] = Globals.data[subject_id]
    p['model'] = model_param['model']
    p['cognitive_params'] = cognitive_params.copy()
    p['t_max'] = len(p['data'][:, 0])
    p['conds'] = p['data'][:, 2].flatten().astype(int) - 1
    p['dic_conds'] = [cond[int(i) - 1] for i in p['data'][:, 2]]
    # --------------------------------------------------------------------- #
    e = fit.env.Environment(**p)
    neg_log_likelihood = e.run()
    return neg_log_likelihood


def run_fmincon(f, model_params, options, optional_args):

    # ---- set options ------- # 
    x0 = model_params['guesses']
    lb = [i[0] for i in model_params['bounds']]
    ub = [i[1] for i in model_params['bounds']]

    # Minimize
    xopt = pyfmincon.opt.fmincon(f, x0=x0, lb=lb, ub=ub,
                                 options=options, optional_args=optional_args)

    return {k: v for k, v in zip(model_params['labels'], xopt)}


def run_subject(subject_id):

    to_save = {}

    for model_id, model_params in enumerate(Globals.model_params):

        to_save[model_params['model'].__name__] = run_fmincon(
                f='main.run_fit',
                model_params=model_params,
                options=Globals.options,
                optional_args=[subject_id, model_id]
        )

    with open(f'{Globals.save_path}/{subject_id}.p', 'wb') as f:
        pickle.dump(to_save, file=f)

    scipy.io.savemat(mdict=to_save, file_name=f'{Globals.save_path}/{subject_id}.mat')


def fitting():

    pyfmincon.opt.start()

    for idx in tqdm.tqdm(Globals.subject_ids, desc="Optimizing"):
        run_subject(subject_id=idx)

    pyfmincon.opt.stop()


def run_analysis():
    tqdm.tqdm.write('Generating graphs...')

    analysis.run()


if __name__ == '__main__':

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

    if p_args.optimize:
        fitting()

    if p_args.analysis:
        run_analysis()

    # if not p_args.simulation and not p_args.analysis and not is_running_in_pycharm:
    #     parser.print_help()
    #
    # if is_running_in_pycharm:
    #     run_analysis()
