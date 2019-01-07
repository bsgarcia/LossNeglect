import pickle
import re
import argparse
import os
import numpy as np
import tqdm
import scipy.io
import pyfmincon
import pyfmincon.opt

import fit.env
import simulation.env
from analysis import analysis
from fit.parameters import params
import simulation.parameters
from simulation.models import (
    QLearning,
    AsymmetricQLearning,
    PerseverationQLearning,
    PriorQLearning,
    AsymmetricPriorQLearning,
    FullQLearning)
import multiprocessing as ml
import mail
import fit.data

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)


def load_agent_fit(data_path):
    files = [data_path + f'/{fname}' for fname in os.listdir(data_path)]
    subject_ids = [int(re.search('(\d+)(?:.p)', f).group(1)) for f in files]
    data = {}
    for s_id, file in zip(subject_ids, files):
        with open(file, 'rb') as f:
            d = pickle.load(f)
            data.update({s_id: d})
    return data


class Globals:
    """
    class used in order to
    declare global variables
    accessible from the whole script.
    """

    # Continuous parameters bounds
    # --------------------------------------------------------------------- #
    alpha_bounds = (1/1000, 1)
    beta_bounds = (1, 1000)
    phi_bounds = (-10, 10)
    q_bounds = (-1, 1)
    q_fixed_bounds = (-1, -1)

    # parameters initial guesses
    # --------------------------------------------------------------------- #
    alpha_guess = 0.5
    beta_guess = 1
    phi_guess = 0
    q_guess = 0
    q_fixed_guess = -1

    qlearning_params = {
        'model': QLearning,
        'labels': ['alpha', 'beta', 'log'],
        'guesses': np.array([alpha_guess, beta_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds])
    }

    asymmetric_params = {
        'model': AsymmetricQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds])
    }

    perseveration_params = {
        'model': PerseverationQLearning,
        'labels': ['alpha', 'beta', 'phi', 'log'],
        'guesses': np.array([alpha_guess, beta_guess, phi_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds, phi_bounds])
    }

    prior_params = {
        'model': PriorQLearning,
        'labels': ['alpha', 'beta', 'q', 'log'],
        'guesses': np.array([alpha_guess, beta_guess, q_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds, q_bounds])
    }

    asymmetric_prior_params = {
        'model': AsymmetricPriorQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'q', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess, q_fixed_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds, q_fixed_bounds]),
    }

    full_params = {
        'model': FullQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'phi', 'q', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess, phi_guess, q_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds, phi_bounds, q_bounds])
    }

    model_params = [
        qlearning_params,
        asymmetric_params,
        perseveration_params,
        prior_params,
        asymmetric_prior_params,
        full_params
    ]

    reg_fit = False
    refit = True
    fit_agents = False
    fit_subjects = True

    assertion_error = 'Only one parameter among {} and {} can be true.'
    assert sum([reg_fit, refit]) == 1, assertion_error.format('reg_fit', 'refit')
    assert sum([fit_agents, fit_subjects]) == 1, assertion_error.format('fit_agents', 'fit_subjects')

    experiment_id = 'full'
    fit_condition = ''

    data = fit.data.fit()

    n_subjects = len(data)
    subject_ids = range(len(data))

    max_evals = 10000

    options = {
        # 'AlwaysHonorConstraints': 'bounds',
        'Algorithm': 'interior-point',
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
    if Globals.reg_fit:
        p = params.copy()
        p['data'] = Globals.data[subject_id]
        p['model'] = model_param['model']
        p['cognitive_params'] = cognitive_params.copy()
        p['t_max'] = int(max(p['data'][:, 1]))
        p['exp_id'] = Globals.experiment_id
    # --------------------------------------------------------------------- #
    else:
        model_fit_id = Globals.model_params[optional_args[2]]['model'].__name__
        p = params.copy()

        p['data'] = Globals.data[subject_id][model_fit_id]

        p['choices'] = p['data']['choices']
        p['rewards'] = p['data']['rewards']
        p['conds'] = p['data']['conds']

        p['model'] = model_param['model']
        p['cognitive_params'] = cognitive_params.copy()
        p['t_max'] = len(p['choices'])
        p['exp_id'] = Globals.experiment_id

    e = fit.env.Environment(**p)
    neg_log_likelihood = e.run()
    return neg_log_likelihood


def run_fmincon(f, model_params, options, optional_args):

    # ---- set options ------- # 
    x0 = model_params['guesses']
    lb = [i[0] for i in model_params['bounds']]
    ub = [i[1] for i in model_params['bounds']]

    # Minimize
    xopt, fval, exitflag = pyfmincon.opt.fmincon(f, x0=x0, lb=lb, ub=ub,
                                       options=options, optional_args=optional_args)

    values = [item for sublist in [xopt, [fval]] for item in sublist]
    return {k: v for k, v in zip(model_params['labels'], values)}


def run_fit_subject(subject_id):

    if Globals.refit:
        to_save = {}
        for model_id, model_params in enumerate(Globals.model_params):
            to_save[model_params['model'].__name__] = {}
            for fit_model_id, fit_model_params in enumerate(Globals.model_params):
                to_save[model_params['model'].__name__][fit_model_params['model'].__name__] = run_fmincon(
                    f='main.run_fit',
                    model_params=model_params,
                    options=Globals.options,
                    optional_args=[subject_id, model_id, fit_model_id]
                )

    else:
        to_save = {}
        for model_id, model_params in enumerate(Globals.model_params):
            to_save[model_params['model'].__name__] = run_fmincon(
                f='main.run_fit',
                model_params=model_params,
                options=Globals.options,
                optional_args=[subject_id, model_id]
            )

    return to_save


def fitting():

    pyfmincon.opt.start()

    data = []
    for subject_id in tqdm.tqdm(Globals.subject_ids, desc='Optimizing'):
        data.append(run_fit_subject(subject_id))

    with open(Globals.save_path, 'wb') as f:
        pickle.dump(file=f, obj=data)
    # mail.auto_send(job_name='fit_recover', main_file=__name__, attachment=Globals.save_path + '/full.p')
    pyfmincon.opt.stop()


def run_simulations():

    #  --------------------- Run Status quo 2 ----------------------------------- # 
    data = Globals.load_subject_fit(f'fit/data/experiment{Globals.experiment_id}_fit/')
    p = simulation.parameters.params.copy()
    cond = simulation.parameters.cond.copy()
    p['t_max'] = cond['status_quo']['t_max']
    p['conds'] = np.repeat([0, 1, 2, 3], 48)
    p['t_when_reversal_occurs'] = cond['status_quo']['t_reversal_equal']
    p['rewards'] = cond['status_quo']['rewards']
    p['p'] = cond['status_quo']['p']
    p['condition'] = 'status_quo_1'
    p['experiment_id'] = Globals.experiment_id

    for subject_id in Globals.subject_ids:

        p['cognitive_params'] = data[subject_id].copy()
        p['subject_id'] = subject_id

        env = simulation.env.Environment(**p)

        env.run()

    #  --------------------- Run Status quo 2 ----------------------------------- # 
    p = simulation.parameters.params.copy()
    cond = simulation.parameters.cond.copy()
    p['t_max'] = cond['status_quo']['t_max']
    p['conds'] = np.repeat([0, 1, 2, 3], 48)
    p['t_when_reversal_occurs'] = cond['status_quo']['t_reversal_bme']
    p['rewards'] = cond['status_quo']['rewards']
    p['p'] = cond['status_quo']['p']
    p['condition'] = 'status_quo_2'
    p['experiment_id'] = Globals.experiment_id

    for subject_id in Globals.subject_ids:

        p['cognitive_params'] = data[subject_id].copy()
        p['subject_id'] = subject_id

        env = simulation.env.Environment(**p)
        env.run()

    #  --------------------- Run risk ----------------------------------- # 
    p = simulation.parameters.params.copy()
    cond = simulation.parameters.cond.copy()
    p['t_max'] = cond['risk']['t_max']
    p['conds'] = np.repeat([0, 1, 2, 3], 48)
    p['dic_conds'] = cond['risk']['conds'].copy()
    p['condition'] = 'risk'
    p['experiment_id'] = Globals.experiment_id

    for subject_id in Globals.subject_ids:

        p['cognitive_params'] = data[subject_id].copy()
        p['subject_id'] = subject_id

        env = simulation.env.Environment(**p)
        env.run()


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

    if p_args.simulation:
        run_simulations()

    if p_args.analysis:
        run_analysis()

    if not any(vars(p_args).values()) and not is_running_in_pycharm:
        parser.print_help()

    if is_running_in_pycharm:
        run_analysis()
