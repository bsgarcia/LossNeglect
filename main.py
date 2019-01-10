# import matlab.engine
import pickle
import argparse
import os
import numpy as np
import tqdm
import multiprocessing as ml
import pyfmincon.opt

import fit.env
import simulation.env
from fit.parameters import params
import simulation.parameters
from fit.parameters import Globals
from simulation.models import (
    QLearning,
    AsymmetricQLearning,
    PerseverationQLearning,
    PriorQLearning,
    AsymmetricPriorQLearning,
    FullQLearning)
import fit.data

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)


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


def run_fmincon(f, model_params, options, optional_args, eng=None):

    # ---- set options ------- # 
    x0 = model_params['guesses']
    lb = [i[0] for i in model_params['bounds']]
    ub = [i[1] for i in model_params['bounds']]

    # Minimize
    xopt, fval, exitflag = pyfmincon.opt.fmincon(f, x0=x0, lb=lb, ub=ub,
                                                 options=options, optional_args=optional_args, engine=eng)

    values = [item for sublist in [xopt, [fval]] for item in sublist]
    return {k: v for k, v in zip(model_params['labels'], values)}


def run_fit_subject(subject_id):

    eng = pyfmincon.opt.new_engine()

    if Globals.refit:

        to_save = {}

        # We fit model_id
        for model_id, model_params in enumerate(Globals.model_params):

            to_save[model_params['model'].__name__] = {}

            #  On fit_model_id data (produced with the model)
            for fit_model_id, fit_model_params in enumerate(Globals.model_params):

                to_save[model_params['model'].__name__][fit_model_params['model'].__name__] = \
                    run_fmincon(
                        f='main.run_fit',
                        model_params=model_params,
                        options=Globals.options,
                        optional_args=[subject_id, model_id, fit_model_id],
                        eng=eng
                    )

    else:

        to_save = {}
        for model_id, model_params in enumerate(Globals.model_params):
            to_save[model_params['model'].__name__] = run_fmincon(
                f='main.run_fit',
                model_params=model_params,
                options=Globals.options,
                optional_args=[subject_id, model_id],
                eng=eng
            )

    eng.quit()
    return to_save


def fitting():

    folder = 'fit_data' if Globals.reg_fit else 'refit_data'

    save_path = os.path.abspath(
        f'fit/data/pooled/{folder}/{Globals.experiment_id}{Globals.condition}.p'
    )

    data = []
    pl = ml.Pool(processes=8)
    for res in tqdm.tqdm(
            pl.imap(func=run_fit_subject, iterable=Globals.subject_ids),
            total=Globals.n_subjects
    ):
        data.append(res)

    # else:
    #
    #     pyfmincon.opt.start()
    #     for subject_id in tqdm.tqdm(Globals.subject_ids):
    #         data.append(run_fit_subject(subject_id, new_eng=False))
    #     pyfmincon.opt.stop()

    with open(save_path, 'wb') as f:
        pickle.dump(file=f, obj=data)


def run_simulations():

    #  --------------------- Run Status quo 1 ----------------------------------- # 
    data = fit.data.fit()
    new_data = []
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

        new_data.append(env.run())

    with open(f'fit/data/pooled/sim_data/{Globals.experiment_id}status_quo_1.p', 'wb') as f:
        pickle.dump(new_data, file=f)

    #  --------------------- Run Status quo 2 ----------------------------------- # 
    new_data = []
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
        new_data.append(env.run())

    with open(f'fit/data/pooled/sim_data/{Globals.experiment_id}status_quo_2.p', 'wb') as f:
        pickle.dump(new_data, file=f)

    #  --------------------- Run risk ----------------------------------- # 
    new_data = []
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
        new_data.append(env.run())

    with open(f'fit/data/pooled/sim_data/{Globals.experiment_id}risk.p', 'wb') as f:
        pickle.dump(new_data, file=f)


def run_analysis():
    from analysis import analysis
    tqdm.tqdm.write('Generating graphs...')
    # --------------------------------------------------------------------------------------------- #
    first_fit = fit.data.fit()
    # analysis.params_model_comparisons(first_fit)
    analysis.single_model_selection(first_fit, title='')
    # --------------------------------------------------------------------------------------------- #
    refit_risk = fit.data.refit(condition='_risk')
    analysis.model_selection_recovery(refit_risk, title='Risky condition')
    # --------------------------------------------------------------------------------------------- #
    refit_status_quo_1 = fit.data.refit(condition='_status_quo_1')
    analysis.model_selection_recovery(refit_status_quo_1,
                                      title='Status quo "either 1/2/3 reversals" condition')
    # --------------------------------------------------------------------------------------------- #
    refit_status_quo_2 = fit.data.refit(condition='_status_quo_2')
    analysis.model_selection_recovery(
        refit_status_quo_2,
        title='Status quo "1 reversal either at the beginning, the middle, the end" condition')


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
