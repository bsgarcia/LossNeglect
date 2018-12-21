import pickle
import numpy as np
from analysis import graph
import scipy.stats
from fit.parameters import params
import os


def compute_aic_bic(ll, n_obs, n_params):
    global eng
    aic, bic = eng.aicbic(ll, n_params, n_obs, nargout=2)
    return aic[0], bic[0]


def model_selection_recovery(experiment_id='_full', n_obs=96):
    global eng
    import matlab.engine
    import matlab

    eng = matlab.engine.start_matlab()

    for condition in ('status_quo_1', 'status_quo_2', 'risk'):

        data = regroup_by_model(experiment_id, condition='risk')
        n_obs = 4 * 48
        n_subjects = len(data['QLearning']['log'])

        new_data = {}

        for model in data.keys():

            for model in data.keys():

                ll = matlab.double([-i for i in data[model].pop('log')])
                n_obs_vector = matlab.double([n_obs, ] * n_subjects)
                n_params = matlab.double([int(len(data[model].keys())), ] * n_subjects)

                aic, bic = compute_aic_bic(
                    ll=ll,
                    n_obs=n_obs_vector,
                    n_params=n_params
                )

                new_data[model] = {'mean_std': ({model: np.mean(bic)}, {model: scipy.stats.sem(bic)})}

            graph.bar_plot_model_comparison(data=new_data, data_scatter=None, ylabel='value', title=f'BIC exp={experiment_id}, condition={condition}')


def single_model_selection(experiment_id='_full', condition=''):
    global eng
    import matlab.engine
    import matlab

    eng = matlab.engine.start_matlab()

    data = regroup_by_model(experiment_id, condition)
    n_obs = 4 * 48
    n_subjects = len(data['QLearning']['log'])

    new_data = {}

    for model in data.keys():

        ll = matlab.double([-i for i in data[model].pop('log')])
        n_obs_vector = matlab.double([n_obs, ] * n_subjects)
        n_params = matlab.double([int(len(data[model].keys())), ] * n_subjects)

        aic, bic = compute_aic_bic(
            ll=ll,
            n_obs=n_obs_vector,
            n_params=n_params
        )

        new_data[model] = {'mean_std': ({model: np.mean(bic)}, {model: scipy.stats.sem(bic)})}

    graph.bar_plot_model_comparison(data=new_data, data_scatter=None, ylabel='value', title=f'BIC exp={experiment_id}, condition={condition}')


def params_model_comparisons(experiment_id='_full'):

    data = regroup_by_model(experiment_id=experiment_id, condition='')

    for model in data.keys():

        new_data_model = {}

        for k in data[model].keys():
            if k == 'beta':
                new_data_model[k] = 1/np.asarray(data[model][k])
            elif k in ('log', 'phi'):
                new_data_model[k] = np.asarray(data[model][k]) * 1/max(data[model][k])
            else:
                new_data_model[k] = data[model][k]

        mean = {k: np.mean(v) for k, v in new_data_model.items()}
        std = {k: scipy.stats.sem(v) for k, v in new_data_model.items()}
        data[model] = {'scatter': None, 'mean_std': (mean, std)}

    graph.bar_plot_model_comparison(data=data, data_scatter=None, ylabel='value')


def run():

    single_model_selection()
    params_model_comparisons()


# ------------------------------------------------------- Utils---------------------------------------------- # 


def regroup_by_model(experiment_id='_full', condition=""):

    if not condition:
         data = [load(f'fit/data/experiment{experiment_id}_fit/{fname}')
            for fname in os.listdir(f'fit/data/experiment{experiment_id}_fit') if fname[-1] == "p"]

    else:
        data = [load(f'fit/data/experiment{experiment_id}_{condition}_fit/{fname}')
            for fname in os.listdir(f'fit/data/experiment{experiment_id}_{condition}_fit') if fname[-1] == "p"]

    models = params['cognitive_params'].keys()

    new_data = {}

    for model in models:

        data_model = params['cognitive_params'][model].copy()
        new_data_model = {}

        for k, v in data_model.items():
            if isinstance(v, np.ndarray):
                new_data_model[f'{k}0'] = []
                new_data_model[f'{k}1'] = []
                continue
            else:
                new_data_model[k] = []

        new_data_model['log'] = []

        for d in data:
            for k in new_data_model.keys():
                new_data_model[k].append(d[model][k])

        new_data[model] = new_data_model

    return new_data


def compute_mean_for_each_model(experiment_id):
    data = regroup_by_model(experiment_id=experiment_id)
    for model in data.keys():
        new_data_model = {}
        for k in data[model].keys():
            new_data_model[k] = data[model][k]
        mean = {k: np.mean(v) for k, v in new_data_model.items()}
        data[model] = mean
    return data


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    exit('Please run the main.py script.')
