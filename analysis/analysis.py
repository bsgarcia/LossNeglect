import pickle
import numpy as np
from analysis import graph
import scipy.stats
from fit.parameters import params
import matlab.engine

eng = matlab.engine.start_matlab()


def compute_aic_bic(ll, n_obs, n_params):
    global eng
    aic, bic = eng.aicbic(ll, n_params, n_obs, nargout=2)
    return aic[0], bic[0]

# def compute_best_model(data):


def model_selection_recovery(data, title):
    global eng
    import matlab

    data = regroup_by_model(data, recovery=True)
    n_obs = 4 * 48
    n_subjects = len(data['QLearning']['AsymmetricQLearning']['log'])
    assert n_subjects == 86
    n_models = len(data)

    new_data_bic = np.zeros((n_models, n_models), dtype=np.ndarray)
    new_data_aic = np.zeros((n_models, n_models), dtype=np.ndarray)
    models = list(enumerate(data.keys()))

    for i, fitted_model in models:
        for j, data_model in models:

            ll = matlab.double(
                [-i for i in data[fitted_model][data_model].pop('log')]
            )
            n_obs_vector = matlab.double(
                [n_obs, ] * n_subjects
            )
            n_params = matlab.double(
                [int(len(data[fitted_model][data_model].keys())), ] * n_subjects
            )

            aic, bic = compute_aic_bic(
                ll=ll,
                n_obs=n_obs_vector,
                n_params=n_params
            )
            new_data_bic[j, i] = bic
            new_data_aic[j, i] = aic

    bic = np.zeros((n_models, n_models), dtype=float)
    aic = np.zeros((n_models, n_models), dtype=float)

    for i in range(len(models)):
        max_bic = new_data_bic[i, :]
        max_aic = new_data_aic[i, :]

        for subject in range(n_subjects):
            tempx = []
            tempy = []
            for x, y in zip(max_bic, max_aic):
                tempx.append(x[subject])
                tempy.append(y[subject])
            bic[i, np.argmin(tempx)] += 1
            aic[i, np.argmin(tempy)] += 1

    bic /= n_subjects/100
    aic /= n_subjects/100

    graph.model_recovery(
        data=np.round(bic),
        models=[i[1] for i in models],
        title=title,
        ylabel='Percentage "selected as best model" (BIC)'
    )

    graph.model_recovery(
        data=np.round(aic),
        models=[i[1] for i in models],
        title=title,
        ylabel='Percentage "selected as best model" (AIC)'
    )


def single_model_selection(data, title):
    global eng
    import matlab

    data = regroup_by_model(data)
    n_obs = 96
    n_subjects = len(data['QLearning']['log'])
    models = data.keys()

    new_data_bic = {}
    new_data_aic = {}
    new_bic = []
    new_aic = []

    for model in models:

        ll = matlab.double([-i for i in data[model].pop('log')])
        n_obs_vector = matlab.double([n_obs, ] * n_subjects)
        n_params = matlab.double([int(len(data[model].keys())), ] * n_subjects)

        aic, bic = compute_aic_bic(
            ll=ll,
            n_obs=n_obs_vector,
            n_params=n_params
        )

        new_bic.append(bic)
        new_aic.append(aic)

    bic = np.zeros(len(models))
    aic = np.zeros(len(models))
    # sem_bic = np.zeros(len(models))
    # sem_aic = np.zeros(len(models))

    for subject in range(n_subjects):
        tempx = []
        tempy = []
        for x, y in zip(new_bic, new_aic):
            tempx.append(x[subject])
            tempy.append(y[subject])

        assert len(tempx) == len(models)
        bic[np.argmin(tempx)] += 1
        aic[np.argmin(tempy)] += 1

    # bic /= n_subjects / 100
    # aic /= n_subjects / 100
    #
    for i, model in enumerate(models):
        new_data_bic[model] = {'mean_std': ({model: bic[i]}, {model: 0})}
        new_data_aic[model] = {'mean_std': ({model: aic[i]}, {model: 0})}

    graph.bar_plot_model_comparison(
        data=new_data_bic, data_scatter=None, title=title,
        ylabel='Percentage "selected as best model" (BIC)'
    )
    graph.bar_plot_model_comparison(
        data=new_data_aic, data_scatter=None,
        ylabel='Percentage "selected as best model" (AIC)',
        title=title
    )


def params_model_comparisons(data):

    data = regroup_by_model(data)

    for model in data.keys():

        new_data_model = {}

        for k in data[model].keys():
            if k == 'beta':
                new_data_model[k] = 1/np.asarray(data[model][k])
            elif k in ('log', 'phi'):
                new_data_model[k] = np.asarray(data[model][k]) * 1/max(data[model][k])
            else:
                new_data_model[k] = data[model][k]
        new_data_model.pop('log')
        mean = {k: np.mean(v) for k, v in new_data_model.items()}
        std = {k: scipy.stats.sem(v) for k, v in new_data_model.items()}
        data[model] = {'scatter': None, 'mean_std': (mean, std)}

    graph.bar_plot_model_comparison(data=data, data_scatter=None, ylabel='value')

# ------------------------------------------------------- Utils---------------------------------------------- # 


def regroup_by_model(data, recovery=False):

    models = params['cognitive_params'].keys()
    new_data = {}

    if recovery:

        for fitted_model in data[0].keys():

            new_data[fitted_model] = {}

            for data_model in data[0]['QLearning'].keys():

                params_model = params['cognitive_params'][fitted_model].copy()
                new_data_model = {}
                for k, v in params_model.items():
                    if isinstance(v, np.ndarray):
                        new_data_model[f'{k}0'] = []
                        new_data_model[f'{k}1'] = []
                        continue
                    else:
                        new_data_model[k] = []

                new_data_model['log'] = []

                for subject in data:
                    for k in new_data_model.keys():
                        new_data_model[k].append(subject[fitted_model][data_model][k])

                new_data[fitted_model][data_model] = new_data_model.copy()

    else:

        for model in models:

            params_model = params['cognitive_params'][model].copy()
            new_data_model = {}

            for k, v in params_model.items():
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

            new_data[model] = new_data_model.copy()

    return new_data
    # pass


def compute_mean_for_each_model(data):
    data = regroup_by_model(data)
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
