import pickle
import numpy as np
from analysis import graph
import scipy.stats as sp
from fit.parameters import params
import os


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def params_model_comparisons():

    data = [load(f'fit/data/experiment_1_fit/{fname}')
            for fname in os.listdir('fit/data/experiment_1_fit') if fname[-1] == "p"]
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

            new_data_model[k] = []

        for d in data:
            for k in new_data_model.keys():
                if k == "beta":
                    new_data_model[k].append([1/d[model][k]])
                else:
                    new_data_model[k].append(d[model][k])

        mean = {k: np.mean(v) for k, v in new_data_model.items()}
        std = {k: sp.sem(v) for k, v in new_data_model.items()}
        new_data[model] = {'scatter': new_data_model, 'mean_std': (mean, std)}

    graph.bar_plot_model_comparison(data=new_data, data_scatter=None, ylabel='value')


def reward_model_comparison():

    # size = (3 conditions, 3 models, 2 values (mean and std))
    data = []

    for cond in ('risk_positive', 'risk_negative', 'risk_neutral'):

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        n_agents = d['params']['n_agents']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'

        new_data = np.zeros(n_agents)

        for model in models:

            for a in range(n_agents):

                new_data[a] = np.sum(d['results'][model]['rewards'][a, :] == 1) / t_max

            # add mean and std for a model in one condition
            one_cond_data.append([np.mean(new_data), np.std(new_data)])

        data.append(one_cond_data)

    graph.reward_model_comparison(data)


def correct_choice_comparison():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []

    for cond in ('risk_positive', 'risk_negative', 'risk_neutral'):

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'

        t_when_reversal = d['params']['t_when_reversal_occurs']

        for i, model in enumerate(models):

            one_model_data = []

            for t in range(t_max):

                mean = np.mean(d['results'][model]['correct_choices'][:, t])
                std = sp.sem(d['results'][model]['correct_choices'][:, t])

                # add mean and std for a model in one condition
                one_model_data.append([mean, std])

            one_cond_data.append(one_model_data)

        data.append(one_cond_data)

    graph.correct_choice_comparison(data, t_when_reversal=t_when_reversal, ylabel='Correct Choice')


def single_choice_comparison():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []

    for cond in ('risk_positive', 'risk_negative', 'risk_neutral'):

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'

        t_when_reversal = d['params']['t_when_reversal_occurs']

        for i, model in enumerate(models):

            one_model_data = []

            for t in range(t_max):

                mean = np.mean(d['results'][model]['choices'][:, t] == 1)
                std = sp.sem(d['results'][model]['choices'][:, t] == 1)

                # add mean and std for a model in one condition
                one_model_data.append([mean, std])

            one_cond_data.append(one_model_data)

        data.append(one_cond_data)

    graph.correct_choice_comparison(data, t_when_reversal=t_when_reversal, ylabel='Chose option B')


def run():

    # single_choice_comparison()
    # correct_choice_comparison()
    # reward_model_comparison()
    params_model_comparisons()


if __name__ == '__main__':
    exit('Please run the main.py script.')
