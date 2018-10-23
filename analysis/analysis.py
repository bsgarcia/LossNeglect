import pickle
import numpy as np
from analysis import graph
import scipy.stats as sp


def load(fname):
    return pickle.load(open(fname, 'rb'))


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

    single_choice_comparison()
    correct_choice_comparison()
    reward_model_comparison()


if __name__ == '__main__':
    exit('Please run the main.py script.')
