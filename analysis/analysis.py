import pickle
import numpy as np
from analysis import graph
import scipy.stats as sp


def load(fname):
    return pickle.load(open(fname, 'rb'))


# --------------------------------------------- BAR PLOTS --------------------------------------------------------- #

def correct_choice_bar_plot_model_comparison():

    # size = (3 conditions, 3 models, 2 values (mean and std))
    data = []
    scatter = []

    for cond in ('risk_positive', 'risk_negative', 'risk_neutral'):

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        n_agents = d['params']['n_agents']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'

        new_data = np.zeros(n_agents)

        for model in models:

            for a in range(n_agents):

                new_data[a] = np.sum(d['results'][model]['correct_choices'][a, :] == 1) / t_max

            # add mean and std for a model in one condition
            one_cond_data.append([np.mean(new_data), np.std(new_data)])
            scatter.append(list(new_data))

        data.append(one_cond_data)

    graph.bar_plot_model_comparison(
        data, scatter,
        ylabel='Correct choice'
    )


def reward_bar_plot_model_comparison():

    # size = (3 conditions, 3 models, 2 values (mean and std))
    data = []
    scatter = []

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
            scatter.append(list(new_data))

        data.append(one_cond_data)

    graph.bar_plot_model_comparison(
        data,
        scatter,
        ylabel='N Rewards/ N trials'
    )


# --------------------------------------------- CURVE PLOTS --------------------------------------------------------- #


def correct_choice_curve_plot_comparison():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []
    conds = 'risk_positive', 'risk_negative'

    for cond in conds:

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

    graph.choice_comparison(
        data,
        t_when_reversal=t_when_reversal,
        ylabel='Correct Choice',
        conds=conds
    )


def single_choice_curve_plot_comparison():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []

    conds = 'risk_positive', 'risk_negative', 'risk_neutral'

    for cond in conds:

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

    graph.choice_comparison(
        data=data,
        t_when_reversal=t_when_reversal,
        ylabel='Choice = B',
        conds=conds
    )


def risky_choice():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []

    conds = 'risk_neutral',

    for cond in conds:

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'

        t_when_reversal = d['params']['t_when_reversal_occurs']

        for i, model in enumerate(models):

            one_model_data = []

            for t in range(t_max):

                mean = np.mean(d['results'][model]['risky_choice'][:, t] == 1)
                std = sp.sem(d['results'][model]['risky_choice'][:, t] == 1)

                # add mean and std for a model in one condition
                one_model_data.append([mean, std])

            one_cond_data.append(one_model_data)

        data.append(one_cond_data)

    graph.choice_comparison(
        data=data,
        t_when_reversal=t_when_reversal,
        ylabel='Risky choice',
        conds=conds
    )


def objective_p_of_winning_a_and_b():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []
    conds = ('risk_positive', ) * 2

    for c in (0, 1):

        one_cond_data = []

        d = load(f'data/data_risk_positive.p')
        t_max = d['params']['t_max']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'
        n_agents = d['params']['n_agents']

        t_when_reversal = d['params']['t_when_reversal_occurs']

        for i, model in enumerate(models):

            one_model_data = []

            p_win = np.zeros((n_agents, t_max))

            for a in range(n_agents):

                cursor = 0

                for t in range(t_max):

                    if t >= 1:

                        if t in t_when_reversal:
                            cursor = t

                        choice = d['results'][model]['choices'][a, cursor:t] == c
                        rewards = d['results'][model]['rewards'][a, cursor:t] == 1

                        if np.sum(choice) == 0:
                            p_reward = 0
                        else:
                            p_reward = np.sum(rewards * choice) / np.sum(choice)

                        p_win[a, t] = p_reward

            for t in range(t_max):

                mean = np.mean(p_win[:, t])
                std = sp.sem(p_win[:, t])

                # add mean and std for a model in one condition
                one_model_data.append([mean, std])

            one_cond_data.append(one_model_data)

        data.append(one_cond_data)

    graph.choice_comparison(
        data=data,
        t_when_reversal=t_when_reversal,
        ylabel='$p(R^t_{} = 1)$',
        conds=conds
    )


def softmax_p_of_choosing_a_and_b():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    conds = 'risk_positive', 'risk_negative', 'risk_neutral'

    for cond in conds:

        data = []

        for c in (0, 1):

            one_cond_data = []

            d = load(f'data/data_{cond}.p')
            t_max = d['params']['t_max']
            models = 'QLearningAgent',

            t_when_reversal = d['params']['t_when_reversal_occurs']

            for i, model in enumerate(models):

                one_model_data = []

                for t in range(t_max):

                    mean = np.mean(d['results'][model]['p_softmax'][:, t, c])
                    std = sp.sem(d['results'][model]['p_softmax'][:, t, c])

                    # add mean and std for a model in one condition
                    one_model_data.append([mean, std])

                one_cond_data.append(one_model_data)

            data.append(one_cond_data)

        graph.choice_comparison(
            data=data,
            t_when_reversal=t_when_reversal,
            ylabel='$p({}^t)$',
            conds=[cond, ] * 2
        )


def p_of_winning_taking_the_best_option():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []
    conds = 'risk_positive',

    for cond in conds:

        one_cond_data = []

        d = load(f'data/data_{cond}.p')
        t_max = d['params']['t_max']
        models = 'QLearningAgent', 'AsymmetricQLearningAgent', 'PerseverationQLearningAgent'
        n_agents = d['params']['n_agents']

        t_when_reversal = d['params']['t_when_reversal_occurs']

        for i, model in enumerate(models):

            one_model_data = []

            p_win = np.zeros((n_agents, t_max))

            for a in range(n_agents):

                for t in range(t_max):

                    if t >= 1:

                        correct_choice = d['results'][model]['correct_choices'][a, :t] == 1
                        rewards = d['results'][model]['rewards'][a, :t] == 1

                        if np.sum(correct_choice) == 0:
                            p_reward_for_correct_choice = 0
                        else:
                            p_reward_for_correct_choice = np.sum(rewards * correct_choice) / np.sum(correct_choice)

                        p_win[a, t] = p_reward_for_correct_choice

            for t in range(t_max):

                mean = np.mean(p_win[:, t])
                std = sp.sem(p_win[:, t])

                # add mean and std for a model in one condition
                one_model_data.append([mean, std])

            one_cond_data.append(one_model_data)

        data.append(one_cond_data)

    graph.choice_comparison(
        data=data,
        t_when_reversal=t_when_reversal,
        ylabel='$p(R = 1)$',
        conds=conds
    )


def choice_each_agent():

    # size = (3 conditions, 3 models, t_max * 2 values (mean and std))
    data = []

    cond = 'risk_positive'

    d = load(f'data/data_{cond}.p')
    n_agents = d['params']['n_agents']
    models = 'QLearningAgent'

    t_when_reversal = d['params']['t_when_reversal_occurs']

    for a in range(n_agents):
        data.append(d['results'][models]['choices'][a, :])

    graph.choice_each_agent(
        data=data,
        t_when_reversal=t_when_reversal,
        ylabel='Choice',
        n_rows=n_agents,
        cond=cond
    )


def run():

    # objective_p_of_winning_a_and_b()
    # p_of_winning_taking_the_best_option()
    # single_choice_curve_plot_comparison()
    # correct_choice_curve_plot_comparison()
    correct_choice_bar_plot_model_comparison()
    # softmax_p_of_choosing_a_and_b()
    # reward_bar_plot_model_comparison()
    # risky_choice()


if __name__ == '__main__':
    exit('Please run the main.py script.')
