#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd


def reward_model_comparison(data):

    # params
    ind = np.arange(len(data))
    width = 0.2
    opacity = 0.7
    xlabels = 'Risk Positive', 'Risk Negative', 'Risk Neutral'
    ylabel = 'Rewards'

    plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    # remove some axis
    ax.spines['top'].set_visible(0)
    ax.spines['right'].set_visible(0)
    # ax.spines['left'].set_visible(0)

    # grid in bg
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray', alpha=0.3, linestyle='dashed')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    # ax.set_title('Rewards over 100 trials for each learning model')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(xlabels, rotation=45)

    for i, label in enumerate(('QLearning', 'Asymmetric', 'Perseveration')):

        # cond is condition
        # i is model
        # in the values for each condition and model
        # mean is first and std is second
        means = [data[cond][i][0] for cond in ind]
        std = [data[cond][i][1] for cond in ind]
        ax.bar(ind + i * width, means, width, yerr=std, label=label, alpha=opacity,
               error_kw=dict(ecolor='gray', capsize=4, capthick=1.5))
        # ax.grid(0)

    ax.legend()
    plt.show()


def correct_choice_comparison(data, t_when_reversal, ylabel):

    gs = gd.GridSpec(ncols=1, nrows=2)
    plt.figure(figsize=(15, 10))

    cond_labels = 'Risk Positive', 'Risk Negative'#, 'Risk Neutral'

    for i, cond in enumerate(data[:-1]):

        ax = plt.subplot(gs[i, 0])

        ax.spines['right'].set_visible(0)
        ax.spines['top'].set_visible(0)

        ax.set_title(cond_labels[i])

        for t, label in zip(t_when_reversal, range(len(t_when_reversal))):

            ax.vlines(
                x=t,
                ymin=0.1,
                ymax=0.9,
                linestyle="--",
                color='gray',
                label='Reversal Event' if label == 0 else None
            )

        for model, label in enumerate(
                ('QLearning', 'Asymmetric', 'Perseveration')):

            means = [cond[model][t][0] for t in range(len(cond[model]))]
            std = [cond[model][t][1] for t in range(len(cond[model]))]

            ax.plot(means, label=label)

            ax.fill_between(
                range(len(means)),
                [m - err for m, err in zip(means, std)],
                [m + err for m, err in zip(means, std)],
                alpha=0.4
            )

            ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel)

            if not i:
                ax.legend(loc=4)

            if i != len(data[:-1]) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('t')

    plt.show()


if __name__ == '__main__':
    exit('Please run the main.py script.')




