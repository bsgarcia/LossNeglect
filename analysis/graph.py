#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import matplotlib as mat

plt.style.use('seaborn-dark')


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


def choice_comparison(data, t_when_reversal, ylabel):

    gs1 = gd.GridSpec(ncols=1, nrows=2)
    plt.figure(figsize=(18, 13))

    cond_labels = 'Risk Positive', 'Risk Negative'#, 'Risk Neutral'
    colors = ['C0', 'C1', 'C2']

    legend_elements = []
    axes = []

    for i, cond in enumerate(data[:-1]):

        gs2 = gd.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs1[i, 0])

        for model, label in enumerate(
                ('QLearning', 'Asymmetric', 'Perseveration')):

            ax = plt.subplot(gs2[model, 0])

            axes.append(ax)

            # remove right and top framing
            ax.spines['right'].set_visible(0)
            ax.spines['top'].set_visible(0)

            # Add reversal lines
            # ------------------------------------------------------------ #
            for t, l in zip(t_when_reversal, range(len(t_when_reversal))):

                ax.vlines(
                    x=t,
                    ymin=0.2,
                    ymax=0.8,
                    linestyle="--",
                    linewidth=2,
                    color='gray',
                )

                if model == 2:
                    ax.set_xticks(t_when_reversal)
            # ------------------------------------------------------------ #

            # Compute data
            means = [cond[model][t][0] for t in range(len(cond[model]))]
            std = [cond[model][t][1] for t in range(len(cond[model]))]

            ax.plot(means, label=label, color=colors[model])

            # Fill with error
            ax.fill_between(
                range(len(means)),
                [m - err for m, err in zip(means, std)],
                [m + err for m, err in zip(means, std)],
                alpha=0.4,
                color=colors[model]
            )

            ax.set_ylim(0.1, 0.9)
            ax.set_ylabel(ylabel)

            # Customize depending on the plot
            # ---------------------------------------- #

            if model == 2:
                ax.set_xlabel('t')
            else:
                ax.spines['bottom'].set_visible(0)
                ax.set_xticks([])

            if i == 0:
                legend_elements.append(
                    mat.lines.Line2D([0], [0], color=colors[model], label=label)
                )

            # ---------------------------------------- #

    legend_elements += [
        mat.lines.Line2D([0], [0], color='gray', linestyle="dashed", label="Reversal Event")
    ]

    for i in (0, 3):
        axes[i].set_title(cond_labels[i - 2])
        axes[i].legend(handles=legend_elements, bbox_to_anchor=(1.09, 1.05), frameon=True)

    plt.show()


if __name__ == '__main__':
    exit('Please run the main.py script.')




