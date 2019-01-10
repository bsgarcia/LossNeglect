#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd


def model_recovery(data, models, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            color = "grey" if data[j, i] >= np.max(data) - 5 else 'white'
            text = ax.text(i, j, int(data[j, i]),
                           ha="center", va="center", color=color)

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def bar_plot_model_comparison(data, data_scatter, ylabel, title=None):

    # plt.style.use('default')
    # params
    ind = np.arange(len(data))
    width = 0.18
    opacity = 0.8

    plt.figure(figsize=(28, 12))
    ax = plt.subplot()

    # remove some axis
    ax.spines['top'].set_visible(0)
    ax.spines['right'].set_visible(0)
    ax.spines['left'].set_visible(0)
    ax.spines['bottom'].set_visible(0)

    # grid in bg
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', alpha=0.2, zorder=2)

    ax.set_ylabel(ylabel)
    # ax.set_ylim(-1, 1.1)
    ax.tick_params(size=0)

    ax.set_title(title)

    xlabels = []
    xticks = []

    for i, (label, model) in enumerate(data.items()):

        x = list(model['mean_std'][0].keys())
        xlabels += x

        means = [model['mean_std'][0][k] for k in x]
        stds = [model['mean_std'][1][k] for k in x]

        inds = [ind[i] + j * width for j in range(len(means))]

        xticks += inds

        rects = ax.bar(
            inds,
            means,
            width,
            yerr=stds,
            label=label,
            zorder=0,
            edgecolor='white',
            error_kw=dict(ecolor='black', capsize=4, capthick=1.5, alpha=1, linewidth=1.5, zorder=1)
        )

        for rect, m in zip(rects, means):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, m + m * 5/100, str(m)[:6], ha='center', va='bottom')

        # bars.append([rect.get_x() for rect in rects])
    # Add scatter points
    # in order to have an idea of the distribution
    # idx = iter([0, ] * 3 + [1, ] * 3 + [2, ] * 3)
    # for (i, d), j in zip(enumerate(data_scatter), list(range(3)) * 3):
    #
    #     if True in [j > 0 for j in d]:
    #
    #         x = bars[j][next(idx)]
    #         x_randomized = [y + (width * np.random.randint(1, 10)/10) for y in np.repeat(x, len(d))]
    #         ax.scatter(x_randomized, d, color=f'C{j}', alpha=0.5, zorder=1)

    ax.set_xticks(xticks)

    for i in range(len(xlabels)):
        if xlabels[i] == 'log':
            xlabels[i] = 'norm LL'
        elif xlabels[i] == 'q':
            pass
        elif xlabels[i][:-1] in 'alpha':
            xlabels[i] = f"$\\{xlabels[i]}$".replace("0", "-").replace("1", "+")
        elif xlabels[i] == 'phi':
            xlabels[i] = f"norm $\\{xlabels[i]}$"
        elif xlabels[i] == 'beta':
            xlabels[i] = "$1/\\beta$"

    ax.set_xticklabels(xlabels)
    ax.legend()
    xmin, xmax = ax.get_xlim()
    ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.8)
    plt.show()


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




