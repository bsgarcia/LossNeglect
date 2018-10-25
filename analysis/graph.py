#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import matplotlib as mat

plt.style.use('seaborn-dark')


def reward_model_comparison(data, data_scatter):

    # get x of bars for scatters
    bars = []

    # params
    ind = np.arange(len(data))
    width = 0.2
    opacity = 0.8
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
        rects = ax.bar(ind + i * width, means, width, yerr=std, label=label, alpha=opacity,
               error_kw=dict(ecolor='black', capsize=4, capthick=1.5, alpha=0.8, linewidth=1.5)
        )

        bars.append([rect.get_x() for rect in rects])
    #
    # bars = np.array(bars).flatten()
    #

    idx = iter([0, ] * 3 + [1, ] * 3 + [2, ] * 3)
    for (i, d), j in zip(enumerate(data_scatter), list(range(3)) * 3):
        x = bars[j][next(idx)]
        ax.scatter(np.repeat(x + width/2, len(d)), d, color=f'C{j}', alpha=0.5)

    ax.legend()
    plt.show()


def choice_comparison_line_plot(data,
                                t_when_reversal,
                                ylabel,
                                title,
                                gs,
                                fig,
                                i,
                                axes,
                                legend_elements):

    colors = ['C0', 'C1', 'C2']

    for model, label in enumerate(
            ('QLearning', 'Asymmetric', 'Perseveration')):

        ax = fig.add_subplot(gs[model, 0])

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
                pass
                # ax.set_xticks(sorted(list(ax.get_xticks()) + list(t_when_reversal)))
        # ------------------------------------------------------------ #

        # Compute data
        means = [data[model][t][0] for t in range(len(data[model]))]
        std = [data[model][t][1] for t in range(len(data[model]))]

        ax.plot(means, label=label, color=colors[model])

        # Fill with error
        ax.fill_between(
            range(len(means)),
            [m - err for m, err in zip(means, std)],
            [m + err for m, err in zip(means, std)],
            alpha=0.4,
            color=colors[model]
        )

        ax.set_ylim(0.0, 0.9)

        if '{}' in ylabel:
            ax.set_ylabel(ylabel.format(('A', 'B')[i]))
        else:
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


def choice_comparison(data, t_when_reversal, ylabel, conds):

    n_cond = len(data)
    gs1 = gd.GridSpec(ncols=1, nrows=n_cond)
    fig = plt.figure(figsize=(18, 13))

    cond_labels = [c.replace('_', ' ').capitalize() for c in conds]

    axes = []
    legend_elements = []

    for i, cond in enumerate(data):

        gs2 = gd.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs1[i, 0])

        choice_comparison_line_plot(cond, t_when_reversal, ylabel, cond_labels[i], gs=gs2, fig=fig, i=i, axes=axes, legend_elements=legend_elements)

    n_cond = len(data)
    n_axes = len(axes)

    legend_elements += [
        mat.lines.Line2D([0], [0], color='gray', linestyle="dashed", label="Reversal Event")
    ]

    for i, idx_ax in zip(range(n_cond), list(range(n_axes))[::n_axes//n_cond]):
        axes[idx_ax].set_title(cond_labels[i])
        axes[idx_ax].legend(handles=legend_elements, bbox_to_anchor=(1.09, 1.05), frameon=True)

    plt.show()


if __name__ == '__main__':
    exit('Please run the main.py script.')




