import numpy as np


# common parameters of each condition
params = {

    # N of agents for each model
    'n_agents': 30,


    # time steps for one session
    't_max': 160,
    # 'n_sessions': 2,
    'n_reversals': 0,
    'n_options': 2,

    'cognitive_params': {

        'QLearningAgent': {
            'alpha': 0,
            'beta': 0,

        },

        'AsymmetricQLearningAgent': {
            # alpha for [losses, gains]
            'alpha': np.array([0, 0]),
            'beta': 0,
        },

        'PerseverationQLearningAgent': {
            'alpha': 0.7,
            'beta': 100,
            'phi': 0.8,
        },

        'PriorQLearningAgent': {
            'alpha': 0,
            'beta': 0,
            'q': 0
        },

        'FullQLearningAgent': {
            'alpha': np.array([0, 0]),
            'beta': 0,
            'phi': 0,
            'q': 0
        }
    }
}

# common_params.update(
#     # t when probabilities and rewards are reversed between
#     # Action A and B
#     {
#         't_when_reversal_occurs':
#             np.arange(
#                 0,
#                 common_params['t_max'] + 1,
#                 common_params['t_max'] // (common_params['n_reversals'] + 1),
#             )[1:-1]
#     }
# )

cond = [
    {
        'condition': '25_25',
        # rewards : A - > [loose, win] , B -> [loose, win]
        'rewards': np.array([[-1, 1], [-1, 1]]),

        # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
        'p': np.array([[0.75, 0.25], [0.75, 0.25]]),
    },
    {
        'condition': '75_25',
        # rewards : A - > [loose, win] , B -> [loose, win]
        'rewards': np.array([[-1, 1], [-1, 1]]),

        # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
        'p': np.array([[0.25, 0.75], [0.75, 0.25]]),
    },
    {
        'condition': '25_75',
        # rewards : A - > [loose, win] , B -> [loose, win]
        'rewards': np.array([[-1, 1], [-1, 1]]),

        # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
        'p': np.array([[0.75, 0.25], [0.25, 0.75]]),
    },
    {
        'condition': '75_75',
        # rewards : A - > [loose, win] , B -> [loose, win]
        'rewards': np.array([[-1, 1], [-1, 1]]),

        # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
        'p': np.array([[0.25, 0.75], [0.25, 0.75]]),
    },
]


if __name__ == '__main__':
    exit('Please run the main.py script.')
