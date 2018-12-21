import numpy as np

# common parameters of each condition
params = {

    'n_options': 2,
    'n_conds': 4,

    't_when_reversal_occurs': [],

    'cognitive_params': {

        'QLearning': {
            'alpha': 0,
            'beta': 0,

        },

        'AsymmetricQLearning': {
            # alpha for [losses, gains]
            'alpha': np.array([0, 0]),
            'beta': 0,
        },

        'PerseverationQLearning': {
            'alpha': 0.7,
            'beta': 100,
            'phi': 0.8,
        },

        'PriorQLearning': {
            'alpha': 0,
            'beta': 0,
            'q': 0
        },

        'AsymmetricPriorQLearning': {
            'alpha': np.array([0, 0]),
            'beta': 0,
            'q': 0
        },

        'FullQLearning': {
            'alpha': np.array([0, 0]),
            'beta': 0,
            'phi': 0,
            'q': 0
        }
    }
}

# common_params.update(
#     t when probabilities and rewards are reversed between
#     Action A and B
    # {
    #     't_when_reversal_occurs':
    #         np.arange(
    #             0,
    #             common_params['t_max'] + 1,
    #             common_params['t_max'] // (common_params['n_reversals'] + 1),
    #         )[1:-1]
    # }
# )

cond = {
    'status_quo': {
        'condition': 'AB',
        # rewards : A - > [loose, win] , B -> [loose, win]
        'rewards': np.array([[-1, 1], [-1, 1]]),

        # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
        'p': np.array([[0.25, 0.75], [0.75, 0.25]]),

        't_max': 192,

        't_reversal_bme': [60, 120, 180],
        't_reversal_equal': [72, 112, 128, 156, 168, 180]
    },

    'risk': {

        't_max': 192,

        'conds': [

            {'condition': 'AB',
             # rewards : A - > [loose, win] , B -> [loose, win]
             'rewards': np.array([[0, 0], [-1, 1]]),

             # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
             'p': np.array([[0.75, 0.25], [0.25, 0.75]]),
             },
            {'condition': 'CD',
             # rewards : A - > [loose, win] , B -> [loose, win]
             'rewards': np.array([[0, 0], [-1, 1]]),

             # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
             'p': np.array([[0.75, 0.25], [0.75, 0.25]]),
             },
            {'condition': 'EF',
             # rewards : A - > [loose, win] , B -> [loose, win]
             'rewards': np.array([[0, 0], [-1, 1]]),

             # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
             'p': np.array([[0.75, 0.25], [0.5, 0.5]]),
             },
            {'condition': 'GH',
             # rewards : A - > [loose, win] , B -> [loose, win]
             'rewards': np.array([[0, 0], [-1, 1]]),

             # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
             'p': np.array([[0.25, 0.75], [0.75, 0.25]]),
             }
        ]
}}

if __name__ == '__main__':
    exit('Please run the main.py script.')
