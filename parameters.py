import numpy as np


# common parameters of each condition
common_params = {

    # N of agents for each model
    'n_agents': 30,

    # time steps for one session
    't_max': 40,
    'n_sessions': 2,
    'n_reversals': 1,
    'n_options': 2,

    'cognitive_params': {

        'QLearningAgent': {
            'alpha': 0.7,
            'beta': 3,
        },

        'AsymmetricQLearningAgent': {
            # alpha for [losses, gains]
            'alpha': np.array([0.6, 0.8]),
            'beta': 3,
        },

        'PerseverationQLearningAgent': {
            'alpha': 0.7,
            'beta': 3,
            'phi': 2.,
        },
    }
}

common_params.update(
    # t when probabilities and rewards are reversed between
    # Action A and B
    {
        't_when_reversal_occurs':
            np.arange(
                0,
                common_params['t_max'] + 1,
                common_params['t_max'] // (common_params['n_reversals'] + 1),
            )[1:-1]
    }
)

# Define specific properties of each condition
risk_positive = {
    'condition': 'risk_positive',
    # rewards : A - > [loose, win] , B -> [loose, win]
    'rewards': np.array([[0, 0], [-1, 1]]),

    # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
    'p': np.array([[0, 1], [0.25, 0.75]]),
}

risk_neutral = {
    'condition': 'risk_neutral',
    # rewards : A - > [loose, win] , B -> [loose, win]
    'rewards': np.array([[0, 0], [-1, 1]]),

    # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
    'p': np.array([[0, 1], [0.5, 0.5]]),
}

risk_negative = {
    'condition': 'risk_negative',
    # rewards : A - > [loose, win] , B -> [loose, win]
    'rewards': np.array([[0, 0], [-1, 1]]),

    # probabilities : A - > [ p of losing, p of winning] , B -> [p of losing, p of winning]
    'p': np.array([[0, 1], [0.75, 0.25]]),
}

# Fill dictionaries
for p in risk_positive, risk_neutral, risk_negative:
    p.update(common_params)


if __name__ == '__main__':
    exit('Please run the main.py script.')
