import numpy as np
from simulation.models import QLearning, AsymmetricQLearning, AsymmetricPriorQLearning,\
    FullQLearning, PerseverationQLearning, PriorQLearning
import fit.data


class Globals:
    """
    class used in order to
    declare global variables
    accessible from the whole script.
    """

    # Continuous parameters bounds
    # --------------------------------------------------------------------- #
    alpha_bounds = (1/1000, 1)
    beta_bounds = (1, 1000)
    phi_bounds = (-10, 10)
    q_bounds = (-1, 1)
    q_fixed_bounds = (-1, -1)

    # parameters initial guesses
    # --------------------------------------------------------------------- #
    alpha_guess = 0.5
    beta_guess = 1
    phi_guess = 0
    q_guess = 0
    q_fixed_guess = -1

    qlearning_params = {
        'model': QLearning,
        'labels': ['alpha', 'beta', 'log'],
        'guesses': np.array([alpha_guess, beta_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds])
    }

    asymmetric_params = {
        'model': AsymmetricQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds])
    }

    perseveration_params = {
        'model': PerseverationQLearning,
        'labels': ['alpha', 'beta', 'phi', 'log'],
        'guesses': np.array([alpha_guess, beta_guess, phi_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds, phi_bounds])
    }

    prior_params = {
        'model': PriorQLearning,
        'labels': ['alpha', 'beta', 'q', 'log'],
        'guesses': np.array([alpha_guess, beta_guess, q_guess]),
        'bounds': np.array([alpha_bounds, beta_bounds, q_bounds])
    }

    asymmetric_prior_params = {
        'model': AsymmetricPriorQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'q', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess, q_fixed_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds, q_fixed_bounds]),
    }

    full_params = {
        'model': FullQLearning,
        'labels': ['alpha0', 'alpha1', 'beta', 'phi', 'q', 'log'],
        'guesses': np.array([alpha_guess, alpha_guess, beta_guess, phi_guess, q_guess]),
        'bounds': np.array([alpha_bounds, alpha_bounds, beta_bounds, phi_bounds, q_bounds])
    }

    model_params = [
        qlearning_params,
        asymmetric_params,
        perseveration_params,
        prior_params,
        asymmetric_prior_params,
        full_params
    ]

    reg_fit = False
    refit = True

    assertion_error = 'Only one parameter among {} and {} can be true.'
    assert sum([reg_fit, refit]) == 1, assertion_error.format('reg_fit', 'refit')

    experiment_id = 'full'
    condition = '_status_quo_2'

    data = fit.data.sim(condition=condition)

    #
    n_subjects = len(data)
    subject_ids = range(len(data))

    max_evals = 10000

    options = {
        # 'AlwaysHonorConstraints': 'bounds',
        'Algorithm': 'interior-point',
        # 'display': 'iter-detailed',
        'MaxIter': max_evals,
        'MaxFunEvals': max_evals,
        'display': 'off',
        # 'Diagnostics': 'on'
    }


# common parameters of each condition
params = {

    'n_options': 2,

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
