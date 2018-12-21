#!/usr/bin/python3.6
import numpy as np
import warnings

warnings.filterwarnings('error')


class QLearning:

    """
    Basic QLearning model
    """

    def __init__(self, alpha, beta, t_max, n_options, n_conds, **kwargs):

        self.alpha = alpha

        if alpha is None:
            self.alpha = np.array([kwargs['alpha0'], kwargs['alpha1']])

        self.beta = beta

        self.q = np.zeros((n_options, n_conds), dtype=float)
        self.pe = np.zeros((t_max, n_options, n_conds), dtype=float)
        self.p_softmax = np.zeros((t_max, n_options, n_conds), dtype=float)

        self.choices = np.zeros(t_max, dtype=int)
        self.rewards = np.zeros(t_max, dtype=float)

    @property
    def memory(self):
        return {
            'choices': self.choices,
            'rewards': self.rewards,
            'prediction_error': self.pe,
            'q_values': self.q,
            'p_softmax': self.p_softmax,
        }

    def save(self, choice, t, reward, cond):

        self.choices[t] = choice
        self.rewards[t] = reward
        self.pe[t, choice, cond] = reward - self.q[choice, cond]

        self.p_softmax[t, :, cond] = self.softmax(cond, t=t)

    def choice(self, cond, t=None):
        return np.random.choice([0, 1], p=self.softmax(cond, t=t))

    def learn(self, t, choice, cond):
        self.q[choice, cond] += self.alpha * self.pe[t, choice, cond]

    def softmax(self, cond, **kwargs):
        m = max(self.q[:, cond] * self.beta)
        return np.exp(
            self.beta * self.q[:, cond] - m
        ) / np.sum(np.exp(
            self.beta * self.q[:, cond] - m
        ))


class AsymmetricQLearning(QLearning):

    """
    Qlearning model using
    asymmetric learning rates:
    we use alpha+ to update qvalues
    when prediction error is superior to 0.
    Otherwise we use alpha-.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, choice, t, cond):
        self.q[choice, cond] += self.alpha[int(self.pe[t, choice, cond] > 0)] * self.pe[t, choice, cond]


class PerseverationQLearning(QLearning):

    """
    Basic Qlearning model
    using a softmax with a 'pi' parameter,
    that is added to the qvalue of the last choice.
    By this, we incentivize the agent to
    reproduce the last choice he made.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phi = kwargs["phi"]

    def softmax(self, cond, t=None):

        # if t == 0 then we use the first implemented
        # softmax because no choice has been made
        if t == 0:
            return super().softmax(cond, t=t)

        # else get last choice
        c = self.choices[t - 1]

        m = max(self.beta * self.q[:, cond])

        # Qvalue for last option chosen (+ phi)
        q1 = self.beta * self.q[c, cond] + self.phi - m
        # Qvalue for the other option
        q2 = self.beta * self.q[int(not c), cond] - m

        # sort depending on value of last choice
        ordered = np.array([q1, q2]) if c == 0 else np.array([q2, q1])

        return np.exp(ordered) / np.sum(np.exp(ordered))


class PriorQLearning(QLearning):

    """
    Basic qlearning with initially biased qvalues

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q[:, :] = kwargs['q']


class AsymmetricPriorQLearning(AsymmetricQLearning):

    """
    Asymmetric with initially biased qvalues

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q[:, :] = kwargs['q']


class FullQLearning(
    PriorQLearning,
    PerseverationQLearning,
    AsymmetricQLearning,
):
    pass


if __name__ == '__main__':
    exit('Please run the main.py script.')
