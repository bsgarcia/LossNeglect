#!/usr/bin/python3.6
import numpy as np
import warnings

warnings.filterwarnings('error')


class QLearningAgent:

    """
    Basic QLearning model
    """

    def __init__(self, alpha, beta, t_max, n_options, n_conds, **kwargs):

        self.alpha = alpha
        self.beta = beta

        self.q = np.zeros((t_max, n_options, n_conds), dtype=float)
        self.pe = np.zeros((t_max, n_options, n_conds), dtype=float)
        self.p_softmax = np.zeros((t_max, n_options, n_conds), dtype=float)

        self.choices = np.zeros(t_max, dtype=int)
        self.rewards = np.zeros(t_max, dtype=int)

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
        self.pe[t, choice, cond] = reward - self.q[t, choice, cond]

        self.p_softmax[t, :, cond] = self.softmax(t, cond)

    def choice(self, t, cond):
        return np.random.choice([0, 1], p=self.softmax(t, cond))

    def learn(self, choice, t, cond):
        self.q[t + 1, choice, cond] = self.q[t, choice, cond] + self.alpha * self.pe[t, choice, cond]

    def softmax(self, t, cond):
        # m = max(self.q[t, :, cond] * self.beta)
        return np.exp(
            self.beta * self.q[t, :, cond]
        ) / np.sum(np.exp(
            self.beta * self.q[t, :, cond]
        ))


class AsymmetricQLearningAgent(QLearningAgent):

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
        self.q[t + 1, choice, cond] = \
            self.q[t, choice, cond] + self.alpha[int(self.pe[t, choice, cond] > 0)] * self.pe[t, choice, cond]


class PerseverationQLearningAgent(QLearningAgent):

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

    def softmax(self, t, cond):

        # if t == 0 then we use the first implemented
        # softmax because no choice has been made
        if t == 0:
            return super().softmax(t, cond)

        # else get last choice
        c = self.choices[t - 1]

        m = max(self.beta * self.q[t, :, cond])

        # Qvalue for last option chosen (+ phi)
        q1 = self.beta * self.q[t, c, cond] + self.phi - m
        # Qvalue for the other option
        q2 = self.beta * self.q[t, int(not c), cond] - m

        # sort depending on value of last choice
        ordered = np.array([q1, q2]) if c == 0 else np.array([q2, q1])

        return np.exp(ordered) / np.sum(np.exp(ordered))


class PriorQLearningAgent(QLearningAgent):

    """
    Basic qlearning with initially biased qvalues

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q[0, :, :] = kwargs['q']


class FullQLearningAgent(
    PriorQLearningAgent,
    PerseverationQLearningAgent,
    AsymmetricQLearningAgent,
):
    pass


if __name__ == '__main__':
    exit('Please run the main.py script.')
