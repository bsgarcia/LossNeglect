#!/usr/bin/python3.6
import numpy as np


class QLearningAgent:

    """
    Basic QLearning model
    """

    def __init__(self, alpha, beta, t_max, n_options, **kwargs):

        self.alpha = alpha

        self.beta = beta

        self.q = np.zeros((t_max, n_options), dtype=float)

        self.pe = np.zeros((t_max, n_options), dtype=float)
        self.choices = np.zeros(t_max, dtype=int)
        self.rewards = np.zeros(t_max, dtype=int)

        self.p_softmax = np.zeros((t_max, n_options), dtype=float)

    def save(self, choice, t, reward):

        self.choices[t] = choice
        self.rewards[t] = reward
        self.pe[t, choice] = reward - self.q[t, choice]

        self.p_softmax[t, :] = self.softmax(t)

    def choice(self, t):
        return np.random.choice([0, 1], p=self.softmax(t))

    def learn(self, choice, t):
        self.q[t + 1, choice] = self.q[t, choice] + self.alpha * self.pe[t, choice]

    def softmax(self, t):
        return np.exp(
            self.beta * self.q[t, :]
        ) / np.sum(np.exp(
            self.beta * self.q[t, :]
        ))

    @property
    def memory(self):
        return {
            'choices': self.choices,
            'rewards': self.rewards,
            'prediction_error': self.pe,
            'q_values': self.q,
            'p_softmax': self.p_softmax,
        }


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

    def learn(self, choice, t):
        self.q[t + 1, choice] = \
            self.q[t, choice] + self.alpha[int(self.pe[t, choice] > 0)] * self.pe[t, choice]


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

    def softmax(self, t):

        # if t == 0 then we use the first implemented
        # softmax because no choice has been made
        if t == 0:
            return super().softmax(t)

        # else get last choice
        c = self.choices[t - 1]

        # Qvalue for last option chosen (+ phi)
        q1 = self.beta * self.q[t, c] + self.phi
        # Qvalue for the other option
        q2 = self.beta * self.q[t, int(not c)]

        # sort depending on value of last choice
        ordered = np.array([q1, q2]) if c == 0 else np.array([q2, q1])

        return np.exp(ordered) / np.sum(np.exp(ordered))


class PriorQLearningAgent(QLearningAgent):

    """
    Basic qlearning with initially biased qvalues

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q[0, :] = kwargs['q']


if __name__ == '__main__':
    exit('Please run the main.py script.')
