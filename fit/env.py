#!/usr/bin/python3.6
import numpy as np


class Environment:

    def __init__(self, **kwargs):

        self.params = kwargs.copy()
        self.data = kwargs['data']

        self.n_sessions = kwargs.get('n_sessions')
        self.t_max = kwargs.get('t_max')

        self.model = kwargs.get('model')

        self.cognitive_params = kwargs['cognitive_params']

        self.condition = kwargs.get('condition')
        self.n_options = kwargs.get('n_options')

        self.t_when_reversal_occurs = kwargs.get('t_when_reversal_occurs')
        self.p = None
        self.rewards = None
        self.dic_conds = kwargs.get('dic_conds')
        self.conds = kwargs.get('conds')

    def run(self):

        agent = self.model(
            alpha=self.cognitive_params.get('alpha'),
            beta=self.cognitive_params.get('beta'),
            phi=self.cognitive_params.get('phi'),
            q=self.cognitive_params.get('q'),
            t_max=self.t_max,
            n_options=self.n_options,
            n_conds=4
        )
        # mapping column 4 to choice variable
        c_col = 4
        r_col = 7

        neg_log_likelihood = 0

        for t in range(self.t_max):

            self.set_condition(t)

            # make agent play by using subject'choice and reward
            choice = 1 if self.data[t, c_col] else -1 if not self.data[t, c_col] else 0
            win = int(self.data[t, r_col] > 0)
            cond = self.conds[t]

            if choice != -1:

                reward = self.play(choice, win)
                agent.save(choice=choice, t=t, reward=reward, cond=cond)

                if t != self.t_max - 1:
                    agent.learn(choice=choice, t=t, cond=cond)

                neg_log_likelihood += np.log(
                    agent.memory['p_softmax'][t, choice, cond] + 1e-10
                )

        return -neg_log_likelihood

    def set_condition(self, t):

        # switch condition
        self.p = self.dic_conds[t]['p'].copy()
        self.rewards = self.dic_conds[t]['rewards'].copy()

    def play(self, choice, win):
        return self.rewards[choice][win]
        #[[np.random.choice(
            # [0, 1],
            # p=self.p[choice]
        # )]

    def plot(self, results):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.data[:, 4])
        fig, ax = plt.subplots()
        ax.plot(results)
        plt.show()


if __name__ == '__main__':
    exit('Please run the main.py script.')
    