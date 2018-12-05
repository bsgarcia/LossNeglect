#!/usr/bin/python3.6
import numpy as np
import tensorflow as tf
import pickle


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

        print(self.cognitive_params)

        choices = np.zeros(self.t_max)

        for t in range(self.t_max):

            # switch condition
            self.p = self.dic_conds[t]['p'].copy()
            self.rewards = self.dic_conds[t]['rewards'].copy()

            # make agent play by using subject'choice
            cond = self.conds[t]
            choice = agent.choice(t=t, cond=cond)
            reward = self.play(choice)
            agent.save(choice=choice, t=t, reward=reward, cond=cond)

            if t != self.t_max - 1:
                agent.learn(choice=choice, t=t, cond=cond)

            choices[t] = choice

        # self.plot(results=choices)

        choices[choices == 0] = -1
        choices = choices[self.data[:, 4] != 0]
        self.data[:, 4] = self.data[self.data[:, 4] != 0, 4]

    def run_fit(self):

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
        col = 4

        values = tf.Variable(tf.zeros(self.t_max, dtype=tf.float32))

        for t in range(self.t_max):

            # switch condition
            self.p = self.dic_conds[t]['p'].copy()
            self.rewards = self.dic_conds[t]['rewards'].copy()

            # make agent play by using subject'choice
            choice = 1 if self.data[t, col] else -1 if not self.data[t, col] else 0
            cond = self.conds[t]

            if choice != -1:

                reward = self.play(choice)
                choice = tf.Variable(choice)
                agent.save(choice=choice, t=t, reward=reward, cond=cond)

                if t != self.t_max - 1:
                    agent.learn(choice=choice, t=t, cond=cond)

                values = values[t].assign(agent.memory['p_softmax'][t, choice, cond])

            else:
                values = values[t].assign(choice)

        # print(-sum(values[values != -1]))
        return -tf.reduce_sum(tf.log(values[values.eval() != -1]))

    def play(self, choice):

        return self.rewards[choice][np.random.choice(
            [0, 1],
            p=self.p[choice]
        )]

    def plot(self, results):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.data[:, 4])
        fig, ax = plt.subplots()
        ax.plot(results)
        plt.show()


if __name__ == '__main__':
    exit('Please run the main.py script.')
    