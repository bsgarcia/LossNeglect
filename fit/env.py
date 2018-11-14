#!/usr/bin/python3.6
import numpy as np
import pickle


class Environment:

    def __init__(self, pbar=None, **kwargs):

        self.params = kwargs.copy()
        self.data = kwargs['data']
        self.pbar = pbar

        self.n_sessions = kwargs.get('n_sessions')
        self.t_max = kwargs.get('t_max')

        self.model = kwargs.get('model')

        self.cognitive_params = kwargs['cognitive_params']

        self.condition = kwargs.get('condition')
        self.n_options = kwargs.get('n_options')
        self.rewards = kwargs.get('rewards')
        self.t_when_reversal_occurs = kwargs.get('t_when_reversal_occurs')
        self.p = kwargs.get('p')

    def run(self):

        name = self.model.__name__

        agent = self.model(
            alpha=self.cognitive_params[name].get('alpha'),
            beta=self.cognitive_params[name].get('beta'),
            phi=self.cognitive_params[name].get('phi'),
            q=self.cognitive_params[name].get('q'),
            t_max=self.t_max,
            n_options=self.n_options,
        )

        # mapping column 0 to choice variable
        col = 6

        values = np.zeros(self.t_max, dtype=float)

        for t in range(self.t_max):

            choice = agent.choice(t)
            reward = self.play(choice)
            agent.save(choice=choice, t=t, reward=reward)

            if t != self.t_max - 1:

                agent.learn(choice=choice, t=t)

            if self.data[t, col] in (-1, 1):

                c = int(self.data[t, col])

                if c == -1:
                    c = 0

                v = 1 - agent.memory['p_softmax'][t, c]

                values[t] = v
            else:
                values[t] = -1

        return sum(np.log(values[values != -1])) * -1

    def play(self, choice):

        return self.rewards[choice][np.random.choice(
            [0, 1],
            p=self.p[choice]
        )]

    def save(self, results):

        data = {
            'params': self.params,
            'condition': self.condition,
            'results': results
        }

        pickle.dump(obj=data, file=open(f'data/data_{self.condition}.p', 'wb'))


if __name__ == '__main__':
    exit('Please run the main.py script.')