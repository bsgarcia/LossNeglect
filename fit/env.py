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

        self.n_options = kwargs.get('n_options')

        self.map = {
            1: {'choices': 6, 'conds': 2, 'rewards': 7},
            2: {'choices': 4, 'conds': 2, 'rewards': 7},
            '_full': {'choices': 6, 'conds': 2, 'rewards': 7},
            # 3: {'choice'}
        }

        exp_id = kwargs.get('exp_id')

        if kwargs.get('choices') is None:
            self.choices = self.data[:, self.map[exp_id]['choices']].astype(int)
            self.rewards = self.data[:, self.map[exp_id]['rewards']]
            self.conds = self.data[:, self.map[exp_id]['conds']].astype(int)
            self.mapping = True
        else:
            self.choices = kwargs['choices']
            self.rewards = kwargs['rewards']
            self.conds = kwargs['conds']
            self.mapping = False

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

        neg_log_likelihood = 0

        for t in range(self.t_max):

            assert self.choices[t] in [-1, 0, 1]
            # make agent play by using subject'choice and reward

            if self.mapping:
                if self.choices[t] == 0:
                    choice = None
                elif self.choices[t] == -1:
                    choice = 0
                else:
                    choice = 1
                cond = self.conds[t] - 1
            else:
                choice = self.choices[t]
                cond = self.conds[t]

            win = int(self.rewards[t] > 0)
            reward = 1 if win else -1

            if choice is not None:

                agent.save(choice=choice, t=t, reward=reward, cond=cond)

                if t != self.t_max - 1:
                    agent.learn(choice=choice, t=t, cond=cond)

                # print(agent.memory['p_softmax'][t, choice, cond])

                neg_log_likelihood += np.log(
                    agent.memory['p_softmax'][t, choice, cond]
                )
        return -neg_log_likelihood


if __name__ == '__main__':
    exit('Please run the main.py script.')
