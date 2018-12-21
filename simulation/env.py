#!/usr/bin/python3.6
import numpy as np
import pickle

from simulation.models import (QLearning,
                               AsymmetricQLearning,
                               PerseverationQLearning,
                               PriorQLearning,
                               AsymmetricPriorQLearning,
                               FullQLearning)


class Environment:

    def __init__(self, **kwargs):

        self.params = kwargs.copy()

        self.t_max = kwargs.get('t_max')

        self.cognitive_params = kwargs['cognitive_params']

        self.n_conds = kwargs.get('n_conds')
        self.conds = kwargs.get('conds')
        self.dic_conds = kwargs.get('dic_conds')

        self.n_options = kwargs.get('n_options')
        self.t_when_reversal_occurs = kwargs.get('t_when_reversal_occurs')

        self.subject_id = kwargs.get('subject_id')

        self.condition = kwargs.get('condition')
        self.experiment_id = kwargs.get('experiment_id')

        self.rewards = kwargs.get('rewards')
        self.p = kwargs.get('p')

    def run(self):

        data = {}

        for model in (QLearning,
                      AsymmetricQLearning,
                      PerseverationQLearning,
                      PriorQLearning,
                      AsymmetricPriorQLearning,
                      FullQLearning):

            choices = np.zeros(self.t_max, dtype=int)
            rewards = np.zeros(self.t_max, dtype=int)
            correct_choices = np.zeros(self.t_max, dtype=int)

            name = model.__name__

            agent = model(
                alpha=self.cognitive_params[name].get('alpha'),
                alpha0=self.cognitive_params[name].get('alpha0'),
                alpha1=self.cognitive_params[name].get('alpha1'),
                beta=self.cognitive_params[name].get('beta'),
                phi=self.cognitive_params[name].get('phi'),
                q=self.cognitive_params[name].get('q'),
                t_max=self.t_max,
                n_options=self.n_options,
                n_conds=self.n_conds
            )

            for t in range(self.t_max):

                if t in self.t_when_reversal_occurs:
                    assert 'status_quo' in self.condition
                    self.reversal()

                cond = self.conds[t]

                if self.condition == 'risk':
                    self.set_condition(cond)

                choice = agent.choice(cond=cond, t=t)

                # save that choice is correct or not
                correct_choices[t] = \
                    sum(self.rewards[choice] * self.p[choice]) > \
                    sum(self.rewards[int(not choice)] * self.p[int(not choice)])

                reward = self.play(choice)

                agent.save(choice=choice, t=t, reward=reward, cond=cond)

                if t != self.t_max - 1:
                    agent.learn(choice=choice, t=t, cond=cond)

            choices[:] = agent.memory['choices']
            rewards[:] = agent.memory['rewards']

            d = {
                model.__name__: {
                    'choices': choices,
                    'rewards': rewards,
                    'conds': self.conds
                }
            }

            data.update(d)

        self.save(data)

    def reversal(self):
        self.p = self.p[::-1]
        self.rewards = self.rewards[::-1]

    def set_condition(self, cond):
        self.p = self.dic_conds[cond]['p'].copy()
        self.rewards = self.dic_conds[cond]['rewards'].copy()

    def play(self, choice):

        return self.rewards[choice][np.random.choice(
            [0, 1],
            p=self.p[choice]
        )]

    def save(self, results):
        import os.path

        path = f'simulation/data/experiment{self.experiment_id}_{self.condition}'

        if not os.path.exists(path):
            os.mkdir(path=path)

        with open(f'{path}/{self.subject_id}.p', 'wb') as f:
            pickle.dump(obj=results, file=f)


if __name__ == '__main__':
    exit('Please run the main.py script.')
