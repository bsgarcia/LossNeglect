import pickle
import os.path

path = os.path.abspath('fit/data/pooled')


def exp(id='full'):
    with open(f'{path}/exp_data/{id}.p', 'rb') as f:
        return pickle.load(f)


def sim(id='full', condition=''):
    with open(f'{path}/sim_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)


def fit(id='full', condition=''):
    with open(f'{path}/fit_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)


def refit(id='full', condition=''):
    with open(f'{path}/refit_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)


