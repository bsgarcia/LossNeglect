import pickle


def exp(id='full'):
    with open(f'pooled/exp_data/{id}.p', 'rb') as f:
        return pickle.load(f)


def sim(id='full', condition=''):
    with open(f'pooled/exp_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)


def fit(id='full', condition=''):
    with open(f'pooled/fit_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)


def refit(id='full', condition=''):
    with open(f'pooled/refit_data/{id}{condition}.p', 'rb') as f:
        return pickle.load(f)

