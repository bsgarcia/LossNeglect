import pickle
import scipy.io
import os


def load(fname):
    with open(f'fit/data/pooled/{fname}', 'rb') as f:
        return pickle.load(f)


def load2(fname):
    return scipy.io.loadmat(f'fit/data/experiment3/{fname}')['data']


def save(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(file=f, obj=obj)


def main():

    d1 = load('1.p')
    d2 = load('2.p')

    new = [j for i, j in sorted(d1.items())]

    for i, v in sorted(d2.items()):
        v[:, 6] = v[:, 4]
        new += [v, ]

    pickle.dump(obj=new, file=open('fit/data/pooled/full.p', 'wb'))

main()