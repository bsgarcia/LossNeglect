import pickle as p
import scipy.io
import os


def load(file_name):
    with open(file_name, 'rb') as f:
        return p.load(f)


def loadmat(file_name):
    with open(file_name, 'rb') as f:
        return scipy.io.loadmat(file_name)


def main():

    pooled = load('fit/data/pooled/experiment2.p')

    ind = [
        loadmat(f'fit/data/experiment2/{fname}')['data'][0][0]
        for fname in sorted(os.listdir('fit/data/experiment2/'))
    ]

    import re
    idx = [int(re.search('\d\d\d', i).group(0)) for i in sorted(os.listdir('fit/data/experiment2/'))]

    for idx, id in enumerate(idx):
        assert (ind[idx] == pooled[id]).all()



main()
