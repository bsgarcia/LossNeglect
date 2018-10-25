#!/usr/bin/python3.6
import argparse
import tqdm
import os
import numpy as np
import multiprocessing as mp
import hyperopt as hp
import tqdm
import pickle


from simulation.env import Environment
from analysis import analysis
from parameters import risk_negative, risk_positive, risk_neutral


# declare global progress bar
pbar = tqdm.tqdm(
    total=1000,#risk_positive['t_max'] * risk_positive['n_agents'] * 3,
    desc="Hyperopt"
)


def run(*args):

    n_reversal, t_max = args[0][:]

    risk_positive['t_max'] = int(t_max)

    risk_positive['t_when_reversal_occurs'] = \
            np.arange(
                0,
                t_max + 1,
                t_max // (n_reversal + 1),
            )[1:-1]

    e = Environment(pbar=pbar, **risk_positive)
    a = e.run()
    pbar.update()

    return a


def run_simulation():

    best = hp.fmin(
        fn=run,
        space=[hp.hp.quniform('n_reversal', 1, 8, 1), hp.hp.quniform('t_max', 41, 200, 20)],
        algo=hp.tpe.suggest,
        max_evals=1000
    )

    pbar.close()

    print(best)
    pickle.dump(obj=best, file=open('data/best.p', 'wb'))


def run_analysis():

    tqdm.tqdm.write('Generating graphs...')

    analysis.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", action="store_true",
                        help="run simulations")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="run analysis and display figures")
    args = parser.parse_args()

    is_running_in_pycharm = "PYCHARM_HOSTED" in os.environ

    if args.simulation:
        run_simulation()

    if args.analysis:
        run_analysis()

    if not args.simulation and not args.analysis and not is_running_in_pycharm:
        parser.print_help()

    if is_running_in_pycharm:
        run_analysis()






