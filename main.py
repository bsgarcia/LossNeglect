#!/usr/bin/python3.6
import argparse
import os
import multiprocessing as mp
import tqdm

from simulation.env import Environment
from analysis import analysis
from parameters import risk_negative, risk_positive, risk_neutral


def run(cond):

    e = Environment(pbar=pbar, **cond)
    e.run()


def run_simulation():

    # declare progress bar
    global pbar
    pbar = tqdm.tqdm(
        total=risk_positive['t_max'] * risk_positive['n_agents'] * 3,
        desc="Computing simulations"
    )

    cond = risk_positive, risk_negative, risk_neutral

    with mp.Pool(processes=3) as p:
        for _ in p.imap_unordered(run, cond):
            pass

    pbar.close()


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






