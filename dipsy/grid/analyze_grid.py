#!/usr/bin/env python
"""
Batch analyze simulation results.
"""
# %%
import argparse
import sys
from multiprocessing import Pool
import time as walltime
from pathlib import Path
from functools import partial

import h5py
import dipsy


def main():

    start = walltime.time()

    # %% -------------- argument parsing ------------------

    # this first parser is the default parser of general options
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF, conflict_handler='resolve')
    PARSER.add_argument('analysis', help='python file with analysis function', type=str)
    PARSER.add_argument('-c', '--cores', help='how many cores to use', type=int, default=1)
    PARSER.add_argument('-h', '--help', help='display help', action='store_true')
    PARSER.add_argument('-t', '--test', help='for testing: only run this single model', type=int, default=None)

    ARGS, unknown_args = PARSER.parse_known_args()
    if ARGS.help:
        unknown_args = ['-h'] + unknown_args
        PARSER.print_help()

    # %% ------- now import the analysis function ------------

    analysis_file = Path(ARGS.analysis).resolve()

    if not analysis_file.is_file():
        print(f'grid file {analysis_file} not found.')
        sys.exit(1)

    print(f'importing analysis function from {analysis_file}')
    analysis = dipsy.utils.remote_import(analysis_file)

    # %% ------------- now that we imported the analysis -------------
    # - get the second parser
    # - parse the other arguments
    # - based on the arguments, determine any specific settings

    ARGS2 = analysis.PARSER.parse_args(unknown_args)

    settings = analysis.process_args(ARGS2)

    fname_in = ARGS2.file
    fname_out = settings['fname_out']

    # %% -------- open the data file -----------

    with h5py.File(fname_in, 'r') as fid:
        n_data = len(fid)
        keys = list(fid.keys())

    # %% ----------------- parallel execution ---------------
    indices = range(n_data)
    if ARGS.test is not None:
        print(f'TESTING: only analyzing simulation #{ARGS.test}')
        indices = [ARGS.test]

    pool = Pool(processes=ARGS.cores)

    n_sim = len(indices)
    keys = [keys[i] for i in indices]
    failed_keys = []

    with h5py.File(Path(fname_out).with_suffix('.hdf5'), 'w') as f:
        for i, res in enumerate(pool.imap(partial(analysis.parallel_analyze, settings=settings), keys)):
            if res is False:
                failed_keys += [keys[i]]
            else:
                dipsy.utils.hdf5_add_dict(f, keys[i], res)
                del res
            print(f'\rRunning ... {(i+1) / n_sim:.1%}', end='', flush=True)

    print('\r--------- DONE ---------')

    end = walltime.time()
    print('{} of {} simulations analyzed in {:.3g} minutes'.format(n_sim - len(failed_keys), n_sim, (end - start) / 60))
    if len(failed_keys) > 0:
        print('Analysis failed for the following keys:')
        for key in failed_keys:
            print('- ' + key)


if __name__ == '__main__':
    main()
