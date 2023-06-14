# stdlib
import multiprocessing
import os

# adjutorium absolute
import adjutorium.logger as log


def cpu_count() -> int:
    try:
        n_jobs = int(os.environ["N_JOBS"])
    except BaseException as e:
        log.error(f"failed to get env n_jobs: {e}")
        n_jobs = multiprocessing.cpu_count()
    log.info(f"Using {n_jobs} cores")
    return n_jobs
