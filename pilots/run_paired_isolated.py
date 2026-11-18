"""Use independent processes so ClaimNov's module patch lock is local to a worker."""

from multiprocessing import get_context

from edara_mvp_eval import paired_trial

_worker = None


def initialize(fn):
    global _worker
    _worker = fn


def process(item):
    return _worker(item)


class IsolatedExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def map(self, fn, items):
        jobs = list(items)
        if not jobs:
            return []
        # Fork before any worker clients are instantiated. Each child patches
        # its own copy of ClaimNov and writes only its assigned artifact.
        with get_context("fork").Pool(min(self.max_workers, len(jobs)), initializer=initialize,
                                      initargs=(fn,)) as pool:
            return pool.map(process, jobs, chunksize=1)


if __name__ == "__main__":
    paired_trial.ThreadPoolExecutor = IsolatedExecutor
    raise SystemExit(paired_trial.main())
