import time


def progress_iter(iterable, *, total=None, desc=None, disable=False):
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        return _PlainProgress(iterable, total=total, desc=desc)


class _PlainProgress:
    def __init__(self, iterable, *, total=None, desc=None):
        self.iterable = iterable
        self.total = total
        self.desc = desc or "progress"
        self.count = 0
        self.start = time.time()
        self._last_print = 0.0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.count += 1
            self._maybe_print()
        self._print(final=True)

    def set_postfix(self, data=None, **kwargs):
        return None

    def update(self, n=1):
        self.count += n
        self._maybe_print()

    def close(self):
        self._print(final=True)

    def _maybe_print(self):
        now = time.time()
        if now - self._last_print >= 30 or self.count == 1 or (self.total and self.count >= self.total):
            self._print()
            self._last_print = now

    def _print(self, final=False):
        total = self.total if self.total is not None else "?"
        elapsed = time.time() - self.start
        tag = "done" if final else "progress"
        print(f"[{self.desc}] {tag}: {self.count}/{total} elapsed={elapsed:.1f}s", flush=True)
