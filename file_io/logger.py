from __future__ import annotations
import os
import time

class FileLogger:
    def __init__(self, path: str, also_print: bool = True):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._print = also_print
        self._t0 = time.time()

    def __call__(self, msg: str) -> None:
        elapsed = time.time() - self._t0
        line = f"[{elapsed:8.1f}s] {msg}"
        if self._print:
            print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def section(self, title: str) -> None:
        bar = "=" * 70
        self(f"\n{bar}\n  {title}\n{bar}")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
