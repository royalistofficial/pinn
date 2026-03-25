import sys
import os
import unittest
import time

import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    set_seed()
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py", top_level_dir=project_root)

    n_tests = _count_tests(suite)
    print(f"\n{'='*70}")
    print(f"  Majorant PINN Test Suite -- {n_tests} tests")
    print(f"{'='*70}\n")

    t0 = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(
        f"  {result.testsRun} ran in {elapsed:.1f}s  |  "
        f"{'OK' if result.wasSuccessful() else 'FAILED'}  |  "
        f"{len(result.failures)} failures, {len(result.errors)} errors, "
        f"{len(result.skipped)} skipped"
    )
    print(f"{'='*70}\n")

    sys.exit(0 if result.wasSuccessful() else 1)

def _count_tests(suite):
    count = 0
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            count += _count_tests(item)
        else:
            count += 1
    return count

if __name__ == "__main__":
    main()