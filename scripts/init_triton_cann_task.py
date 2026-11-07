#!/usr/bin/env python3
"""Generate an experimental Ascend950PR Triton task using the shared CANN harness.

All arguments match init_pypto_cann_task.py; the language is fixed to Triton.
Without --reference-file this builds the official GQA/library-reference task.
"""
import sys

from init_pypto_cann_task import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--language", "triton"])
