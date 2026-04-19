#!/usr/bin/env python3
"""Backward-compatible CLI entrypoint for the synthetic-user simulation.

Prefer: ``python -m simulation`` (same implementation).
"""
from simulation.cli import main

if __name__ == "__main__":
    main()
