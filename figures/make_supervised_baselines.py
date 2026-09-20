#!/usr/bin/env python3
"""Plot the result CSVs."""
from _data import validate
from _plots import baselines

if __name__ == "__main__":
    validate()
    baselines()
