#!/usr/bin/env python3
"""Plot the result CSVs."""
from _data import validate
from _plots import cascade

if __name__ == "__main__":
    validate()
    cascade()
    from _plots import cascade_all
    cascade_all()
