"""Implemantation of the SGTC algorithm."""

import argparse

from sgtc.core import main

aparser = argparse.ArgumentParser()

if __name__ == "__main__":
    aparser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration file."
    )
    args = aparser.parse_args()
    main(args.config_file)
