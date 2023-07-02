"""_summary_"""

import os
from typing import cast

from sgtc.utils import Configuration, load_configurations, read_kcenters, write_results
from sgtc.sgtc import SGTC

__all__ = ["main"]


def main(configs: str) -> None:
    """main function to run experiments."""
    configurations = load_configurations(configs)

    if configurations.kcenters is not None:
        run_for_kcenters(configurations)
    else:
        run(configurations)


def run(configurations: Configuration, outfile: str = "results.json") -> None:
    """Runs the algorithm and saves the experiment results

    Args:
        configurations (Configurations): User defined parameters.
    """

    sgtc_algorithm = SGTC(configurations, True)

    results = sgtc_algorithm.optimize()

    write_results(results, outfile)


def run_for_kcenters(configurations: Configuration) -> None:
    """Allows modularity and easy maintainability

    Args:
        configurations (Configuration): User defined parameters.
    """
    kcenters = cast(str, configurations.kcenters)
    if not os.path.exists(kcenters):
        raise FileNotFoundError(f"Directory {kcenters} doesn't exists!")

    list_of_kcenters_files = os.listdir(kcenters)
    for kfile in list_of_kcenters_files:
        configurations.kcenters = read_kcenters(kfile)

        out_file = kfile.split(".", maxsplit=1)[0] + "_results.json"
        run(configurations, out_file)
