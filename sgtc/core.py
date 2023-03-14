"""_summary_"""

from typing import cast

from .utils import load_configurations, read_kcenters, write_results
from .sgtc import SGTC


def main(configs: str) -> None:
    """main function to run experiments."""
    configurations = load_configurations(configs)

    if configurations.kcenters is not None:
        kcenters = cast(str, configurations.kcenters)
        configurations.kcenters = read_kcenters(kcenters)

    sgtc_algorithm = SGTC(configurations, True)

    results = sgtc_algorithm.optimize()

    write_results(results)
