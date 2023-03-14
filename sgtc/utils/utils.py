"""_summary_"""


import dataclasses
import json
from typing import Any, Dict, List, Union

import networkx as nx  # type: ignore
import yaml  # type: ignore


__all__ = [
    "load_configurations",
    "read_kcenters",
    "save_graph_and_metrics",
    "write_results",
]


@dataclasses.dataclass
class Configuration:  # pylint: disable=too-many-instance-attributes
    """stores user configurations."""

    nnodes: List[int]
    mbsratio: float
    mbsradius: float
    scradius: float
    temperature: int
    maxitera: int
    seeds: List[int]
    mode: str
    weight: float
    kcenters: Union[str, List[int], None]


def load_configurations(config_file: str) -> Configuration:
    """loads user configurations."""
    _main = "configs"
    _nnodes = "nnodes"
    _mbsratio = "mbs_ratio"
    _mbsradius = "mbs_radius"
    _scradius = "sc_radius"
    _temp = "temperature"
    _maxi = "max_itera"
    _seeds = "seeds"
    _mode = "mode"
    _weight = "weight"
    _kcenters = "kcenters"
    with open(config_file, "r", encoding="utf-8") as filehandle:
        configs = yaml.load(filehandle, Loader=yaml.loader.FullLoader)[_main]
    parser = Configuration(
        nnodes=configs[_nnodes],
        mbsratio=configs[_mbsratio],
        mbsradius=configs[_mbsradius],
        scradius=configs[_scradius],
        temperature=configs[_temp],
        maxitera=configs[_maxi],
        seeds=configs[_seeds],
        mode=configs[_mode],
        weight=configs[_weight],
        kcenters=configs[_kcenters],
    )
    return parser


def read_kcenters(filename: str) -> List[int]:
    """
    Reads a file with node ids for k-centers,
    each element should be separate for a single space.
    """
    with open(file=filename, mode="r", encoding="utf-8") as fhandle:
        fhandle.readline()
        kcenters = fhandle.readline().split(" ")[:-1]
    return list(map(int, kcenters))


def save_graph_and_metrics(
    graph: Any, fname: str, seed: str, n_nodes: str, mbs: str
) -> None:
    """
    saves a graph with its corresponding metrics in the current working directory.
    """
    filename = "./" + seed + "_" + n_nodes + "_" + mbs + "_metrics.txt"
    with open(file=filename, mode="a", encoding="utf-8") as fhandle:
        fhandle.write(
            ",".join(
                [
                    str(graph.graph_diameter()),
                    str(graph.graph_cost()),
                    str(graph.graph_entropy()),
                    str(graph.graph_robustness()),
                    str(graph.LSG()) + "\n",
                ]
            )
        )
    nx.write_gpickle(
        graph, "./" + seed + "_" + n_nodes + "_" + mbs + "_" + fname + ".gpickle"
    )


def write_results(results: Dict[int, Any]) -> None:
    """saves results to a file."""
    with open("results.json", "w", encoding="utf-8") as fhandle:
        json.dump(results, fhandle)
