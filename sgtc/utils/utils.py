"""_summary_"""


import dataclasses
import json
from typing import Any, Dict, List, Union

import networkx as nx  # type: ignore
import yaml  # type: ignore


__all__ = [
    "Configuration",
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
    temperature: float
    maxitera: int
    seeds: List[int]
    mode: str
    weight: float
    kcenters: Union[str, List[int], None]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Stores the names of the required parameters"""

    kcenters = "kcenters"
    main = "configs"
    maxi = "max_itera"
    mbsradius = "mbs_radius"
    mbsratio = "mbs_ratio"
    mode = "mode"
    nnodes = "nnodes"
    seeds = "seeds"
    scradius = "sc_radius"
    temp = "temperature"
    weight = "weight"


def load_configurations(config_file: str) -> Configuration:
    """loads user configurations."""
    with open(config_file, "r", encoding="utf-8") as filehandle:
        configs = yaml.load(filehandle, Loader=yaml.loader.FullLoader)[Parameters.main]
    parser = Configuration(
        nnodes=configs[Parameters.nnodes],
        mbsratio=configs[Parameters.mbsratio],
        mbsradius=configs[Parameters.mbsradius],
        scradius=configs[Parameters.scradius],
        temperature=configs[Parameters.temp],
        maxitera=configs[Parameters.maxi],
        seeds=configs[Parameters.seeds],
        mode=configs[Parameters.mode],
        weight=configs[Parameters.weight],
        kcenters=configs[Parameters.kcenters],
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
                    str(graph.lsg()) + "\n",
                ]
            )
        )
    nx.write_gpickle(
        graph, "./" + seed + "_" + n_nodes + "_" + mbs + "_" + fname + ".gpickle"
    )


def write_results(results: Dict[int, Any], filename: str) -> None:
    """saves results to a file."""
    with open(filename, "w", encoding="utf-8") as fhandle:
        json.dump(results, fhandle)
