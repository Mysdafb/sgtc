"""_summary_"""

from typing import cast, Dict, Tuple
import copy
from math import exp

import numpy as np
import tqdm

from .graph import Graph, GraphParameters
from .utils.utils import Configuration, save_graph_and_metrics


__all__ = ["SGTC"]


class SGTC:
    """Base class to run experiments."""

    _COORD = "coordinates"
    _MACRO = "mbs"
    _SCELL = "sc"
    _TYPE = "type"

    def __init__(self, configs: Configuration, save_tsp: bool = False) -> None:
        self.configs = configs
        self.mbs = self.configs.mbsratio or self.configs.kcenters
        self.save_tsp = save_tsp

    def _engine(self, graph: Graph) -> Tuple[float, Graph]:
        """main process to do optimization."""
        alpha = self.configs.temperature
        g_best = copy.deepcopy(graph)
        save_graph_and_metrics(
            g_best,
            "initial",
            str(graph.params.seed),
            str(graph.params.nnodes),
            str(self.mbs),
        )
        lambda_current = lambda_best = graph.lsg()
        for iteration in tqdm.trange(self.configs.maxitera):
            g_of_i = self._get_uniformly_at_random(graph)

            lambda_candidate = g_of_i.lsg()

            differential = lambda_candidate - lambda_current
            p_accpt = round(exp(-differential / alpha), 10)

            if lambda_candidate > lambda_best and g_of_i.is_feasible():
                lambda_best = lambda_candidate
                g_best = copy.deepcopy(g_of_i)
                save_graph_and_metrics(
                    g_best,
                    "iter_" + str(iteration),
                    str(graph.params.seed),
                    str(graph.params.nnodes),
                    str(self.mbs),
                )

            if differential > 0 or np.random.rand() <= p_accpt:
                graph = copy.deepcopy(g_of_i)
                lambda_current = lambda_candidate

            alpha = self.configs.temperature // (iteration + 1)
        return lambda_best, cast(Graph, g_best)

    def _get_uniformly_at_random(self, graph: Graph) -> Graph:
        """returns a neighbor for a given graph."""
        g_of_i = copy.deepcopy(graph)
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        node_i, node_j = np.random.choice(nodes, 2, replace=False)  # type: ignore

        is_edge_in_graph = (node_i, node_j) in edges

        if is_edge_in_graph:
            g_of_i.remove_edge(node_i, node_j)
            return g_of_i

        g_of_i.add_edge(node_i, node_j)
        return g_of_i

    def _run_multi_nodes(self, seed: int) -> Dict[int, float]:
        """
        Creates a for loop for multiple experiments with different number of nodes.
        """
        results = {}
        for nnodes in self.configs.nnodes:
            params = GraphParameters(
                self.configs.mode,
                nnodes,
                self.configs.mbsradius,
                self.configs.scradius,
                seed,
                self.configs.weight,
                self.configs.mbsratio,
                self.configs.kcenters,  # type: ignore
            )
            graph = Graph(params)
            graph.create_tsp_file()
            results[nnodes], _ = self._engine(graph)
        return results

    def _run_multi_seed(
        self,
    ) -> Dict[int, Dict[int, float]]:
        """Runs experiments for multiple seeds."""
        results = {}
        for seed in self.configs.seeds:
            results[seed] = self._run_multi_nodes(seed)
        return results

    def optimize(self) -> Dict[int, Dict[int, float]]:
        """Performs the optimization process using the SGTC algorithm."""
        results = self._run_multi_seed()
        return results
