"""_summary_"""

from typing import cast, Dict, Tuple, Union
import copy
from math import exp
import numpy as np
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
        self.graph: Union[Graph, None] = None
        self.mbs = self.configs.mbsratio or self.configs.kcenters
        self.save_tsp = save_tsp

    def _engine(self) -> Tuple[float, Graph]:
        """main process to do optimization."""
        alpha = self.configs.temperature
        g_best = copy.deepcopy(self.graph)
        save_graph_and_metrics(
            g_best,
            "initial",
            str(self.graph.params.seed),  # type: ignore
            str(self.graph.params.nnodes),  # type: ignore
            str(self.mbs),
        )
        lambda_current = lambda_best = self.graph.lsg()  # type: ignore
        for iteration in range(self.configs.maxitera):
            g_of_i = self._get_uniformly_at_random()

            lambda_candidate = g_of_i.lsg()

            differential = lambda_candidate - lambda_current
            p_accpt = round(exp(-differential / alpha), 10)

            if lambda_candidate > lambda_best:
                lambda_best = lambda_candidate
                g_best = copy.deepcopy(g_of_i)
                save_graph_and_metrics(
                    g_best,
                    "iter_" + str(iteration),
                    str(self.graph.params.seed),  # type: ignore
                    str(self.graph.params.nnodes),  # type: ignore
                    str(self.mbs),
                )

            if differential > 0 or np.random.rand() <= p_accpt:
                self.graph = copy.deepcopy(g_of_i)
                lambda_current = lambda_candidate

            alpha = self.configs.temperature // (iteration + 1)
        return lambda_best, cast(Graph, g_best)

    def _get_uniformly_at_random(self) -> Graph:
        """get a neighbor for a given graph."""
        g_of_i = copy.deepcopy(self.graph)
        nodes = self.graph.get_nodes()  # type: ignore
        edges = self.graph.get_edges()  # type: ignore
        while True:
            node_i, node_j = np.random.choice(nodes, 2, replace=False)  # type: ignore

            is_edge_in_graph = (node_i, node_j) in edges

            if is_edge_in_graph:
                g_of_i.remove_edge(node_i, node_j)  # type: ignore
                return cast(Graph, g_of_i)

            is_i_a_mbs = self.graph.get_nodes()[node_i][self._TYPE] == self._MACRO  # type: ignore
            is_j_a_mbs = self.graph.get_nodes()[node_j][self._TYPE] == self._MACRO  # type: ignore

            distance = np.linalg.norm(
                self.graph.get_nodes()[node_i][self._COORD]  # type: ignore
                - self.graph.get_nodes()[node_j][self._COORD]  # type: ignore
            )

            if is_i_a_mbs or is_j_a_mbs:
                if distance <= self.graph.params.mbsradius:  # type: ignore
                    g_of_i.add_edge(node_i, node_j)  # type: ignore
                    return g_of_i  # type: ignore

            if distance <= self.graph.params.scradius:  # type: ignore
                g_of_i.add_edge(node_i, node_j)  # type: ignore
                return g_of_i  # type: ignore
            continue

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
            self.graph = Graph(params)
            results[nnodes], _ = self._engine()
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
