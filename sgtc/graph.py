"""_summary_"""

import dataclasses
from itertools import combinations
from typing import Dict, List, Iterable, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

__all__ = ["Graph"]


@dataclasses.dataclass
class GraphParameters:  # pylint: disable=too-many-instance-attributes
    """Stores main parameters to build a graph."""

    mode: str
    nnodes: int
    mbsradius: float
    scradius: float
    seed: int
    weight: float
    mbsratio: Optional[float] = None
    kcenters: Optional[List[int]] = None


class Graph:
    """_summary_"""

    _COORD = "coordinates"
    _FULL = "fully"

    _GRAPHOPTIONS = {
        "font_size": 12,
        "node_size": 300,
        "node_color": "blue",
        "edge_color": "green",
        "linewidths": 2,
        "width": 1,
    }

    _MACRO = "mbs"
    _SCELL = "sc"
    _SPECIALMODE = "empty"
    _TYPE = "type"
    _VORO = "voronoi"
    _WEIGHT = "weight"

    def __init__(self, params: GraphParameters) -> None:
        np.random.seed(params.seed)
        self.params = params

        self.__graph = self._create_graph()
        self._add_mbs()

        if self.params.mode != self._SPECIALMODE:
            self._incorporate_edges()

    def _add_mbs(self) -> None:
        if self.params.kcenters is not None:
            for node in self.params.kcenters:
                self.__graph.nodes[node][self._TYPE] = self._MACRO

        elif self.params.mbsratio is not None:
            selected = np.random.choice(  # type: ignore
                self.get_nodes(),
                int(round(self.params.nnodes * self.params.mbsratio)),
                replace=False,
            )
            for node in selected:
                self.__graph.nodes[node][self._TYPE] = self._MACRO

        else:
            raise RuntimeError("Not MBS ratio or Kcenters defined!")

    def _create_graph(self) -> nx.Graph:
        """Creates a graph object with a given number of nodes."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.params.nnodes))
        for node_id in range(self.params.nnodes):
            graph.nodes[node_id][self._COORD] = np.random.rand(2)
            graph.nodes[node_id][self._TYPE] = self._SCELL
        return graph

    def _create_fully_connected_graph(self):
        for first in range(self.params.nnodes - 1):
            second = first + 1

            node_i = self.__graph.nodes[first]
            while second < self.params.nnodes:
                node_j = self.__graph.nodes[second]

                distance = np.linalg.norm(node_i[self._COORD] - node_j[self._COORD])

                is_i_macro = node_i[self._TYPE] == self._MACRO
                is_j_macro = node_j[self._TYPE] == self._MACRO

                if is_i_macro or is_j_macro:
                    if distance <= self.params.mbsradius:
                        self.add_edge(first, second)
                else:
                    if distance <= self.params.scradius:
                        self.add_edge(first, second)
                second += 1

    def _create_voronoi_graph(self) -> None:
        """Creates a voronoi graph with its centroids fully connected."""
        center_nodes = {
            node
            for node in range(self.params.nnodes)
            if self.get_nodes()[node][self._TYPE] == self._MACRO  # type: ignore
        }

        self._get_complete_graph()

        cells = nx.voronoi_cells(self.__graph, center_nodes, weight=self._WEIGHT)

        self.__graph.remove_edges_from(self.get_edges())

        all_possible_edges_for_cells = list(combinations(list(center_nodes), 2))

        for edge in all_possible_edges_for_cells:
            distance = np.linalg.norm(
                self.__graph.nodes[edge[0]][self._COORD]
                - self.__graph.nodes[edge[1]][self._COORD]
            )
            if distance <= self.params.mbsradius:
                self.add_edge(edge[0], edge[1])

        for key in cells:
            for value in cells[key]:
                if value != key:
                    distance = np.linalg.norm(
                        self.__graph.nodes[key][self._COORD]
                        - self.__graph.nodes[value][self._COORD]
                    )

                    if distance <= self.params.mbsradius:
                        self.add_edge(key, value)

    def _get_complete_graph(self) -> None:
        """create a complete weighted graph."""
        for node_i in range(self.params.nnodes - 1):
            node_j = node_i + 1
            while node_j < self.params.nnodes:
                distance = np.linalg.norm(
                    self.__graph.nodes[node_i][self._COORD]
                    - self.__graph.nodes[node_j][self._COORD]
                )

                self.__graph.add_edge(
                    node_i,
                    node_j,
                    weight=distance,
                )

                node_j += 1

    def _incorporate_edges(self) -> None:
        """Adds edges based on the mode parameter."""
        if self.params.mode == self._FULL:
            self._create_fully_connected_graph()
        elif self.params.mode == self._VORO:
            self._create_voronoi_graph()
        else:
            raise NotImplementedError(f"{self.params.mode} mode is nor supported!")

    def add_edge(self, node_1: int, node_2: int) -> None:
        """Interface to add single edges."""
        self.__graph.add_edge(node_1, node_2, weight=self.params.weight)

    def create_tsp_file(self) -> None:
        """Saves the graph in tsp format."""
        mbs: Union[float, List[int], None] = (
            self.params.mbsratio or self.params.kcenters
        )
        filename = (
            str(self.params.seed)
            + "_"
            + str(self.params.nnodes)
            + "_"
            + str(mbs)
            + ".tsp"
        )
        with open(filename, "w", encoding="utf-8") as fhandle:
            for idx, node in enumerate(self.get_nodes(True)):
                fhandle.write(
                    " ".join(
                        [
                            str(idx),
                            str(node[1][self._COORD][0]),
                            str(node[1][self._COORD][1]),
                            "\n",
                        ]
                    )
                )

    def find_paths(self) -> Dict[int, Dict[int, List[int]]]:
        """Returns all pairs shortest paths."""
        paths: Dict[int, Dict[int, List[int]]] = dict(
            nx.all_pairs_shortest_path(self.__graph)
        )
        return paths

    def get_edge_concurrence(self, plot: bool = False) -> List[int]:
        """
        Computes the number of times an edge is visited over all pairs shortest path.
        """
        all_possible_paths = self.find_paths()
        concurrence = dict.fromkeys(self.get_edges(), 0)

        for single_node_paths in all_possible_paths.values():
            for path in list(single_node_paths.values())[1:]:
                for node_idx_i, _ in enumerate(path):
                    node_idx_j = node_idx_i + 1

                    while node_idx_j < len(path):
                        if (path[node_idx_i], path[node_idx_j]) in concurrence:
                            concurrence[(path[node_idx_i], path[node_idx_j])] += 1
                        elif (path[node_idx_j], path[node_idx_i]) in concurrence:
                            concurrence[(path[node_idx_j], path[node_idx_i])] += 1

                        node_idx_j += 1
        if plot:
            figure = plt.figure(figsize=(15.0, 7.0))
            bars = figure.add_axes([0, 0, 1, 1])
            edges = [str(i) for i in self.get_edges()]
            bars.bar(edges, list(concurrence.values()))
            bars.set_ylabel("Frequency")
            bars.set_xlabel("Edge")
            plt.show()
        return list(concurrence.values())

    def get_adj_matrix(self) -> np.ndarray:
        """Returns the adjacency matrix of the graph."""
        return nx.adjacency_matrix(self.__graph).todense()

    def get_edges(self, data: bool = False) -> Iterable:
        """Returns a list containing the graph's edges."""
        return self.__graph.edges(data=data)

    def get_nodes(self, data: bool = False) -> Iterable:
        """Returns a list containing the graph's nodes."""
        return self.__graph.nodes(data=data)

    def graph_cost(self) -> float:
        """
        Computes the graph's cost defined as the sum of the distance between nodes.
        """
        c_of_g: float = 0.0
        edges = self.__graph.edges()

        for edge in edges:
            c_of_g += np.linalg.norm(  # type: ignore
                self.__graph.nodes[edge[0]][self._COORD]
                - self.__graph.nodes[edge[1]][self._COORD]
            )  # Eq. 1

        return c_of_g

    def graph_diameter(self) -> int:
        """Computes the longest shortest path."""
        d_of_g = 0
        all_possible_paths = self.find_paths()

        for source in all_possible_paths:
            for path in all_possible_paths[source].values():
                if d_of_g < len(path[1:]):
                    d_of_g = len(path[1:])

        return d_of_g

    def graph_entropy(self) -> float:
        """Computes graph's entropy using edge concurrence."""
        s_of_g = 0

        f_of_es = (  # Numerator in Eq. 3
            self.get_edge_concurrence()
        )  # counts the number of shortest paths that include link e, for all e in E

        eta = sum(f_of_es)  # Denominator in Eq. 3
        for f_of_e in f_of_es:  # Sumation in Eq. 2
            pr_of_e = f_of_e / eta  # Eq. 3
            s_of_g += pr_of_e * np.log2(pr_of_e)  # Eq. 2

        return -s_of_g

    def graph_plot(self) -> None:
        """
        Draws nodes in their true position on the cartesian plane.
        """
        nodes = self.__graph.nodes()
        xaxis = []
        yaxis = []
        pos = {}

        for node in range(self.params.nnodes):
            xaxis.append(nodes[node][self._COORD][0])
            yaxis.append(nodes[node][self._COORD][1])
            pos[node] = xaxis[node], yaxis[node]

        options = self._GRAPHOPTIONS
        nx.draw_networkx(self.__graph, pos, **options)

        mbs = [
            i
            for i in range(self.params.nnodes)
            if self.__graph.nodes[i][self._TYPE] == self._MACRO
        ]

        nx.draw_networkx_nodes(
            self.__graph,
            pos,
            nodelist=mbs,
            node_color="tab:red",
            node_size=300,
            alpha=0.8,
        )

    def graph_robustness(self) -> float:
        """Returns the number of edges needed to disconnect the grapgh."""
        if self.is_connected():
            return len(nx.minimum_edge_cut(self.__graph))
        return 0

    def is_connected(self) -> bool:
        """Checks the connectivity of the graph."""
        return nx.is_connected(self.__graph)

    def is_feasible(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        if not self.is_connected():
            return False

        edges = self.get_edges()

        for edge in edges:
            node_i = self.get_nodes()[edge[0]]  # type: ignore
            node_j = self.get_nodes()[edge[1]]  # type: ignore

            distance = np.linalg.norm(node_i[self._COORD] - node_j[self._COORD])

            is_i_a_mbs = node_i[self._TYPE] == self._MACRO
            is_j_a_mbs = node_j[self._TYPE] == self._MACRO

            if is_i_a_mbs or is_j_a_mbs:
                if distance > self.params.mbsradius:
                    return False

            elif distance > self.params.scradius:
                return False

        return True

    def lsg(self) -> float:
        """Computes the laplacian spectral gap of the grapgh."""
        return round(sorted(nx.laplacian_spectrum(self.__graph))[1], 5)

    def remove_edge(self, node_1, node_2) -> None:
        """Interface to remove single edges."""
        self.__graph.remove_edge(node_1, node_2)
