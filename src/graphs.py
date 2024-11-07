from tkinter import E
import networkx as nx
import pandas as pd
from collections import Counter
from data_manip import Manipulator


class BaseGraphMetric:
    def __init__(self, name: str, description: str, graph: nx.Graph = None):
        self.metric_name = name
        self.metric_description = description
        self.graph = graph

    def apply_metric(self):
        raise NotImplementedError("Must implement apply_metric method")

class DegreeCentralityGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Degree Centrality", "Calculates the degree centrality of each node", graph)

    def apply_metric(self, weight=None):
        return nx.degree_centrality(self.graph)

class BetweennessCentralityGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Betweenness Centrality", "Calculates the betweenness centrality of each node", graph)

    def apply_metric(self, weight=None):
        return nx.betweenness_centrality(self.graph)

class ClosenessCentralityGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Closeness Centrality", "Calculates the closeness centrality of each node", graph)

    def apply_metric(self, weight=None):
        if weight is None:
            return nx.closeness_centrality(self.graph)
        else:
            return nx.closeness_centrality(self.graph, distance=weight)

class ShortestPathGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Shortest Path", "Finds the shortest path between nodes", graph)

    def apply_metric(self, source, target):
        return nx.shortest_path(self.graph, source=source, target=target)

class DegreeDistributionGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Degree Distribution", "Something", graph)

    def _build_degree_count(self):
        degree_sequence = sorted((d for _, d in self.graph.degree()), reverse=True)
        degree_df = pd.DataFrame({"Degree": degree_sequence})
        return degree_df.value_counts()

    def apply_metric(self, weight=None):
        return self._build_degree_count()

class WeightDistributionGraphMetric(BaseGraphMetric):
    def __init__(self, graph: nx.Graph):
        super().__init__("Weight Distribution", "Something", graph)

    def apply_metric(self, weight):
        edges = self.graph.edges(data=True)
        edges_with_weights = [(u, v, d[weight]) for u, v, d in edges]
        edge_df = pd.DataFrame(edges_with_weights, columns=['Node1', 'Node2', 'Weight'])
        return edge_df["Weight"].value_counts()


class IPGraph:
    def __init__(self, manipulator: Manipulator = None, graph_type: str = "graph", weight_type: str = "ascending", weight_column: str = None, labels: list = None):
        self.manipulator = manipulator
        self.label_field = 'Class' # self.manipulator.label_field
        self.src_ip = self.manipulator.src_ip # "Source IP" #self.manipulator.src_ip
        self.dst_ip = self.manipulator.dst_ip # "Destination IP" #self.manipulator.dst_ip
        self.df = self.manipulator.processed_df # pd.DataFrame({
           # 'Source IP': ['192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.3'],
           # 'Destination IP': ['192.168.1.2', '192.168.1.2', '192.168.1.3', '192.168.1.1', '192.168.1.2'],
           # 'Class': ['A', 'B', 'A', 'C', 'B']
        #}) # self.manipulator.processed_df
        self.graph_type = graph_type
        self.weight_type = weight_type
        self.weight_column = weight_column
        self.weight = 'weight' if self.weight_column and self.weight_column in self.df.columns else None
        self.graph = None
        self.filter_classes(labels)
        self._init_graph(self.graph_type, self.weight_type, self.weight_column)

    def filter_classes(self, labels=None):
        if labels is not None:
            self.df = self.df[self.df[self.label_field].isin(labels)]

    def _init_graph(self, graph_type: str = "graph", weight_type: str = "ascending", weight_column: str = None):
        if graph_type.lower() not in ["graph", "digraph"]:
            raise ValueError("graph_type must be one of: graph, digraph")

        if weight_type.lower() not in ["ascending", "descending"]:
            raise ValueError("weight_type must be one of: ascending, descending")

        print("[*] Initialising Graph")
        if graph_type == "graph":
            pairs = self._prepare_undirected()
        elif graph_type == "digraph":
            pairs = list(zip(self.df[self.src_ip], self.df[self.dst_ip]))

        # Initialise graph
        G = nx.DiGraph() if graph_type == "digraph" else nx.Graph()

        # Use weight column if available, otherwise calculate frequency
        if weight_column and weight_column in self.df.columns:
            weights = self.df[weight_column].tolist()
        else:
            pairs_count = Counter(pairs)
            weights = [pairs_count[pair] for pair in pairs]

        for (source, destination), weight in zip(pairs, weights):
            if weight_type == "descending":
                weight = 1 / weight

            # Add edges with weights
            if G.has_edge(source, destination):
                G[source][destination][self.weight] += weight
            else:
                G.add_edge(source, destination, weight=weight)

        self.graph = G

    def _prepare_undirected(self):
        # Ensure ordered pairs for undirected graph
        pairs = []
        for src, dst in zip(self.df[self.src_ip], self.df[self.dst_ip]):
            if src > dst:
                src, dst = dst, src
            pairs.append((src, dst))
        return pairs

    def apply_metric(self, metric: BaseGraphMetric, **kwargs):
        print("[*] Applying Graph Metric")
        metric.graph = self.graph
        return metric.apply_metric(**kwargs)


# Example usage

m = Manipulator("/home/rob/Documents/PhD/WhiffSuite/tests/csvs", metadata_path="/home/rob/Documents/PhD/WhiffSuite/tests/metadata/our_metadata.json", target_label="Attack", metadata_manip=False)

ip_graph = IPGraph(m)
ip_graph._init_graph(graph_type='graph', weight_column=m.backward_packets_field)
metric = WeightDistributionGraphMetric(ip_graph.graph) # 196.133.39.158
result = ip_graph.apply_metric(metric, weight="weight")
print(result)

# If weight is defined, we need to pass the weight to the appropriate metrics

# Also, probably want to do some representative downsampling as I imagine shoving an entire
# dataset through these measures will take hours

# Want to expand to the following measures
# Graph Density
# Clustering Metrics
# Degree histogram
# Connected components --- maybe parts of the network are isolated from other parts?
# Edge betweenness -- 
# Node redudancy/overlaps