from typing import List
from ..graphs.base_graph import Node,Edge,SkeletonGraph
from ..graphs.full_graph import FullActionGraph
from ..graphs.pruned_graph import PrunedActionGraph

class GraphDataset(Dataset):
   """
   Wraps EASGData into explicit Full / Pruned graphs.
   """

   def __init__(self, path_annts, path_data, split, verbs, objs, rels):
      # reuse your previous dataset to build graph_batch dicts
      self.graphs = ...
      self.skeleton = SkeletonGraph(verbs, objs, rels)

      self.full_graphs: List[FullActionGraph] = []
      self.pruned_graphs: List[PrunedActionGraph] = []

      for graph_batch in self.graphs:
         full_g = FullActionGraph.from_easg_graph(self.skeleton, graph_batch, rels)
         pruned_g = PrunedActionGraph.from_full_graph(full_g)
         self.full_graphs.append(full_g)
         self.pruned_graphs.append(pruned_g)

   def __len__(self):
      return len(self.full_graphs)

   def __getitem__(self, idx) -> Tuple[FullActionGraph, PrunedActionGraph]:
      return self.full_graphs[idx], self.pruned_graphs[idx]
