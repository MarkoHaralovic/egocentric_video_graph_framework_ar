from .full_graph import FullActionGraph
from .base_graph import Node, Edge
from typing import Dict, List

class PrunedActionGraph(FullActionGraph):
   def __init__(self, verbs, objs, rels):
      super(FullActionGraph).__init__(verbs, objs, rels)
      self.nodes: Dict[int, Node] = {}
      self.edges: List[Edge] = []

   def create_graph(self):
      ...