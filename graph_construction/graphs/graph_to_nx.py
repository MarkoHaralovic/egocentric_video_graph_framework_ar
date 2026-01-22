import networkx as nx

def full_action_graph_to_nx(g, directed=True):
   """
   g: FullActionGraph or GazePrunedActionSceneGraphs instance
   """
   G = nx.DiGraph() if directed else nx.Graph()

   for node_id, node in g.nodes.items():
      G.add_node(node_id, node_type=getattr(node, "node_type", None), idx=getattr(node, "idx", None))
      if getattr(node, "node_type", None) == "camera_wearer":
         cw_node_id =node_id
   # add edges
   for e in g.edges:
      u, v = e.src, e.dst
      G.add_edge(u, v, rel_idx=getattr(e, "rel_idx", None))

   return G, cw_node_id

def cw_eccentricity(Gnx, cw_node_id):
   G = Gnx.to_undirected()
   lengths = nx.single_source_shortest_path_length(G, cw_node_id)
   lengths.pop(cw_node_id, None)
   return max(lengths.values()) if lengths else 0

