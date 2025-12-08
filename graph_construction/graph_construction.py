import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from typing import Dict, List, Tuple, Optional


class EASGGraphBuilder:
    """
    Builds EASG-style action scene graphs from CSV annotations.
    Handles camera wearer, verbs, objects, and relationships.
    """
    
    def __init__(self, csv_data: str = None, df: pd.DataFrame = None):
        """
        Initialize with either CSV string or DataFrame.
        
        Args:
            csv_data: CSV string
            df: pandas DataFrame
        """
        if df is not None:
            self.df = df
        elif csv_data:
            from io import StringIO
            self.df = pd.read_csv(StringIO(csv_data))
        else:
            raise ValueError("Must provide either csv_data or df")
    
    def parse_row(self, row_idx: int = 0) -> Dict:
        """
        Parse a single row from the dataframe into graph components.
        
        Args:
            row_idx: Row index to parse
            
        Returns:
            Dictionary with graph components
        """
        row = self.df.iloc[row_idx]
        
        # Parse all_objects
        try:
            all_objects = ast.literal_eval(row['all_objects'])
        except:
            all_objects = []
        
        # Parse preposition_object_pairs
        try:
            prep_pairs = ast.literal_eval(row['preposition_object_pairs'])
        except:
            prep_pairs = []
        
        graph_data = {
            'frame_id': row['frame_id'],
            'subject': row['subject'],
            'verb': row['verb'],
            'verb_type': row['verb_type'],
            'direct_object': row['direct_object'] if pd.notna(row['direct_object']) else None,
            'all_objects': all_objects,
            'phrasal_verb': row['phrasal_verb'] if pd.notna(row['phrasal_verb']) else None,
            'prep_pairs': prep_pairs,
        }
        
        return graph_data
    
    def build_graph_structure(self, row_idx: int = 0) -> Dict:
        """
        Build the complete graph structure with nodes and edges.
        
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        data = self.parse_row(row_idx)
        
        nodes = []
        edges = []
        
        # 1. Camera Wearer Node (root)
        nodes.append({
            'id': 'cw',
            'label': 'CW',
            'full_label': data['subject'].replace('_', ' '),
            'type': 'camera_wearer',
            'position': (0, 0)  # Will be set during layout
        })
        
        # 2. Verb Node
        verb_label = data['phrasal_verb'] if data['phrasal_verb'] else data['verb']
        nodes.append({
            'id': 'verb',
            'label': verb_label.upper(),
            'full_label': verb_label,
            'type': 'verb',
            'verb_type': data['verb_type'],
            'position': (0, 1)
        })
        
        # Edge from CW to Verb
        edges.append({
            'from': 'cw',
            'to': 'verb',
            'label': 'direct obj',
            'type': 'action'
        })
        
        # 3. Object Nodes
        for i, obj in enumerate(data['all_objects']):
            node_id = f'obj_{i}'
            nodes.append({
                'id': node_id,
                'label': obj.upper(),
                'full_label': obj,
                'type': 'object',
                'is_direct_object': obj == data['direct_object'],
                'position': (i, 2)
            })
            
            # Find preposition for this object
            prep = None
            for pair in data['prep_pairs']:
                if obj in pair:
                    prep = pair[obj]
                    break
            
            # Edge from Verb to Object
            edge_label = prep if prep else ''
            edges.append({
                'from': 'verb',
                'to': node_id,
                'label': edge_label,
                'type': 'relationship',
                'is_direct': obj == data['direct_object']
            })
        
        # 4. Build relationships between objects (if any)
        # You can extend this based on your specific needs
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'frame_id': data['frame_id'],
                'verb_type': data['verb_type'],
                'caption': self._generate_caption(data)
            }
        }
    
    def _generate_caption(self, data: Dict) -> str:
        """Generate natural language caption from graph data."""
        caption = data['subject'].replace('_', ' ')
        
        if data['phrasal_verb']:
            caption += ' ' + data['phrasal_verb']
        else:
            caption += ' ' + data['verb']
        
        if data['direct_object']:
            caption += ' ' + data['direct_object']
        
        for pair in data['prep_pairs']:
            for obj, prep in pair.items():
                if obj != data['direct_object']:  # Don't repeat direct object
                    caption += ' ' + prep + ' ' + obj
        
        return caption
    
    def visualize_graph(self, row_idx: int = 0, 
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[str] = None):
        """
        Visualize the graph using matplotlib.
        
        Args:
            row_idx: Row index to visualize
            figsize: Figure size
            save_path: Optional path to save figure
        """
        graph = self.build_graph_structure(row_idx)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')
        
        # Define node positions
        positions = self._calculate_positions(graph['nodes'])
        
        # Color scheme
        colors = {
            'camera_wearer': {'fill': '#fff3cd', 'edge': '#ffc107'},
            'verb': {'fill': '#d4edda', 'edge': '#28a745'},
            'object': {'fill': '#f8d7da', 'edge': '#dc3545'}
        }
        
        # Draw edges first (so they appear below nodes)
        for edge in graph['edges']:
            from_pos = positions[edge['from']]
            to_pos = positions[edge['to']]
            
            # Create arrow
            arrow = FancyArrowPatch(
                from_pos, to_pos,
                arrowstyle='->,head_width=0.4,head_length=0.8',
                color='#495057',
                linewidth=2,
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add edge label
            if edge['label']:
                mid_x = (from_pos[0] + to_pos[0]) / 2
                mid_y = (from_pos[1] + to_pos[1]) / 2
                ax.text(mid_x + 0.1, mid_y + 0.1, edge['label'],
                       fontsize=10, style='italic', color='#495057',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.8))
        
        # Draw nodes
        for node in graph['nodes']:
            pos = positions[node['id']]
            color_scheme = colors[node['type']]
            
            # Draw node box
            box = FancyBboxPatch(
                (pos[0] - 0.4, pos[1] - 0.15),
                0.8, 0.3,
                boxstyle="round,pad=0.05",
                facecolor=color_scheme['fill'],
                edgecolor=color_scheme['edge'],
                linewidth=2.5,
                zorder=2
            )
            ax.add_patch(box)
            
            # Add node label
            ax.text(pos[0], pos[1], node['label'],
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   zorder=3)
        
        # Add title and caption
        plt.title(f"Action Scene Graph (Frame {graph['metadata']['frame_id']})",
                 fontsize=16, fontweight='bold', pad=20)
        
        caption = graph['metadata']['caption']
        plt.figtext(0.5, 0.02, caption,
                   ha='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=colors['camera_wearer']['fill'],
                          edgecolor=colors['camera_wearer']['edge'],
                          label='Camera Wearer', linewidth=2),
            mpatches.Patch(facecolor=colors['verb']['fill'],
                          edgecolor=colors['verb']['edge'],
                          label='Verb', linewidth=2),
            mpatches.Patch(facecolor=colors['object']['fill'],
                          edgecolor=colors['object']['edge'],
                          label='Object', linewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def _calculate_positions(self, nodes: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Calculate layout positions for nodes."""
        positions = {}
        
        # Find node types
        cw_node = next(n for n in nodes if n['type'] == 'camera_wearer')
        verb_node = next(n for n in nodes if n['type'] == 'verb')
        obj_nodes = [n for n in nodes if n['type'] == 'object']
        
        # Camera wearer at top center
        positions[cw_node['id']] = (2.5, 3.0)
        
        # Verb below camera wearer
        positions[verb_node['id']] = (2.5, 2.0)
        
        # Objects at bottom, spread horizontally
        num_objs = len(obj_nodes)
        if num_objs == 1:
            positions[obj_nodes[0]['id']] = (2.5, 0.8)
        else:
            spacing = 1.5
            start_x = 2.5 - (num_objs - 1) * spacing / 2
            for i, obj_node in enumerate(obj_nodes):
                positions[obj_node['id']] = (start_x + i * spacing, 0.8)
        
        return positions
    
    def export_to_pyg(self, row_idx: int = 0, 
                     global_features: Optional[np.ndarray] = None,
                     object_features: Optional[np.ndarray] = None):
        """
        Export graph to PyTorch Geometric format.
        
        Args:
            row_idx: Row index
            global_features: Global/clip features, shape (D,)
            object_features: Per-object features, shape (num_obj, D)
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric required: pip install torch-geometric")
        
        graph = self.build_graph_structure(row_idx)
        
        # Build node feature matrix
        obj_nodes = [n for n in graph['nodes'] if n['type'] == 'object']
        num_nodes = len(graph['nodes'])
        
        if object_features is not None:
            # Use provided object features
            feature_dim = object_features.shape[1]
            x = torch.zeros(num_nodes, feature_dim)
            
            # Assign object features
            for i, node in enumerate(obj_nodes):
                obj_idx = int(node['id'].split('_')[1])
                x[i + 2] = torch.from_numpy(object_features[obj_idx])
        else:
            # Use simple one-hot encoding
            x = torch.eye(num_nodes)
        
        # Build edge index
        edge_index = []
        for edge in graph['edges']:
            from_idx = next(i for i, n in enumerate(graph['nodes']) 
                          if n['id'] == edge['from'])
            to_idx = next(i for i, n in enumerate(graph['nodes']) 
                        if n['id'] == edge['to'])
            edge_index.append([from_idx, to_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Global features
        if global_features is not None:
            u = torch.from_numpy(global_features).unsqueeze(0)
        else:
            u = torch.zeros(1, 128)  # Placeholder
        
        data = Data(
            x=x,
            edge_index=edge_index,
            u=u,
            num_nodes=num_nodes
        )
        
        return data


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Sample CSV data
    csv_data = """frame_id,subject,verb,verb_type,direct_object,all_objects,phrasal_verb,preposition_object_pairs,pos_mask,tag_mask,dep_mask
0,The_camera_wearer,look,INTRANVERB,,"['fan', 'room']",look-up,"[{'fan': 'at'}, {'room': 'in'}]","['NOUN', 'AUX', 'VERB', 'ADP', 'ADP', 'DET', 'NOUN', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN']","['NN', 'VBZ', 'VBG', 'RP', 'IN', 'DT', 'NN', 'NN', 'IN', 'DT', 'JJ', 'NN']","['nsubj', 'aux', 'ROOT', 'prt', 'prep', 'det', 'compound', 'pobj', 'prep', 'det', 'amod', 'pobj']"
"""
    
    # Create builder
    builder = EASGGraphBuilder(csv_data=csv_data)
    
    # Build and print graph structure
    print("=" * 70)
    print("GRAPH STRUCTURE")
    print("=" * 70)
    
    graph = builder.build_graph_structure(row_idx=0)
    
    print(f"\nCaption: {graph['metadata']['caption']}")
    print(f"\nNodes ({len(graph['nodes'])}):")
    for node in graph['nodes']:
        print(f"  - {node['id']:10s} | {node['label']:15s} | {node['type']}")
    
    print(f"\nEdges ({len(graph['edges'])}):")
    for edge in graph['edges']:
        label = f"[{edge['label']}]" if edge['label'] else ""
        print(f"  - {edge['from']:10s} -> {edge['to']:10s} {label}")
    
    # Visualize
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    builder.visualize_graph(row_idx=1)
    
    # Export to PyTorch Geometric (if available)
    print("\n" + "=" * 70)
    print("PYTORCH GEOMETRIC EXPORT")
    print("=" * 70)
    try:
        # Simulate global and object features
        global_features = np.random.randn(2304)  # Clip features
        object_features = np.random.randn(2, 1024)  # 2 objects, 1024-dim features
        
        pyg_data = builder.export_to_pyg(row_idx=1, 
                                        global_features=global_features,
                                        object_features=object_features)
        print(f"PyG Data created:")
        print(f"  - Nodes: {pyg_data.num_nodes}")
        print(f"  - Edges: {pyg_data.edge_index.shape[1]}")
        print(f"  - Node features shape: {pyg_data.x.shape}")
        print(f"  - Global features shape: {pyg_data.u.shape}")
    except ImportError:
        print("PyTorch Geometric not available. Install to use this feature.")