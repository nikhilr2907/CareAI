from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

from .node import HospitalNode
from .edge import HospitalEdge

class HospitalGraph:
    def __init__(self):
        # Adjacency list using the objects we've brainstormed
        self.nodes: Dict[str, HospitalNode] = {}
        self.edges: Dict[str, List[HospitalEdge]] = {}

    def add_node(self, node: HospitalNode):
        self.nodes[node.node_id] = node
        self.edges[node.node_id] = []

    def add_edge(self, edge: HospitalEdge):
        self.edges[edge.from_node].append(edge)

    def get_detailed_state(self) -> np.ndarray:
        """
        Converts the graph's rich metadata into a single state-mappable 
        object (tensor/array) for a Deep Learning model.
        """
        state_vector = []
        for node_id, node in self.nodes.items():
            # Flatten node metadata
            state_vector.extend([
                node.x, node.y, 
                float(node.is_cluttered),  # Cues like nursing terminals [cite: 177]
                node.current_occupancy / node.max_capacity,
                node.max_reach_height  # Robot spec limit: 1350mm [cite: 570]
            ])
            
            # Flatten edge metadata for edges originating here
            for edge in self.edges[node_id]:
                state_vector.extend([
                    edge.current_weight,  # Dynamic weight property
                    float(edge.has_patient_bed),  # Width constraint: 2150-3330mm [cite: 436, 438]
                    edge.clutter_level  # From "Tetris-ed" trolleys 
                ])
        
        return np.array(state_vector, dtype=np.float32)