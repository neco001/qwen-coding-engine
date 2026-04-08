"""Knowledge Graph Lite - Decision Dependency Tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph (an ADR decision)."""
    
    node_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents a dependency edge between two ADR nodes."""
    
    source: str  # The ADR that depends on another
    target: str  # The ADR being depended upon
    relationship: str = "depends_on"


class KnowledgeGraph:
    """
    Lightweight knowledge graph for tracking ADR dependencies.
    
    Provides:
    - Node management (add/query ADR decisions)
    - Edge management (track dependencies between ADRs)
    - Cycle detection (identify circular dependencies)
    - Export capabilities (JSON, parquet-compatible format)
    """
    
    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], str] = {}
    
    def add_node(self, node_id: str, data: dict[str, Any]) -> None:
        """
        Add a node (ADR decision) to the graph.
        
        Args:
            node_id: Unique identifier for the ADR (e.g., "ADR-001")
            data: ADR metadata (title, status, timestamp, etc.)
        """
        self.nodes[node_id] = data
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all associated edges from the graph.
        
        Args:
            node_id: The node ID to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
        
        # Remove all edges involving this node
        edges_to_remove = [
            edge for edge in self.edges 
            if edge[0] == node_id or edge[1] == node_id
        ]
        for edge in edges_to_remove:
            del self.edges[edge]
    
    def add_edge(self, source: str, target: str, relationship: str = "depends_on") -> None:
        """
        Add a dependency edge between two nodes.
        
        Args:
            source: The ADR that depends on another (dependent)
            target: The ADR being depended upon (dependency)
            relationship: Type of relationship (default: "depends_on")
        
        Raises:
            ValueError: If source or target node doesn't exist
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' does not exist")
        
        self.edges[(source, target)] = relationship
    
    def remove_edge(self, source: str, target: str) -> None:
        """
        Remove an edge between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
        """
        if (source, target) in self.edges:
            del self.edges[(source, target)]
    
    def get_dependencies(self, node_id: str) -> list[str]:
        """
        Get all nodes that a given node depends on.
        
        Args:
            node_id: The node to query
        
        Returns:
            List of node IDs that this node depends on
        """
        dependencies = []
        for (source, target), _ in self.edges.items():
            if source == node_id:
                dependencies.append(target)
        return dependencies
    
    def get_dependents(self, node_id: str) -> list[str]:
        """
        Get all nodes that depend on a given node.
        
        Args:
            node_id: The node to query
        
        Returns:
            List of node IDs that depend on this node
        """
        dependents = []
        for (source, target), _ in self.edges.items():
            if target == node_id:
                dependents.append(source)
        return dependents
    
    def detect_cycles(self) -> list[list[str]]:
        """
        Detect all circular dependencies in the graph.
        
        Uses DFS-based cycle detection algorithm.
        
        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []
        
        def dfs(node: str) -> bool:
            """DFS traversal to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Get all dependencies of this node
            for (source, target), _ in self.edges.items():
                if source == node:
                    if target not in visited:
                        if dfs(target):
                            return True
                    elif target in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(target)
                        cycle = path[cycle_start:] + [target]
                        cycles.append(cycle)
                        return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Run DFS from each unvisited node
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def has_cycles(self) -> bool:
        """
        Check if the graph contains any circular dependencies.
        
        Returns:
            True if cycles exist, False otherwise
        """
        return len(self.detect_cycles()) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """
        Export the graph to a dictionary (JSON-compatible) format.
        
        Returns:
            Dictionary with 'nodes' and 'edges' keys
        """
        nodes_list = [
            {"id": node_id, **data} 
            for node_id, data in self.nodes.items()
        ]
        
        edges_list = [
            {"source": source, "target": target, "relationship": relationship}
            for (source, target), relationship in self.edges.items()
        ]
        
        return {
            "nodes": nodes_list,
            "edges": edges_list
        }
    
    def to_records(self) -> list[dict[str, Any]]:
        """
        Export graph data in parquet-compatible record format.
        
        Returns:
            List of dictionaries suitable for parquet table creation
        """
        records: list[dict[str, Any]] = []
        
        # Add node records
        for node_id, data in self.nodes.items():
            record = {
                "record_type": "node",
                "node_id": node_id,
                "title": data.get("title", ""),
                "status": data.get("status", ""),
                "timestamp": data.get("timestamp", ""),
                "data": str(data),  # Serialize full data as string
            }
            records.append(record)
        
        # Add edge records
        for (source, target), relationship in self.edges.items():
            record = {
                "record_type": "edge",
                "source_node": source,
                "target_node": target,
                "relationship": relationship,
            }
            records.append(record)
        
        return records
    
    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """
        Get a node's data by ID.
        
        Args:
            node_id: The node ID to retrieve
        
        Returns:
            Node data dictionary or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> list[str]:
        """
        Get all node IDs in the graph.
        
        Returns:
            List of all node IDs
        """
        return list(self.nodes.keys())
    
    def get_all_edges(self) -> list[tuple[str, str, str]]:
        """
        Get all edges in the graph.
        
        Returns:
            List of tuples (source, target, relationship)
        """
        return [(source, target, rel) for (source, target), rel in self.edges.items()]
    
    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        return f"KnowledgeGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
