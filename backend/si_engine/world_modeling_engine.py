"""
WorldModelingEngine - Causal reasoning and counterfactual analysis
Maintains a causal graph for reasoning about cause and effect
"""

import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CausalNode:
    """A node in the causal graph"""
    id: str
    name: str
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'domain': self.domain,
            'properties': self.properties,
            'confidence': self.confidence
        }


@dataclass
class CausalEdge:
    """An edge representing causal relationship"""
    source_id: str
    target_id: str
    mechanism: str
    strength: float  # 0-1, how strongly source causes target
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'mechanism': self.mechanism,
            'strength': self.strength,
            'evidence': self.evidence
        }


@dataclass
class CausalMechanism:
    """A known causal mechanism"""
    id: str
    name: str
    description: str
    domain: str
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'domain': self.domain,
            'examples': self.examples
        }


class WorldModelingEngine:
    """
    Causal reasoning engine with world model
    Supports counterfactual reasoning and intervention analysis
    """
    
    def __init__(self, pattern_database):
        self.db = pattern_database
        
        # Causal graph
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # outgoing edges
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # incoming edges
        
        # Mechanism library
        self.mechanisms: Dict[str, CausalMechanism] = {}
        
        # Initialize with common causal mechanisms
        self._init_mechanisms()
        self._build_causal_graph()
        
    def _init_mechanisms(self):
        """Initialize library of causal mechanisms"""
        mechanisms_data = [
            {
                'name': 'physical_force',
                'description': 'Direct physical causation through force, energy, or matter transfer',
                'domain': 'physics',
                'examples': ['collision causes movement', 'heat causes expansion']
            },
            {
                'name': 'biological_process',
                'description': 'Causation through biological/chemical processes',
                'domain': 'biology',
                'examples': ['DNA replication', 'enzyme catalysis', 'neural transmission']
            },
            {
                'name': 'information_transfer',
                'description': 'Causation through information or signal transmission',
                'domain': 'general',
                'examples': ['message causes response', 'learning changes behavior']
            },
            {
                'name': 'economic_incentive',
                'description': 'Causation through economic incentives and markets',
                'domain': 'economics',
                'examples': ['price change affects demand', 'incentive changes behavior']
            },
            {
                'name': 'psychological_influence',
                'description': 'Causation through psychological mechanisms',
                'domain': 'psychology',
                'examples': ['belief causes action', 'emotion influences decision']
            },
            {
                'name': 'logical_entailment',
                'description': 'Necessary logical or mathematical consequence',
                'domain': 'logic',
                'examples': ['premises entail conclusion', 'axioms determine theorems']
            },
            {
                'name': 'social_causation',
                'description': 'Causation through social structures and norms',
                'domain': 'sociology',
                'examples': ['norm shapes behavior', 'institution constrains action']
            },
            {
                'name': 'evolutionary_selection',
                'description': 'Causation through evolutionary processes',
                'domain': 'biology',
                'examples': ['selection pressure shapes traits', 'adaptation to environment']
            }
        ]
        
        for mech_data in mechanisms_data:
            mech_id = str(uuid.uuid4())
            self.mechanisms[mech_id] = CausalMechanism(
                id=mech_id,
                **mech_data
            )
            
    def _build_causal_graph(self):
        """Build initial causal graph from pattern database"""
        # Extract causal relationships from patterns
        patterns = self.db.get_all_patterns()
        
        for pattern in patterns:
            # Extract concepts from pattern
            concepts = self._extract_concepts(pattern.response)
            
            # Create nodes for main concepts
            for concept in concepts[:3]:
                if concept not in [n.name for n in self.nodes.values()]:
                    node_id = str(uuid.uuid4())
                    self.nodes[node_id] = CausalNode(
                        id=node_id,
                        name=concept,
                        domain=pattern.domain
                    )
                    
            # Look for causal language to create edges
            causal_relations = self._extract_causal_relations(pattern.response)
            for cause, effect, mechanism in causal_relations:
                self._add_causal_edge(cause, effect, mechanism)
                
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        import re
        # Simple extraction - get noun phrases
        words = re.findall(r'\b[A-Za-z][a-z]+\b', text)
        # Filter to significant words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'and', 'but', 'or', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by'}
        concepts = [w.lower() for w in words if w.lower() not in stop_words and len(w) > 3]
        return list(dict.fromkeys(concepts))[:10]  # Unique, preserve order
        
    def _extract_causal_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract causal relationships from text"""
        relations = []
        text_lower = text.lower()
        
        # Causal indicators
        causal_patterns = [
            (r'(\w+)\s+causes?\s+(\w+)', 'direct'),
            (r'(\w+)\s+leads?\s+to\s+(\w+)', 'direct'),
            (r'(\w+)\s+results?\s+in\s+(\w+)', 'direct'),
            (r'because\s+(?:of\s+)?(\w+).*?(\w+)', 'explanation'),
            (r'(\w+)\s+produces?\s+(\w+)', 'production'),
            (r'(\w+)\s+enables?\s+(\w+)', 'enabling'),
        ]
        
        import re
        for pattern, mechanism in causal_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) >= 2:
                    cause, effect = match[0], match[1]
                    if len(cause) > 2 and len(effect) > 2:
                        relations.append((cause, effect, mechanism))
                        
        return relations[:5]  # Limit to avoid noise
        
    def _add_causal_edge(self, cause: str, effect: str, mechanism: str):
        """Add a causal edge to the graph"""
        # Find or create nodes
        cause_node = self._get_or_create_node(cause)
        effect_node = self._get_or_create_node(effect)
        
        # Check if edge already exists
        for edge in self.edges:
            if edge.source_id == cause_node.id and edge.target_id == effect_node.id:
                # Strengthen existing edge
                edge.strength = min(1.0, edge.strength + 0.1)
                return
                
        # Create new edge
        edge = CausalEdge(
            source_id=cause_node.id,
            target_id=effect_node.id,
            mechanism=mechanism,
            strength=0.5
        )
        self.edges.append(edge)
        self.adjacency[cause_node.id].append(effect_node.id)
        self.reverse_adjacency[effect_node.id].append(cause_node.id)
        
    def _get_or_create_node(self, name: str) -> CausalNode:
        """Get existing node or create new one"""
        for node in self.nodes.values():
            if node.name == name:
                return node
                
        node_id = str(uuid.uuid4())
        node = CausalNode(id=node_id, name=name, domain='general')
        self.nodes[node_id] = node
        return node
        
    def reason_causally(self, query: str) -> Dict[str, Any]:
        """
        Perform causal reasoning about a query
        """
        # Extract concepts from query
        concepts = self._extract_concepts(query)
        
        # Find relevant nodes
        relevant_nodes = []
        for concept in concepts:
            for node in self.nodes.values():
                if concept in node.name or node.name in concept:
                    relevant_nodes.append(node)
                    
        if not relevant_nodes:
            return {
                'success': False,
                'message': 'No causal knowledge found for query concepts',
                'concepts': concepts
            }
            
        # Build causal chain
        causal_chain = self._build_causal_chain(relevant_nodes)
        
        # Identify mechanisms
        mechanisms_used = self._identify_mechanisms(causal_chain)
        
        return {
            'success': True,
            'query': query,
            'concepts': concepts,
            'causal_chain': causal_chain,
            'mechanisms': mechanisms_used,
            'explanation': self._generate_causal_explanation(causal_chain, mechanisms_used)
        }
        
    def _build_causal_chain(self, nodes: List[CausalNode]) -> List[Dict]:
        """Build causal chain from nodes"""
        chain = []
        visited = set()
        
        for node in nodes:
            if node.id in visited:
                continue
                
            # Get causes (incoming edges)
            causes = self.reverse_adjacency.get(node.id, [])
            # Get effects (outgoing edges)
            effects = self.adjacency.get(node.id, [])
            
            chain.append({
                'node': node.to_dict(),
                'causes': [self.nodes[c].name for c in causes if c in self.nodes],
                'effects': [self.nodes[e].name for e in effects if e in self.nodes]
            })
            visited.add(node.id)
            
        return chain
        
    def _identify_mechanisms(self, causal_chain: List[Dict]) -> List[Dict]:
        """Identify mechanisms involved in causal chain"""
        mechanisms = []
        seen = set()
        
        for item in causal_chain:
            node_id = item['node']['id']
            
            # Find edges involving this node
            for edge in self.edges:
                if edge.source_id == node_id or edge.target_id == node_id:
                    if edge.mechanism not in seen:
                        mechanisms.append({
                            'type': edge.mechanism,
                            'strength': edge.strength
                        })
                        seen.add(edge.mechanism)
                        
        return mechanisms
        
    def _generate_causal_explanation(
        self, 
        chain: List[Dict], 
        mechanisms: List[Dict]
    ) -> str:
        """Generate natural language causal explanation"""
        if not chain:
            return "No causal relationships identified."
            
        explanation_parts = []
        
        for item in chain:
            node_name = item['node']['name']
            causes = item['causes']
            effects = item['effects']
            
            if causes:
                explanation_parts.append(f"{node_name} is caused by {', '.join(causes)}")
            if effects:
                explanation_parts.append(f"{node_name} leads to {', '.join(effects)}")
                
        if mechanisms:
            mech_names = [m['type'] for m in mechanisms]
            explanation_parts.append(f"Mechanisms involved: {', '.join(mech_names)}")
            
        return ". ".join(explanation_parts) + "."
        
    def counterfactual_reasoning(
        self, 
        intervention: str, 
        target: str
    ) -> Dict[str, Any]:
        """
        Counterfactual reasoning: What if intervention happened?
        """
        intervention_node = None
        target_node = None
        
        # Find nodes
        for node in self.nodes.values():
            if intervention.lower() in node.name.lower():
                intervention_node = node
            if target.lower() in node.name.lower():
                target_node = node
                
        if not intervention_node or not target_node:
            return {
                'success': False,
                'message': f"Could not find nodes for '{intervention}' or '{target}'"
            }
            
        # Find path from intervention to target
        path = self._find_causal_path(intervention_node.id, target_node.id)
        
        if not path:
            return {
                'success': True,
                'intervention': intervention,
                'target': target,
                'effect': 'no_direct_effect',
                'explanation': f"No causal path found from {intervention} to {target}"
            }
            
        # Calculate effect strength
        effect_strength = self._calculate_path_strength(path)
        
        return {
            'success': True,
            'intervention': intervention,
            'target': target,
            'effect': 'potential_effect',
            'strength': effect_strength,
            'path': [self.nodes[n].name for n in path if n in self.nodes],
            'explanation': f"Intervening on {intervention} would affect {target} through path: {' → '.join([self.nodes[n].name for n in path if n in self.nodes])}"
        }
        
    def _find_causal_path(self, source_id: str, target_id: str) -> List[str]:
        """Find path from source to target in causal graph"""
        if source_id == target_id:
            return [source_id]
            
        visited = set()
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
                    
        return []
        
    def _calculate_path_strength(self, path: List[str]) -> float:
        """Calculate combined strength of causal path"""
        if len(path) < 2:
            return 0.0
            
        strength = 1.0
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            for edge in self.edges:
                if edge.source_id == source and edge.target_id == target:
                    strength *= edge.strength
                    break
                    
        return strength
        
    def simulate_intervention(
        self, 
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate the effects of an intervention
        """
        node_name = intervention.get('node')
        change = intervention.get('change', 'increase')
        
        # Find the node
        target_node = None
        for node in self.nodes.values():
            if node_name.lower() in node.name.lower():
                target_node = node
                break
                
        if not target_node:
            return {
                'success': False,
                'message': f"Node '{node_name}' not found in causal graph"
            }
            
        # Find downstream effects
        effects = []
        visited = set()
        queue = [target_node.id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for downstream in self.adjacency.get(current, []):
                if downstream in self.nodes:
                    downstream_node = self.nodes[downstream]
                    effects.append({
                        'node': downstream_node.name,
                        'expected_change': change,
                        'distance': len(visited)
                    })
                    queue.append(downstream)
                    
        return {
            'success': True,
            'intervention': intervention,
            'direct_effects': len(effects),
            'effects': effects[:10],  # Limit output
            'summary': f"Intervening on '{node_name}' would affect {len(effects)} downstream nodes"
        }
        
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the causal graph"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'mechanisms': len(self.mechanisms),
            'domains': list(set(n.domain for n in self.nodes.values())),
            'avg_connections': len(self.edges) / max(len(self.nodes), 1)
        }
