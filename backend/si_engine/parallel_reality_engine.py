"""
ParallelRealityEngine - Quantum-inspired parallel simulation
Explores multiple reality branches simultaneously
"""

import uuid
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RealityBranch:
    """A single branch of reality/possibility"""
    id: str
    name: str
    assumptions: List[str]
    outcomes: List[str]
    probability_amplitude: float  # Complex amplitude (simplified to float)
    coherence_score: float
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'assumptions': self.assumptions,
            'outcomes': self.outcomes,
            'probability_amplitude': self.probability_amplitude,
            'coherence_score': self.coherence_score,
            'parent_id': self.parent_id
        }


@dataclass
class SimulationResult:
    """Result of a parallel reality simulation"""
    query: str
    branches: List[RealityBranch]
    collapsed_outcome: Optional[str]
    confidence: float
    total_probability: float
    dominant_branch_id: str
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'branches': [b.to_dict() for b in self.branches],
            'collapsed_outcome': self.collapsed_outcome,
            'confidence': self.confidence,
            'total_probability': self.total_probability,
            'dominant_branch_id': self.dominant_branch_id
        }


class ParallelRealityEngine:
    """
    Quantum-inspired reasoning engine
    Explores multiple possibilities simultaneously before collapsing to answer
    """
    
    def __init__(self, pattern_database, world_model=None):
        self.db = pattern_database
        self.world_model = world_model
        
        # Active branches
        self.branches: Dict[str, RealityBranch] = {}
        
        # History of simulations
        self.simulation_history: List[SimulationResult] = []
        
        # Configuration
        self.max_branches = 8
        self.coherence_threshold = 0.3
        self.collapse_threshold = 0.7
        
    def simulate_parallel_realities(
        self,
        query: str,
        initial_assumptions: Optional[List[str]] = None,
        num_branches: int = 4
    ) -> SimulationResult:
        """
        Simulate multiple parallel realities/possibilities
        """
        # Clear previous branches
        self.branches.clear()
        
        # Generate initial branches based on different assumptions
        branches = self._generate_branches(query, initial_assumptions, num_branches)
        
        # Evolve each branch
        for branch in branches:
            self._evolve_branch(branch, query)
            self.branches[branch.id] = branch
            
        # Calculate total probability
        total_prob = sum(b.probability_amplitude ** 2 for b in branches)
        
        # Find dominant branch
        dominant = max(branches, key=lambda b: b.probability_amplitude * b.coherence_score)
        
        # Attempt collapse if one branch is clearly dominant
        collapsed_outcome = None
        confidence = dominant.probability_amplitude * dominant.coherence_score
        
        if confidence > self.collapse_threshold:
            collapsed_outcome = self._collapse_to_outcome(dominant, query)
            
        result = SimulationResult(
            query=query,
            branches=branches,
            collapsed_outcome=collapsed_outcome,
            confidence=confidence,
            total_probability=total_prob,
            dominant_branch_id=dominant.id
        )
        
        self.simulation_history.append(result)
        return result
        
    def _generate_branches(
        self,
        query: str,
        initial_assumptions: Optional[List[str]],
        num_branches: int
    ) -> List[RealityBranch]:
        """Generate diverging reality branches"""
        branches = []
        
        # Get relevant patterns for context
        patterns = self.db.search(query, limit=5)
        
        # Generate different assumption sets
        assumption_sets = self._generate_assumption_sets(query, patterns, num_branches)
        
        for i, assumptions in enumerate(assumption_sets):
            branch = RealityBranch(
                id=str(uuid.uuid4()),
                name=f"Branch_{i+1}",
                assumptions=assumptions,
                outcomes=[],
                probability_amplitude=1.0 / math.sqrt(num_branches),  # Equal superposition
                coherence_score=1.0
            )
            branches.append(branch)
            
        return branches
        
    def _generate_assumption_sets(
        self,
        query: str,
        patterns: List[Any],
        num_sets: int
    ) -> List[List[str]]:
        """Generate different sets of assumptions"""
        assumption_sets = []
        
        # Extract key concepts from query
        query_words = set(query.lower().split())
        
        # Base assumptions from patterns
        base_assumptions = []
        for pattern in patterns[:3]:
            if hasattr(pattern, 'response'):
                # Extract key statements
                response = pattern.response
                sentences = response.split('.')[:2]
                for s in sentences:
                    if len(s.strip()) > 10:
                        base_assumptions.append(s.strip())
                        
        # Generate variations
        for i in range(num_sets):
            assumptions = []
            
            if i == 0:
                # Conservative: Use base assumptions
                assumptions = base_assumptions[:3] if base_assumptions else [f"Standard interpretation of: {query}"]
            elif i == 1:
                # Alternative: Question assumptions
                assumptions = [f"What if the opposite were true for: {query}"]
                if base_assumptions:
                    assumptions.append(f"Questioning: {base_assumptions[0][:50]}...")
            elif i == 2:
                # Edge case: Extreme interpretation
                assumptions = [f"Extreme interpretation: {query}"]
                assumptions.append("Assuming boundary conditions")
            else:
                # Random variation
                assumptions = [f"Alternative perspective {i} on: {query}"]
                if base_assumptions:
                    assumptions.extend(random.sample(base_assumptions, min(2, len(base_assumptions))))
                    
            assumption_sets.append(assumptions)
            
        return assumption_sets
        
    def _evolve_branch(self, branch: RealityBranch, query: str):
        """Evolve a branch forward through logical consequences"""
        # Use pattern matching to find likely outcomes
        patterns = self.db.search(query, limit=3)
        
        outcomes = []
        coherence_factors = []
        
        for assumption in branch.assumptions:
            # Find patterns related to assumption
            related = self.db.search(assumption, limit=2)
            
            for pattern in related:
                if hasattr(pattern, 'response'):
                    # Extract outcome-like statements
                    response = pattern.response
                    outcome = f"If {assumption[:30]}... then likely: {response[:100]}"
                    outcomes.append(outcome)
                    
                    # Coherence based on pattern confidence
                    coherence_factors.append(getattr(pattern, 'confidence', 0.5))
                    
        branch.outcomes = outcomes[:3]
        
        # Update coherence score
        if coherence_factors:
            branch.coherence_score = sum(coherence_factors) / len(coherence_factors)
        else:
            branch.coherence_score = 0.5
            
        # Adjust probability amplitude based on coherence
        branch.probability_amplitude *= branch.coherence_score
        
    def _collapse_to_outcome(self, branch: RealityBranch, query: str) -> str:
        """Collapse superposition to single outcome"""
        if branch.outcomes:
            # Weight by coherence and return best outcome
            return branch.outcomes[0]
        else:
            return f"Based on {branch.name}: Query '{query}' resolves to standard interpretation"
            
    def branch_reality(
        self,
        parent_branch_id: str,
        new_assumption: str
    ) -> Optional[RealityBranch]:
        """Create a new branch from existing branch with new assumption"""
        parent = self.branches.get(parent_branch_id)
        if not parent:
            return None
            
        # Check max branches
        if len(self.branches) >= self.max_branches:
            # Remove lowest probability branch
            lowest = min(self.branches.values(), key=lambda b: b.probability_amplitude)
            del self.branches[lowest.id]
            
        # Create child branch
        child = RealityBranch(
            id=str(uuid.uuid4()),
            name=f"{parent.name}_child",
            assumptions=parent.assumptions + [new_assumption],
            outcomes=[],
            probability_amplitude=parent.probability_amplitude / math.sqrt(2),
            coherence_score=parent.coherence_score * 0.9,  # Slight degradation
            parent_id=parent_branch_id
        )
        
        # Reduce parent amplitude (probability conservation)
        parent.probability_amplitude /= math.sqrt(2)
        
        self.branches[child.id] = child
        return child
        
    def merge_branches(
        self,
        branch_ids: List[str]
    ) -> Optional[RealityBranch]:
        """Merge compatible branches (interference)"""
        branches_to_merge = [self.branches[bid] for bid in branch_ids if bid in self.branches]
        
        if len(branches_to_merge) < 2:
            return None
            
        # Check coherence compatibility
        coherences = [b.coherence_score for b in branches_to_merge]
        if max(coherences) - min(coherences) > 0.5:
            # Too incoherent to merge
            return None
            
        # Create merged branch
        merged_assumptions = []
        merged_outcomes = []
        
        for b in branches_to_merge:
            merged_assumptions.extend(b.assumptions)
            merged_outcomes.extend(b.outcomes)
            
        # Constructive interference
        total_amplitude = sum(b.probability_amplitude for b in branches_to_merge)
        avg_coherence = sum(coherences) / len(coherences)
        
        merged = RealityBranch(
            id=str(uuid.uuid4()),
            name="Merged_Branch",
            assumptions=list(set(merged_assumptions))[:5],
            outcomes=list(set(merged_outcomes))[:5],
            probability_amplitude=total_amplitude,
            coherence_score=avg_coherence
        )
        
        # Remove original branches
        for bid in branch_ids:
            if bid in self.branches:
                del self.branches[bid]
                
        self.branches[merged.id] = merged
        return merged
        
    def measure_interference(
        self,
        branch1_id: str,
        branch2_id: str
    ) -> Dict[str, Any]:
        """Calculate interference between two branches"""
        b1 = self.branches.get(branch1_id)
        b2 = self.branches.get(branch2_id)
        
        if not b1 or not b2:
            return {'error': 'Branch not found'}
            
        # Calculate overlap in assumptions
        a1_set = set(a.lower() for a in b1.assumptions)
        a2_set = set(a.lower() for a in b2.assumptions)
        
        overlap = len(a1_set & a2_set) / max(len(a1_set | a2_set), 1)
        
        # Calculate phase difference (simplified)
        phase_diff = abs(b1.coherence_score - b2.coherence_score)
        
        # Interference type
        if overlap > 0.5 and phase_diff < 0.2:
            interference_type = 'constructive'
            combined_amplitude = b1.probability_amplitude + b2.probability_amplitude
        elif overlap > 0.5 and phase_diff > 0.5:
            interference_type = 'destructive'
            combined_amplitude = abs(b1.probability_amplitude - b2.probability_amplitude)
        else:
            interference_type = 'partial'
            combined_amplitude = math.sqrt(b1.probability_amplitude**2 + b2.probability_amplitude**2)
            
        return {
            'branch1': branch1_id,
            'branch2': branch2_id,
            'overlap': overlap,
            'phase_difference': phase_diff,
            'interference_type': interference_type,
            'combined_amplitude': combined_amplitude
        }
        
    def get_probability_distribution(self) -> Dict[str, float]:
        """Get current probability distribution across branches"""
        distribution = {}
        total = sum(b.probability_amplitude ** 2 for b in self.branches.values())
        
        for branch in self.branches.values():
            prob = (branch.probability_amplitude ** 2) / total if total > 0 else 0
            distribution[branch.id] = {
                'name': branch.name,
                'probability': prob,
                'coherence': branch.coherence_score
            }
            
        return distribution
        
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of all simulations"""
        return {
            'total_simulations': len(self.simulation_history),
            'active_branches': len(self.branches),
            'avg_confidence': sum(r.confidence for r in self.simulation_history) / len(self.simulation_history) if self.simulation_history else 0,
            'collapsed_count': sum(1 for r in self.simulation_history if r.collapsed_outcome),
            'branches': [b.to_dict() for b in self.branches.values()]
        }
