"""
Consciousness Controller - DSINN Integration for Image Generation
Tracks consciousness states during visual reasoning
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class ConsciousnessTrace:
    """A trace of consciousness during image generation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = "unknown"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    active_states: List[int] = field(default_factory=list)
    coherence_scores: List[float] = field(default_factory=list)
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'operation': self.operation,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': (self.end_time - self.start_time) * 1000 if self.end_time else None,
            'active_states': self.active_states,
            'coherence_scores': self.coherence_scores,
            'adjustments': self.adjustments
        }


@dataclass
class VisualReasoningState:
    """State of visual reasoning process"""
    current_phase: str = "idle"  # decomposition, matching, composition, rendering
    attention_focus: str = ""
    cognitive_load: float = 0.0
    coherence_level: float = 1.0
    active_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'current_phase': self.current_phase,
            'attention_focus': self.attention_focus,
            'cognitive_load': self.cognitive_load,
            'coherence_level': self.coherence_level,
            'active_strategies': self.active_strategies
        }


class ImageGenerationController:
    """
    Controls image generation with consciousness tracking
    Integrates with SI reasoning strategies
    
    Visual Reasoning States (0-23):
    - 0-5: Text decomposition states
    - 6-11: Pattern matching states  
    - 12-17: Scene composition states
    - 18-23: Rendering and refinement states
    """
    
    # State mappings for visual reasoning
    VISUAL_STATES = {
        'decomposition': list(range(0, 6)),
        'matching': list(range(6, 12)),
        'composition': list(range(12, 18)),
        'rendering': list(range(18, 24))
    }
    
    def __init__(
        self, 
        visual_pattern_db,
        text_decomposer,
        scene_composer,
        renderer,
        self_modeling_engine=None,
        reasoning_engine=None
    ):
        self.pattern_db = visual_pattern_db
        self.decomposer = text_decomposer
        self.composer = scene_composer
        self.renderer = renderer
        self.self_model = self_modeling_engine
        self.reasoning = reasoning_engine
        
        # Consciousness state
        self.state = VisualReasoningState()
        self.traces: List[ConsciousnessTrace] = []
        self.current_trace: Optional[ConsciousnessTrace] = None
        
        # Coherence thresholds
        self.min_coherence = 0.3
        self.max_iterations = 5
        
    @contextmanager
    def consciousness_trace(self, operation: str):
        """Context manager for tracking consciousness during operation"""
        trace = ConsciousnessTrace(operation=operation)
        self.current_trace = trace
        self.traces.append(trace)
        
        try:
            yield trace
        finally:
            trace.end_time = time.time()
            self.current_trace = None
    
    def activate_states(self, state_ids: List[int]):
        """Activate specific visual reasoning states"""
        self.state.active_strategies = [f"state_{s}" for s in state_ids]
        if self.current_trace:
            self.current_trace.active_states.extend(state_ids)
    
    def update_coherence(self, score: float):
        """Update coherence level"""
        self.state.coherence_level = score
        if self.current_trace:
            self.current_trace.coherence_scores.append(score)
    
    def is_coherent_state(self) -> bool:
        """Check if current state is coherent enough"""
        return self.state.coherence_level >= self.min_coherence
    
    def generate_image(
        self, 
        description: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main image generation pipeline with consciousness tracking
        
        Steps:
        1. Text decomposition (States 0-5)
        2. Pattern matching (States 6-11)
        3. Scene composition (States 12-17)
        4. Rendering with feedback (States 18-23)
        """
        context = context or {}
        start_time = time.time()
        
        with self.consciousness_trace("image_generation") as trace:
            result = {
                'description': description,
                'success': False,
                'consciousness_trace': None
            }
            
            try:
                # Phase 1: Text Decomposition
                self._update_phase("decomposition", description[:50])
                self.activate_states(self.VISUAL_STATES['decomposition'])
                
                decomposition = self.decomposer.decompose(description)
                decomp_coherence = self._evaluate_decomposition_coherence(decomposition)
                self.update_coherence(decomp_coherence)
                
                if not self.is_coherent_state():
                    # Adjust using self-modeling
                    decomposition = self._adjust_decomposition(decomposition)
                
                result['decomposition'] = decomposition.to_dict()
                
                # Phase 2: Pattern Matching
                self._update_phase("matching", "matching patterns")
                self.activate_states(self.VISUAL_STATES['matching'])
                
                matched_patterns = self.decomposer.match_concepts_to_patterns(
                    decomposition, 
                    self.pattern_db
                )
                
                match_coherence = self._evaluate_matching_coherence(matched_patterns)
                self.update_coherence(match_coherence)
                
                result['matched_patterns'] = len(matched_patterns)
                
                # Phase 3: Scene Composition
                self._update_phase("composition", "composing scene")
                self.activate_states(self.VISUAL_STATES['composition'])
                
                scene_graph = self.composer.compose_scene(matched_patterns, decomposition)
                
                composition_coherence = self._evaluate_composition_coherence(scene_graph)
                self.update_coherence(composition_coherence)
                
                # Iterative refinement if needed
                iterations = 0
                while not self.is_coherent_state() and iterations < self.max_iterations:
                    scene_graph = self._adjust_composition(scene_graph)
                    composition_coherence = self._evaluate_composition_coherence(scene_graph)
                    self.update_coherence(composition_coherence)
                    iterations += 1
                    trace.adjustments.append({
                        'iteration': iterations,
                        'coherence': composition_coherence
                    })
                
                result['scene_graph'] = scene_graph.to_dict()
                
                # Phase 4: Rendering
                self._update_phase("rendering", "rendering image")
                self.activate_states(self.VISUAL_STATES['rendering'])
                
                render_result = self.renderer.render_scene_graph(scene_graph)
                
                result['svg'] = render_result.get('svg')
                result['png_base64'] = render_result.get('png_base64')
                result['render_time_ms'] = render_result.get('render_time_ms')
                
                result['success'] = True
                result['total_time_ms'] = (time.time() - start_time) * 1000
                
            except Exception as e:
                result['error'] = str(e)
                result['success'] = False
            
            finally:
                result['consciousness_trace'] = trace.to_dict()
                result['final_state'] = self.state.to_dict()
                self._update_phase("idle", "")
        
        return result
    
    def _update_phase(self, phase: str, focus: str):
        """Update current phase and attention focus"""
        self.state.current_phase = phase
        self.state.attention_focus = focus
        
        # Estimate cognitive load based on phase
        load_map = {
            'idle': 0.0,
            'decomposition': 0.4,
            'matching': 0.5,
            'composition': 0.8,
            'rendering': 0.6
        }
        self.state.cognitive_load = load_map.get(phase, 0.5)
    
    def _evaluate_decomposition_coherence(self, decomposition) -> float:
        """Evaluate coherence of text decomposition"""
        score = 1.0
        
        # Check if we found any concepts
        if not decomposition.concepts:
            score -= 0.5
        
        # Check if concepts have attributes
        concepts_with_attrs = sum(1 for c in decomposition.concepts if c.attributes)
        if decomposition.concepts:
            score -= 0.2 * (1 - concepts_with_attrs / len(decomposition.concepts))
        
        return max(0.0, score)
    
    def _evaluate_matching_coherence(self, matched_patterns: List[Dict]) -> float:
        """Evaluate coherence of pattern matching"""
        if not matched_patterns:
            return 0.3
        
        # Check how many patterns were successfully matched
        matched_count = sum(1 for m in matched_patterns if m.get('pattern_id'))
        match_ratio = matched_count / len(matched_patterns)
        
        return 0.3 + 0.7 * match_ratio
    
    def _evaluate_composition_coherence(self, scene_graph) -> float:
        """Evaluate coherence of scene composition"""
        score = 0.8  # Base score for having a scene graph
        
        # Count nodes
        all_nodes = self._count_nodes(scene_graph)
        
        # Penalize empty scenes
        if all_nodes == 0:
            score -= 0.4
        
        # Check layer distribution
        # (Would ideally check for overlapping elements, etc.)
        
        return max(0.0, min(1.0, score))
    
    def _count_nodes(self, node) -> int:
        """Count total nodes in scene graph"""
        count = 1 if node.pattern_id else 0
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _adjust_decomposition(self, decomposition):
        """Adjust decomposition using self-modeling"""
        # For now, return as-is
        # Would integrate with SelfModelingEngine for real adjustments
        return decomposition
    
    def _adjust_composition(self, scene_graph):
        """Adjust composition using self-modeling"""
        # Would make adjustments based on self-model feedback
        # For now, just return the original
        return scene_graph
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about image generation"""
        if not self.traces:
            return {'total_generations': 0}
        
        durations = [
            (t.end_time - t.start_time) * 1000 
            for t in self.traces 
            if t.end_time
        ]
        
        coherence_scores = []
        for trace in self.traces:
            coherence_scores.extend(trace.coherence_scores)
        
        return {
            'total_generations': len(self.traces),
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'avg_coherence': sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0,
            'current_state': self.state.to_dict()
        }
