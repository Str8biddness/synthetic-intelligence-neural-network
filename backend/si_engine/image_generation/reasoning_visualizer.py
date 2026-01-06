"""
Reasoning Visualizer - Shows pattern matching process visually
Displays semantic tag matching, confidence scores, and pattern selection flow
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    TEXT_ANALYSIS = "text_analysis"
    SEMANTIC_EXTRACTION = "semantic_extraction"
    PATTERN_SEARCH = "pattern_search"
    PATTERN_MATCH = "pattern_match"
    CONFIDENCE_CALC = "confidence_calc"
    COMPOSITION_PLANNING = "composition_planning"
    LAYER_ASSIGNMENT = "layer_assignment"
    RENDERING = "rendering"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_type: ReasoningStepType = ReasoningStepType.TEXT_ANALYSIS
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    duration_ms: float = 0.0
    children: List['ReasoningStep'] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'step_type': self.step_type.value,
            'description': self.description,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'confidence': self.confidence,
            'duration_ms': self.duration_ms,
            'children': [c.to_dict() for c in self.children]
        }


@dataclass
class PatternMatchResult:
    """Result of pattern matching"""
    pattern_id: str
    pattern_name: str
    matched_tags: List[str]
    score: float
    category: str
    abstraction_level: int
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'matched_tags': self.matched_tags,
            'score': self.score,
            'category': self.category,
            'abstraction_level': self.abstraction_level
        }


@dataclass
class ReasoningVisualization:
    """Complete reasoning visualization data"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_text: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    pattern_matches: List[PatternMatchResult] = field(default_factory=list)
    selected_patterns: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    final_confidence: float = 0.0
    svg_visualization: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'input_text': self.input_text,
            'steps': [s.to_dict() for s in self.steps],
            'pattern_matches': [m.to_dict() for m in self.pattern_matches],
            'selected_patterns': self.selected_patterns,
            'total_duration_ms': self.total_duration_ms,
            'final_confidence': self.final_confidence,
            'svg_visualization': self.svg_visualization
        }


class ReasoningVisualizer:
    """
    Visualizes the pattern matching and reasoning process
    Creates step-by-step visualization of how patterns are selected
    """
    
    def __init__(self, pattern_db):
        self.pattern_db = pattern_db
        self.current_session: Optional[ReasoningVisualization] = None
    
    def start_session(self, input_text: str) -> ReasoningVisualization:
        """Start a new reasoning visualization session"""
        self.current_session = ReasoningVisualization(input_text=input_text)
        return self.current_session
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the current session"""
        if self.current_session:
            self.current_session.steps.append(step)
    
    def record_text_analysis(self, text: str, extracted_concepts: List[str], 
                             keywords: List[str], duration_ms: float) -> ReasoningStep:
        """Record text analysis step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.TEXT_ANALYSIS,
            description=f"Analyzing input text: '{text[:50]}...'",
            input_data={'text': text},
            output_data={
                'extracted_concepts': extracted_concepts,
                'keywords': keywords,
                'word_count': len(text.split())
            },
            confidence=0.9 if keywords else 0.5,
            duration_ms=duration_ms
        )
        self.add_step(step)
        return step
    
    def record_semantic_extraction(self, concepts: List[str], 
                                   semantic_tags: List[str], 
                                   duration_ms: float) -> ReasoningStep:
        """Record semantic extraction step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.SEMANTIC_EXTRACTION,
            description=f"Extracting semantic meaning from {len(concepts)} concepts",
            input_data={'concepts': concepts},
            output_data={
                'semantic_tags': semantic_tags,
                'tag_count': len(semantic_tags)
            },
            confidence=min(1.0, len(semantic_tags) / 5),
            duration_ms=duration_ms
        )
        self.add_step(step)
        return step
    
    def record_pattern_search(self, tags: List[str], 
                              candidates_found: int,
                              duration_ms: float) -> ReasoningStep:
        """Record pattern search step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.PATTERN_SEARCH,
            description=f"Searching patterns with {len(tags)} semantic tags",
            input_data={'search_tags': tags},
            output_data={
                'candidates_found': candidates_found,
                'search_scope': len(self.pattern_db.patterns) if self.pattern_db else 0
            },
            confidence=min(1.0, candidates_found / 10) if candidates_found > 0 else 0.1,
            duration_ms=duration_ms
        )
        self.add_step(step)
        return step
    
    def record_pattern_match(self, pattern_id: str, pattern_name: str,
                             matched_tags: List[str], score: float,
                             category: str, level: int) -> PatternMatchResult:
        """Record a pattern match result"""
        result = PatternMatchResult(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            matched_tags=matched_tags,
            score=score,
            category=category,
            abstraction_level=level
        )
        if self.current_session:
            self.current_session.pattern_matches.append(result)
        return result
    
    def record_composition_planning(self, selected_patterns: List[str],
                                    layout_strategy: str,
                                    duration_ms: float) -> ReasoningStep:
        """Record composition planning step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.COMPOSITION_PLANNING,
            description=f"Planning composition with {len(selected_patterns)} patterns",
            input_data={'candidate_patterns': selected_patterns},
            output_data={
                'layout_strategy': layout_strategy,
                'selected_count': len(selected_patterns)
            },
            confidence=0.8 if selected_patterns else 0.3,
            duration_ms=duration_ms
        )
        self.add_step(step)
        if self.current_session:
            self.current_session.selected_patterns = selected_patterns
        return step
    
    def record_layer_assignment(self, layers: Dict[str, List[str]],
                                duration_ms: float) -> ReasoningStep:
        """Record layer assignment step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.LAYER_ASSIGNMENT,
            description=f"Assigning patterns to {len(layers)} depth layers",
            input_data={'layer_names': list(layers.keys())},
            output_data={
                'layer_assignments': {k: len(v) for k, v in layers.items()},
                'total_elements': sum(len(v) for v in layers.values())
            },
            confidence=0.85,
            duration_ms=duration_ms
        )
        self.add_step(step)
        return step
    
    def finalize_session(self, total_duration_ms: float, 
                        final_confidence: float) -> ReasoningVisualization:
        """Finalize the session and generate visualization"""
        if not self.current_session:
            return ReasoningVisualization()
        
        self.current_session.total_duration_ms = total_duration_ms
        self.current_session.final_confidence = final_confidence
        self.current_session.svg_visualization = self._generate_svg_visualization()
        
        return self.current_session
    
    def _generate_svg_visualization(self) -> str:
        """Generate SVG visualization of reasoning process"""
        if not self.current_session:
            return ""
        
        width, height = 600, 400
        svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
        
        # Background
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>')
        
        # Title
        svg_parts.append(f'<text x="{width//2}" y="25" fill="#00ff88" font-family="monospace" font-size="14" text-anchor="middle">REASONING VISUALIZATION</text>')
        
        # Steps visualization
        steps = self.current_session.steps
        if steps:
            step_width = (width - 40) // max(len(steps), 1)
            for i, step in enumerate(steps):
                x = 20 + i * step_width
                y = 60
                
                # Step box
                color = self._get_step_color(step.step_type)
                svg_parts.append(f'<rect x="{x}" y="{y}" width="{step_width-5}" height="60" fill="{color}" opacity="0.3" rx="5"/>')
                svg_parts.append(f'<rect x="{x}" y="{y}" width="{step_width-5}" height="60" fill="none" stroke="{color}" stroke-width="1" rx="5"/>')
                
                # Step label
                label = step.step_type.value[:12]
                svg_parts.append(f'<text x="{x + step_width//2 - 2}" y="{y + 25}" fill="#ffffff" font-family="monospace" font-size="8" text-anchor="middle">{label}</text>')
                
                # Confidence bar
                bar_width = (step_width - 15) * step.confidence
                svg_parts.append(f'<rect x="{x + 5}" y="{y + 40}" width="{bar_width}" height="8" fill="{color}" rx="2"/>')
                svg_parts.append(f'<text x="{x + step_width//2 - 2}" y="{y + 55}" fill="#888888" font-family="monospace" font-size="7" text-anchor="middle">{step.confidence:.0%}</text>')
                
                # Arrow to next step
                if i < len(steps) - 1:
                    svg_parts.append(f'<line x1="{x + step_width - 5}" y1="{y + 30}" x2="{x + step_width + 2}" y2="{y + 30}" stroke="#00ff88" stroke-width="1" marker-end="url(#arrowhead)"/>')
        
        # Pattern matches visualization
        matches = self.current_session.pattern_matches[:8]  # Show top 8
        if matches:
            svg_parts.append(f'<text x="20" y="150" fill="#00ff88" font-family="monospace" font-size="10">PATTERN MATCHES:</text>')
            for i, match in enumerate(matches):
                y = 165 + i * 25
                # Score bar
                bar_width = 100 * match.score
                svg_parts.append(f'<rect x="20" y="{y}" width="{bar_width}" height="18" fill="#3b82f6" opacity="0.5" rx="3"/>')
                svg_parts.append(f'<text x="25" y="{y + 13}" fill="#ffffff" font-family="monospace" font-size="9">{match.pattern_name[:20]}</text>')
                svg_parts.append(f'<text x="130" y="{y + 13}" fill="#888888" font-family="monospace" font-size="8">{match.score:.2f}</text>')
                # Tags
                tags_text = ", ".join(match.matched_tags[:3])
                svg_parts.append(f'<text x="170" y="{y + 13}" fill="#666666" font-family="monospace" font-size="7">{tags_text[:30]}</text>')
        
        # Statistics
        stats_y = 360
        total_ms = self.current_session.total_duration_ms
        confidence = self.current_session.final_confidence
        svg_parts.append(f'<text x="20" y="{stats_y}" fill="#00ff88" font-family="monospace" font-size="10">Total: {total_ms:.1f}ms | Confidence: {confidence:.0%} | Patterns: {len(matches)}</text>')
        
        # Arrow marker definition
        svg_parts.insert(1, '<defs><marker id="arrowhead" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><polygon points="0 0, 6 3, 0 6" fill="#00ff88"/></marker></defs>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _get_step_color(self, step_type: ReasoningStepType) -> str:
        """Get color for step type"""
        colors = {
            ReasoningStepType.TEXT_ANALYSIS: '#f59e0b',
            ReasoningStepType.SEMANTIC_EXTRACTION: '#8b5cf6',
            ReasoningStepType.PATTERN_SEARCH: '#3b82f6',
            ReasoningStepType.PATTERN_MATCH: '#10b981',
            ReasoningStepType.CONFIDENCE_CALC: '#ef4444',
            ReasoningStepType.COMPOSITION_PLANNING: '#ec4899',
            ReasoningStepType.LAYER_ASSIGNMENT: '#06b6d4',
            ReasoningStepType.RENDERING: '#84cc16'
        }
        return colors.get(step_type, '#6b7280')
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get a summary of the reasoning process"""
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'input_text': self.current_session.input_text[:100],
            'step_count': len(self.current_session.steps),
            'pattern_matches_count': len(self.current_session.pattern_matches),
            'selected_patterns_count': len(self.current_session.selected_patterns),
            'total_duration_ms': self.current_session.total_duration_ms,
            'final_confidence': self.current_session.final_confidence,
            'steps_summary': [
                {
                    'type': s.step_type.value,
                    'confidence': s.confidence,
                    'duration_ms': s.duration_ms
                } for s in self.current_session.steps
            ],
            'top_patterns': [
                {
                    'name': m.pattern_name,
                    'score': m.score,
                    'tags': m.matched_tags[:5]
                } for m in sorted(
                    self.current_session.pattern_matches,
                    key=lambda x: x.score,
                    reverse=True
                )[:5]
            ]
        }
