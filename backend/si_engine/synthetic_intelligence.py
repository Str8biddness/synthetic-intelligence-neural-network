"""
SyntheticIntelligence - Main SI Engine Orchestrator
Coordinates all SI components for query processing

Now with FAISS-based ScalablePatternDatabase for high-performance pattern matching,
WebSearchModule for real-time web search, and DailyPatternUpdater for knowledge updates.
"""

import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .pattern_database import PatternDatabase
from .scalable_pattern_db import ScalablePatternDatabase, Pattern as ScalablePattern
from .pattern_matcher import PatternMatcher
from .quasi_reasoning_engine import QuasiReasoningEngine, CognitiveStrategy
from .synthetic_language_generator import SyntheticLanguageGenerator
from .emergent_reasoning import EmergentReasoning
from .world_modeling_engine import WorldModelingEngine
from .self_modeling_engine import SelfModelingEngine
from .entity_knowledge_base import EntityKnowledgeBase
from .parallel_reality_engine import ParallelRealityEngine

# Image Generation modules
from .image_generation.visual_patterns import VisualPatternDatabase
from .image_generation.text_to_visual import TextToVisualDecomposer
from .image_generation.scene_composer import SceneComposer
from .image_generation.renderer import PatternRenderer, RenderSettings
from .image_generation.consciousness_controller import ImageGenerationController
from .image_generation.optimization import RealTimeOptimizer
from .web_access import WebAccess
from .hardware_acceleration import HardwareAccelerator
from .memory_system import MemorySystem

# New modules
from .web_search_module import WebSearchModule
from .daily_pattern_updater import DailyPatternUpdater


@dataclass
class ConsciousnessState:
    """Current state of SI consciousness"""
    attention_focus: str
    active_domains: List[str]
    reasoning_mode: str
    confidence_level: float
    cognitive_load: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'attention_focus': self.attention_focus,
            'active_domains': self.active_domains,
            'reasoning_mode': self.reasoning_mode,
            'confidence_level': self.confidence_level,
            'cognitive_load': self.cognitive_load,
            'timestamp': self.timestamp
        }


@dataclass
class SIResponse:
    """Complete response from SI"""
    id: str
    query: str
    response: str
    confidence: float
    reasoning_strategy: str
    patterns_used: int
    domains_involved: List[str]
    reasoning_steps: List[Dict]
    insights: List[Dict]
    metadata: Dict[str, Any]
    response_time_ms: float
    consciousness_state: Dict
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'query': self.query,
            'response': self.response,
            'confidence': self.confidence,
            'reasoning_strategy': self.reasoning_strategy,
            'patterns_used': self.patterns_used,
            'domains_involved': self.domains_involved,
            'reasoning_steps': self.reasoning_steps,
            'insights': self.insights,
            'metadata': self.metadata,
            'response_time_ms': self.response_time_ms,
            'consciousness_state': self.consciousness_state
        }


class SyntheticIntelligence:
    """
    Main Synthetic Intelligence Engine
    Orchestrates all SI components for intelligent query processing
    
    Now includes:
    - ScalablePatternDatabase (FAISS-based) for high-performance pattern matching
    - WebSearchModule for real-time web search and pattern extraction
    - DailyPatternUpdater for automated knowledge updates
    """
    
    def __init__(self, use_scalable_db: bool = True):
        """
        Initialize SI Engine
        
        Args:
            use_scalable_db: Whether to use FAISS-based scalable pattern database (default: True)
        """
        self.use_scalable_db = use_scalable_db
        
        # Initialize pattern databases
        # Keep original for compatibility with other components
        self.pattern_db = PatternDatabase()
        self.pattern_db.initialize_with_seed_data()
        
        # Initialize scalable pattern database (FAISS-based)
        if use_scalable_db:
            print("🚀 Initializing FAISS-based ScalablePatternDatabase...")
            self.scalable_pattern_db = ScalablePatternDatabase(
                dimension=256,
                nlist=100,
                nprobe=10
            )
            self._migrate_patterns_to_scalable_db()
            print(f"✅ ScalablePatternDatabase ready with {len(self.scalable_pattern_db.patterns)} patterns")
        else:
            self.scalable_pattern_db = None
        
        self.entity_kb = EntityKnowledgeBase()
        self.entity_kb.initialize()
        
        self.pattern_matcher = PatternMatcher(self.pattern_db)
        self.reasoning_engine = QuasiReasoningEngine(self.pattern_db, self.pattern_matcher)
        self.language_generator = SyntheticLanguageGenerator(self.pattern_db)
        self.emergent_reasoning = EmergentReasoning(self.pattern_db, self.entity_kb)
        self.world_model = WorldModelingEngine(self.pattern_db)
        self.self_model = SelfModelingEngine(self.pattern_db)
        self.parallel_reality = ParallelRealityEngine(self.pattern_db, self.world_model)
        
        # Image Generation components
        self.visual_pattern_db = VisualPatternDatabase()
        self.visual_pattern_db.initialize()
        
        self.text_decomposer = TextToVisualDecomposer(self.pattern_db, self.entity_kb)
        self.scene_composer = SceneComposer(self.visual_pattern_db, self.reasoning_engine, self.self_model)
        self.image_renderer = PatternRenderer(self.visual_pattern_db)
        
        self.image_controller = ImageGenerationController(
            self.visual_pattern_db,
            self.text_decomposer,
            self.scene_composer,
            self.image_renderer,
            self.self_model,
            self.reasoning_engine
        )
        
        self.image_optimizer = RealTimeOptimizer(
            self.visual_pattern_db,
            self.text_decomposer,
            self.scene_composer,
            self.image_renderer
        )
                
        # New capabilities
        self.web_access = WebAccess()
        self.hardware = HardwareAccelerator()
        self.memory = MemorySystem()
        
        # Initialize web search module
        self.web_search = WebSearchModule(
            cache_size=1000,
            cache_ttl=3600,
            auto_add_to_db=True
        )
        # Connect to pattern database for auto-adding patterns
        if use_scalable_db:
            self.web_search.set_pattern_database(self.scalable_pattern_db)
        else:
            self.web_search.set_pattern_database(self.pattern_db)
        
        # Initialize daily pattern updater
        self.daily_updater = DailyPatternUpdater(
            pattern_db=self.scalable_pattern_db if use_scalable_db else self.pattern_db,
            run_time="02:00",
            max_patterns_per_run=100
        )
        
        # Log hardware info
        hw_info = self.hardware.hardware_info
        print(f"🔧 Hardware: {hw_info.cpu_count} CPUs, {hw_info.ram_total_gb:.1f}GB RAM")
        
        # Consciousness state
        self.consciousness = ConsciousnessState(
            attention_focus='idle',
            active_domains=[],
            reasoning_mode='ready',
            confidence_level=0.8,
            cognitive_load=0.0
        )
        
        # Session tracking
        self.sessions: Dict[str, Dict] = {}
        self.current_session_id: Optional[str] = None
    
    def _migrate_patterns_to_scalable_db(self):
        """Migrate patterns from PatternDatabase to ScalablePatternDatabase"""
        seed_patterns = self.pattern_db._get_seed_patterns()
        
        for p_data in seed_patterns:
            scalable_pattern = ScalablePattern(
                id=p_data.get('id', str(uuid.uuid4())),
                pattern=p_data['pattern'],
                response=p_data['response'],
                domain=p_data['domain'],
                topics=p_data['topics'],
                keywords=p_data.get('keywords', []),
                success_rate=p_data.get('success_rate', 0.9),
                confidence=p_data.get('confidence', 0.85)
            )
            self.scalable_pattern_db.add_pattern(scalable_pattern)
        
        # Build FAISS index
        self.scalable_pattern_db.build_index(show_progress=True)
        
    def process_query(self, query: str, session_id: Optional[str] = None) -> SIResponse:
        """
        Main query processing pipeline
        """
        start_time = time.time()
        
        # Update consciousness state
        self._update_consciousness('processing', query)
        
        # Step 1: Pattern Matching
        matched_patterns = self.pattern_matcher.match(query, top_k=5)
        
        # Step 2: Entity Grounding
        grounded_entities = self.entity_kb.ground_query(query)
        
        # Step 3: Reasoning
        reasoning_result = self.reasoning_engine.reason(query, matched_patterns)
        
        # Step 4: Cross-domain insights
        insights = self.emergent_reasoning.find_cross_domain_connections(
            query, matched_patterns, limit=2
        )
        
        # Step 5: Generate response
        best_match = matched_patterns[0] if matched_patterns else None
        base_response = best_match.response if best_match else reasoning_result.conclusion
        
        response_text = self.language_generator.generate_response(
            query=query,
            matched_response=base_response,
            response_type=self._determine_response_type(reasoning_result.query_type),
            confidence=reasoning_result.confidence
        )
        
        # Step 6: Check for aha moment
        aha = self.emergent_reasoning.generate_aha_moment(query, matched_patterns, reasoning_result)
        if aha:
            response_text += f"\n\n{aha['message']}"
            
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Record trace for self-improvement
        domains = list(set(p.domain for p in matched_patterns if hasattr(p, 'domain')))
        self.self_model.record_trace(
            query=query,
            response_time_ms=response_time_ms,
            confidence=reasoning_result.confidence,
            strategy=reasoning_result.primary_strategy.value,
            patterns_matched=len(matched_patterns),
            domain=domains[0] if domains else 'general',
            query_type=reasoning_result.query_type.value
        )
        
        # Update pattern statistics
        for pattern in matched_patterns[:3]:
            self.pattern_db.update_pattern_stats(pattern.pattern_id, success=True)
            
        # Build response
        response = SIResponse(
            id=str(uuid.uuid4()),
            query=query,
            response=response_text,
            confidence=reasoning_result.confidence,
            reasoning_strategy=reasoning_result.primary_strategy.value,
            patterns_used=len(matched_patterns),
            domains_involved=domains,
            reasoning_steps=[s.to_dict() for s in reasoning_result.steps],
            insights=[i.to_dict() for i in insights],
            metadata={
                'query_type': reasoning_result.query_type.value,
                'entities_found': len(grounded_entities),
                'alternatives': reasoning_result.alternative_interpretations
            },
            response_time_ms=response_time_ms,
            consciousness_state=self.consciousness.to_dict()
        )
        
        # Update consciousness
        self._update_consciousness('completed', query, reasoning_result.confidence)
        
        return response
        
    def _determine_response_type(self, query_type) -> str:
        """Map query type to response type"""
        mapping = {
            'what': 'definition',
            'how': 'explanation',
            'why': 'reason',
            'compare': 'comparison',
            'define': 'definition',
            'explain': 'explanation'
        }
        return mapping.get(query_type.value, 'synthesis')
        
    def _update_consciousness(self, mode: str, focus: str, confidence: float = 0.8):
        """Update consciousness state"""
        self.consciousness.attention_focus = focus[:50]
        self.consciousness.reasoning_mode = mode
        self.consciousness.confidence_level = confidence
        self.consciousness.cognitive_load = 0.5 if mode == 'processing' else 0.1
        self.consciousness.timestamp = time.time()
        
    def simulate_realities(
        self, 
        query: str, 
        assumptions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run parallel reality simulation
        """
        result = self.parallel_reality.simulate_parallel_realities(
            query=query,
            initial_assumptions=assumptions,
            num_branches=4
        )
        return result.to_dict()
        
    def causal_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Perform causal reasoning about a query
        """
        return self.world_model.reason_causally(query)
        
    def counterfactual(self, intervention: str, target: str) -> Dict[str, Any]:
        """
        Counterfactual reasoning
        """
        return self.world_model.counterfactual_reasoning(intervention, target)
        
    def get_patterns(
        self, 
        domain: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get patterns from database"""
        if domain:
            patterns = self.pattern_db.get_patterns_by_domain(domain)
        else:
            patterns = self.pattern_db.get_all_patterns()
            
        return [p.to_dict() for p in patterns[:limit]]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive SI statistics"""
        pattern_stats = self.pattern_db.get_statistics()
        entity_stats = self.entity_kb.get_statistics()
        performance_stats = self.self_model.get_performance_stats()
        causal_stats = self.world_model.get_graph_stats()
        
        return {
            'patterns': {
                'total': pattern_stats['total_patterns'],
                'domains': pattern_stats['domains'],
                'avg_success_rate': pattern_stats['avg_success_rate']
            },
            'entities': {
                'total': entity_stats['total_entities'],
                'categories': entity_stats['categories']
            },
            'performance': performance_stats['overview'],
            'causal_graph': causal_stats,
            'consciousness': self.consciousness.to_dict(),
            'self_model': self.self_model.get_self_model().to_dict()
        }
        
    def self_improve(self) -> Dict[str, Any]:
        """
        Trigger self-improvement analysis
        """
        return self.self_model.recursive_improve()
        
    def observe_self(self) -> Dict[str, Any]:
        """
        Get self-observation report
        """
        return self.self_model.observe_self()
        
    def add_pattern(self, pattern_data: Dict) -> Dict:
        """Add a new pattern to the database"""
        pattern = self.pattern_db.add_pattern(pattern_data)
        self.pattern_matcher.refresh_cache()
        self.language_generator.refresh_model()
        return pattern.to_dict()
        
    def provide_feedback(self, response_id: str, success: bool):
        """Provide feedback on a response"""
        # Find the trace for this response and update
        for trace in self.self_model.traces:
            if trace.id == response_id:
                self.self_model.record_feedback(trace.id, success)
                break
                
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get reasoning strategy statistics"""
        return self.reasoning_engine.get_strategy_stats()
        
    def search_entities(self, query: str) -> List[Dict]:
        """Search entities in knowledge base"""
        entities = self.entity_kb.search_entities(query, limit=10)
        return [e.to_dict() for e in entities]
    
    # ==================== IMAGE GENERATION ====================
    
    def generate_image(self, description: str, use_optimizer: bool = True) -> Dict[str, Any]:
        """
        Generate an image from text description using pattern-based composition
        
        Args:
            description: Text description of the image to generate
            use_optimizer: Whether to use the optimized pipeline (default: True)
            
        Returns:
            Dict with SVG, PNG (base64), timing info, and consciousness trace
        """
        if use_optimizer:
            return self.image_optimizer.generate_optimized(description)
        else:
            return self.image_controller.generate_image(description)
    
    def get_visual_patterns(self, tags: Optional[List[str]] = None, limit: int = 20) -> List[Dict]:
        """Get visual patterns from database"""
        if tags:
            patterns = self.visual_pattern_db.search_by_tags(tags, limit=limit)
        else:
            patterns = self.visual_pattern_db.get_all_patterns()[:limit]
        return [p.to_dict() for p in patterns]
    
    def get_visual_pattern_preview(self, pattern_id: str) -> Dict[str, Any]:
        """Get SVG preview of a visual pattern"""
        return self.image_renderer.render_pattern_preview(pattern_id)
    
    def get_image_generation_stats(self) -> Dict[str, Any]:
        """Get image generation statistics"""
        return {
            'visual_patterns': self.visual_pattern_db.get_statistics(),
            'optimizer_metrics': self.image_optimizer.get_metrics(),
            'controller_stats': self.image_controller.get_generation_stats()
        }
