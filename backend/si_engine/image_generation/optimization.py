"""
Performance Optimization for Image Generation
Multi-level caching and parallel execution for <500ms response time
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class LRUCache:
    """Least Recently Used cache with max size"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if not found"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove oldest
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


@dataclass
class PipelineStage:
    """A stage in the rendering pipeline"""
    name: str
    func: Callable
    target_time_ms: float = 100.0
    enabled: bool = True


class RealTimeOptimizer:
    """
    Optimization controller for real-time image generation
    Target: <500ms total response time
    """
    
    def __init__(self, pattern_db, decomposer, composer, renderer):
        self.pattern_db = pattern_db
        self.decomposer = decomposer
        self.composer = composer
        self.renderer = renderer
        
        # Multi-level caching
        self.L1_cache = LRUCache(maxsize=100)  # Recent generations (RAM)
        self.L2_cache = LRUCache(maxsize=500)  # Larger cache for patterns
        
        # Pre-computed pattern data
        self.pattern_svg_cache: Dict[str, str] = {}
        
        # Pipeline configuration
        self.pipeline_stages: List[PipelineStage] = [
            PipelineStage(
                name="text_decomposition",
                func=self._stage_decompose,
                target_time_ms=50.0
            ),
            PipelineStage(
                name="pattern_matching",
                func=self._stage_match,
                target_time_ms=50.0
            ),
            PipelineStage(
                name="scene_composition",
                func=self._stage_compose,
                target_time_ms=100.0
            ),
            PipelineStage(
                name="rendering",
                func=self._stage_render,
                target_time_ms=200.0
            ),
            PipelineStage(
                name="validation",
                func=self._stage_validate,
                target_time_ms=50.0
            )
        ]
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_time_ms': 0.0,
            'stage_times': {stage.name: [] for stage in self.pipeline_stages}
        }
    
    def generate_optimized(self, description: str) -> Dict[str, Any]:
        """
        Generate image with optimization
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        # Check L1 cache first
        cache_key = self._generate_cache_key(description)
        cached = self.L1_cache.get(cache_key)
        
        if cached:
            self.metrics['cache_hits'] += 1
            cached['cache_hit'] = True
            cached['total_time_ms'] = (time.time() - start_time) * 1000
            return cached
        
        self.metrics['cache_misses'] += 1
        
        # Execute pipeline
        context = {
            'description': description,
            'stage_times': {},
            'success': True
        }
        
        for stage in self.pipeline_stages:
            if not stage.enabled:
                continue
                
            stage_start = time.time()
            try:
                context = stage.func(context)
            except Exception as e:
                context['success'] = False
                context['error'] = f"Stage {stage.name} failed: {str(e)}"
                break
            
            stage_time = (time.time() - stage_start) * 1000
            context['stage_times'][stage.name] = stage_time
            self.metrics['stage_times'][stage.name].append(stage_time)
            
            # Warning if stage exceeds target
            if stage_time > stage.target_time_ms * 1.5:
                context.setdefault('warnings', []).append(
                    f"Stage {stage.name} exceeded target: {stage_time:.0f}ms > {stage.target_time_ms}ms"
                )
        
        total_time = (time.time() - start_time) * 1000
        context['total_time_ms'] = total_time
        context['cache_hit'] = False
        
        # Update metrics
        n = self.metrics['total_requests']
        self.metrics['avg_time_ms'] = (
            (self.metrics['avg_time_ms'] * (n - 1) + total_time) / n
        )
        
        # Cache successful results
        if context['success']:
            self.L1_cache.put(cache_key, context.copy())
        
        return context
    
    def _generate_cache_key(self, description: str) -> str:
        """Generate cache key from description"""
        # Normalize and hash
        normalized = description.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _stage_decompose(self, context: Dict) -> Dict:
        """Text decomposition stage"""
        description = context['description']
        decomposition = self.decomposer.decompose(description)
        context['decomposition'] = decomposition
        return context
    
    def _stage_match(self, context: Dict) -> Dict:
        """Pattern matching stage"""
        decomposition = context['decomposition']
        matched = self.decomposer.match_concepts_to_patterns(
            decomposition, 
            self.pattern_db
        )
        context['matched_patterns'] = matched
        return context
    
    def _stage_compose(self, context: Dict) -> Dict:
        """Scene composition stage"""
        matched = context['matched_patterns']
        decomposition = context['decomposition']
        scene_graph = self.composer.compose_scene(matched, decomposition)
        context['scene_graph'] = scene_graph
        return context
    
    def _stage_render(self, context: Dict) -> Dict:
        """Rendering stage"""
        scene_graph = context['scene_graph']
        render_result = self.renderer.render_scene_graph(scene_graph)
        context['svg'] = render_result.get('svg')
        context['png_base64'] = render_result.get('png_base64')
        context['render_time_ms'] = render_result.get('render_time_ms')
        return context
    
    def _stage_validate(self, context: Dict) -> Dict:
        """Validation stage"""
        # Basic validation
        if not context.get('svg'):
            context['success'] = False
            context['error'] = "No SVG generated"
        return context
    
    def precompute_pattern_svgs(self):
        """Pre-compute SVG for all patterns"""
        for pattern in self.pattern_db.get_all_patterns():
            svg = pattern.to_svg()
            self.pattern_svg_cache[pattern.id] = svg
    
    def warm_cache(self, common_queries: List[str]):
        """Warm up cache with common queries"""
        for query in common_queries:
            self.generate_optimized(query)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        
        # Calculate average stage times
        stage_avgs = {}
        for stage_name, times in metrics['stage_times'].items():
            if times:
                stage_avgs[stage_name] = sum(times) / len(times)
            else:
                stage_avgs[stage_name] = 0.0
        
        metrics['avg_stage_times'] = stage_avgs
        metrics['cache_hit_rate'] = (
            self.metrics['cache_hits'] / max(self.metrics['total_requests'], 1)
        )
        metrics['l1_cache_size'] = self.L1_cache.size()
        
        return metrics
    
    def clear_caches(self):
        """Clear all caches"""
        self.L1_cache.clear()
        self.L2_cache.clear()
        self.pattern_svg_cache.clear()
    
    def enable_stage(self, stage_name: str, enabled: bool = True):
        """Enable/disable a pipeline stage"""
        for stage in self.pipeline_stages:
            if stage.name == stage_name:
                stage.enabled = enabled
                break
