"""
SelfModelingEngine - Self-observation and recursive self-improvement
Tracks performance metrics and implements meta-learning
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timezone


@dataclass
class PerformanceTrace:
    """A trace of SI performance"""
    id: str
    timestamp: float
    query: str
    response_time_ms: float
    confidence: float
    strategy_used: str
    patterns_matched: int
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'query': self.query,
            'response_time_ms': self.response_time_ms,
            'confidence': self.confidence,
            'strategy_used': self.strategy_used,
            'patterns_matched': self.patterns_matched,
            'success': self.success
        }


@dataclass
class SelfModel:
    """The SI's model of itself"""
    strengths: List[str]
    weaknesses: List[str]
    preferred_strategies: Dict[str, float]
    domain_expertise: Dict[str, float]
    avg_confidence: float
    avg_response_time: float
    total_queries: int
    success_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'preferred_strategies': self.preferred_strategies,
            'domain_expertise': self.domain_expertise,
            'avg_confidence': self.avg_confidence,
            'avg_response_time': self.avg_response_time,
            'total_queries': self.total_queries,
            'success_rate': self.success_rate
        }


class SelfModelingEngine:
    """
    Self-observation and improvement engine
    Implements meta-learning and recursive self-improvement
    """
    
    def __init__(self, pattern_database):
        self.db = pattern_database
        
        # Performance traces
        self.traces: List[PerformanceTrace] = []
        self.max_traces = 1000
        
        # Aggregated metrics
        self.metrics = {
            'total_queries': 0,
            'total_response_time_ms': 0,
            'total_confidence': 0,
            'successes': 0,
            'failures': 0,
            'strategy_usage': defaultdict(int),
            'domain_queries': defaultdict(int),
            'query_types': defaultdict(int)
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.improvement_threshold = 0.05
        
        # Self model
        self._self_model: Optional[SelfModel] = None
        
    def record_trace(
        self,
        query: str,
        response_time_ms: float,
        confidence: float,
        strategy: str,
        patterns_matched: int,
        domain: str = 'general',
        query_type: str = 'unknown'
    ) -> PerformanceTrace:
        """Record a performance trace"""
        trace = PerformanceTrace(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            query=query,
            response_time_ms=response_time_ms,
            confidence=confidence,
            strategy_used=strategy,
            patterns_matched=patterns_matched
        )
        
        self.traces.append(trace)
        
        # Maintain max size
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]
            
        # Update metrics
        self.metrics['total_queries'] += 1
        self.metrics['total_response_time_ms'] += response_time_ms
        self.metrics['total_confidence'] += confidence
        self.metrics['strategy_usage'][strategy] += 1
        self.metrics['domain_queries'][domain] += 1
        self.metrics['query_types'][query_type] += 1
        
        # Invalidate cached self model
        self._self_model = None
        
        return trace
        
    def record_feedback(self, trace_id: str, success: bool):
        """Record feedback on a trace"""
        for trace in self.traces:
            if trace.id == trace_id:
                trace.success = success
                if success:
                    self.metrics['successes'] += 1
                else:
                    self.metrics['failures'] += 1
                break
                
    def get_self_model(self) -> SelfModel:
        """Get the current self-model"""
        if self._self_model is not None:
            return self._self_model
            
        # Build self model from traces
        total = self.metrics['total_queries'] or 1
        
        # Calculate averages
        avg_confidence = self.metrics['total_confidence'] / total
        avg_response_time = self.metrics['total_response_time_ms'] / total
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Check response time
        if avg_response_time < 100:
            strengths.append("Fast response time (<100ms)")
        elif avg_response_time > 500:
            weaknesses.append("Slow response time (>500ms)")
            
        # Check confidence
        if avg_confidence > 0.7:
            strengths.append("High confidence responses")
        elif avg_confidence < 0.5:
            weaknesses.append("Low confidence responses")
            
        # Check domain expertise
        domain_expertise = {}
        for domain, count in self.metrics['domain_queries'].items():
            expertise = min(1.0, count / 100)  # Max out at 100 queries
            domain_expertise[domain] = expertise
            if expertise > 0.7:
                strengths.append(f"Strong in {domain}")
            elif expertise < 0.2:
                weaknesses.append(f"Limited {domain} knowledge")
                
        # Strategy preferences
        strategy_prefs = {}
        total_strategy = sum(self.metrics['strategy_usage'].values()) or 1
        for strategy, count in self.metrics['strategy_usage'].items():
            strategy_prefs[strategy] = count / total_strategy
            
        # Success rate
        total_feedback = self.metrics['successes'] + self.metrics['failures']
        success_rate = self.metrics['successes'] / total_feedback if total_feedback > 0 else 0.5
        
        if success_rate > 0.8:
            strengths.append("High success rate")
        elif success_rate < 0.5:
            weaknesses.append("Low success rate")
            
        self._self_model = SelfModel(
            strengths=strengths,
            weaknesses=weaknesses,
            preferred_strategies=strategy_prefs,
            domain_expertise=domain_expertise,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            total_queries=total,
            success_rate=success_rate
        )
        
        return self._self_model
        
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for self-improvement"""
        suggestions = []
        model = self.get_self_model()
        
        # Response time improvements
        if model.avg_response_time > 300:
            suggestions.append({
                'area': 'performance',
                'issue': 'High average response time',
                'current': f"{model.avg_response_time:.0f}ms",
                'target': '< 300ms',
                'action': 'Consider caching frequent patterns'
            })
            
        # Confidence improvements
        if model.avg_confidence < 0.6:
            suggestions.append({
                'area': 'accuracy',
                'issue': 'Low average confidence',
                'current': f"{model.avg_confidence:.2f}",
                'target': '> 0.6',
                'action': 'Expand pattern database in weak domains'
            })
            
        # Domain-specific improvements
        for domain, expertise in model.domain_expertise.items():
            if expertise < 0.3:
                suggestions.append({
                    'area': 'knowledge',
                    'issue': f'Limited expertise in {domain}',
                    'current': f"{expertise:.2f}",
                    'target': '> 0.5',
                    'action': f'Add more patterns for {domain}'
                })
                
        # Strategy balance
        if model.preferred_strategies:
            dominant = max(model.preferred_strategies.values())
            if dominant > 0.6:
                suggestions.append({
                    'area': 'reasoning',
                    'issue': 'Over-reliance on single strategy',
                    'current': f"{dominant:.0%} dominance",
                    'target': 'More balanced',
                    'action': 'Diversify reasoning strategies'
                })
                
        return suggestions
        
    def meta_learn(self) -> Dict[str, Any]:
        """
        Perform meta-learning to improve SI behavior
        Returns adjustments to make
        """
        adjustments = {
            'strategy_weights': {},
            'domain_focus': [],
            'pattern_suggestions': []
        }
        
        # Analyze successful vs failed queries
        successful_traces = [t for t in self.traces if t.success == True]
        failed_traces = [t for t in self.traces if t.success == False]
        
        # Learn which strategies work better
        strategy_success = defaultdict(lambda: {'success': 0, 'total': 0})
        for trace in self.traces:
            if trace.success is not None:
                strategy_success[trace.strategy_used]['total'] += 1
                if trace.success:
                    strategy_success[trace.strategy_used]['success'] += 1
                    
        for strategy, stats in strategy_success.items():
            if stats['total'] > 5:  # Minimum samples
                rate = stats['success'] / stats['total']
                adjustments['strategy_weights'][strategy] = rate
                
        # Identify domains needing more patterns
        model = self.get_self_model()
        for domain, expertise in model.domain_expertise.items():
            if expertise < 0.3:
                adjustments['domain_focus'].append(domain)
                
        # Suggest new patterns based on failed queries
        for trace in failed_traces[-10:]:  # Last 10 failures
            adjustments['pattern_suggestions'].append({
                'query': trace.query,
                'reason': 'Query resulted in unsuccessful response'
            })
            
        return adjustments
        
    def observe_self(self) -> Dict[str, Any]:
        """
        Self-observation: analyze own behavior
        """
        model = self.get_self_model()
        recent_traces = self.traces[-100:] if self.traces else []
        
        # Calculate trends
        if len(recent_traces) >= 10:
            first_half = recent_traces[:len(recent_traces)//2]
            second_half = recent_traces[len(recent_traces)//2:]
            
            first_avg_conf = sum(t.confidence for t in first_half) / len(first_half)
            second_avg_conf = sum(t.confidence for t in second_half) / len(second_half)
            
            confidence_trend = 'improving' if second_avg_conf > first_avg_conf else 'declining'
            
            first_avg_time = sum(t.response_time_ms for t in first_half) / len(first_half)
            second_avg_time = sum(t.response_time_ms for t in second_half) / len(second_half)
            
            speed_trend = 'improving' if second_avg_time < first_avg_time else 'declining'
        else:
            confidence_trend = 'insufficient_data'
            speed_trend = 'insufficient_data'
            
        return {
            'self_model': model.to_dict(),
            'trends': {
                'confidence': confidence_trend,
                'speed': speed_trend
            },
            'recent_performance': {
                'queries': len(recent_traces),
                'avg_confidence': sum(t.confidence for t in recent_traces) / len(recent_traces) if recent_traces else 0,
                'avg_response_time': sum(t.response_time_ms for t in recent_traces) / len(recent_traces) if recent_traces else 0
            },
            'improvements': self.get_improvement_suggestions()
        }
        
    def recursive_improve(self) -> Dict[str, Any]:
        """
        Attempt recursive self-improvement
        """
        # Get meta-learning insights
        meta_insights = self.meta_learn()
        
        # Get self-observation
        observation = self.observe_self()
        
        # Generate improvement plan
        improvement_plan = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_state': observation['self_model'],
            'identified_issues': observation['improvements'],
            'proposed_changes': [],
            'expected_outcomes': []
        }
        
        # Generate specific changes based on meta-insights
        if meta_insights['strategy_weights']:
            best_strategies = sorted(
                meta_insights['strategy_weights'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            improvement_plan['proposed_changes'].append({
                'type': 'strategy_rebalancing',
                'details': f"Prioritize strategies: {[s[0] for s in best_strategies]}"
            })
            
        if meta_insights['domain_focus']:
            improvement_plan['proposed_changes'].append({
                'type': 'knowledge_expansion',
                'details': f"Focus on domains: {meta_insights['domain_focus']}"
            })
            
        if meta_insights['pattern_suggestions']:
            improvement_plan['proposed_changes'].append({
                'type': 'pattern_learning',
                'details': f"Learn from {len(meta_insights['pattern_suggestions'])} failed queries"
            })
            
        return improvement_plan
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        model = self.get_self_model()
        
        return {
            'overview': {
                'total_queries': model.total_queries,
                'success_rate': model.success_rate,
                'avg_confidence': model.avg_confidence,
                'avg_response_time_ms': model.avg_response_time
            },
            'by_strategy': dict(self.metrics['strategy_usage']),
            'by_domain': dict(self.metrics['domain_queries']),
            'by_query_type': dict(self.metrics['query_types']),
            'strengths': model.strengths,
            'weaknesses': model.weaknesses
        }
