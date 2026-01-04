"""
EmergentReasoning - Cross-domain pattern connections
Generates analogical insights and "aha moments"
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Insight:
    """Represents an emergent insight"""
    id: str
    source_domain: str
    target_domain: str
    connection: str
    explanation: str
    confidence: float
    novelty_score: float
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source_domain': self.source_domain,
            'target_domain': self.target_domain,
            'connection': self.connection,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'novelty_score': self.novelty_score
        }


class EmergentReasoning:
    """
    Cross-domain reasoning engine
    Finds unexpected connections between different knowledge domains
    """
    
    def __init__(self, pattern_database, entity_knowledge_base=None):
        self.db = pattern_database
        self.entity_kb = entity_knowledge_base
        
        # Store discovered insights
        self.insights: List[Insight] = []
        
        # Cross-domain connection templates
        self.connection_templates = [
            "The relationship between {concept1} and {concept2} mirrors how {domain1} concepts apply to {domain2}",
            "{concept1} in {domain1} is analogous to {concept2} in {domain2}",
            "Understanding {concept1} can illuminate {concept2} through shared principles of {principle}",
            "Both {concept1} and {concept2} demonstrate {shared_property}"
        ]
        
        # Domain relationship weights
        self.domain_affinities = {
            ('science', 'philosophy'): 0.8,
            ('science', 'technology'): 0.9,
            ('philosophy', 'psychology'): 0.85,
            ('mathematics', 'science'): 0.95,
            ('technology', 'economics'): 0.7,
            ('psychology', 'philosophy'): 0.8,
            ('history', 'politics'): 0.85,
            ('linguistics', 'philosophy'): 0.75,
        }
        
    def find_cross_domain_connections(
        self, 
        query: str,
        source_patterns: List[Any],
        limit: int = 3
    ) -> List[Insight]:
        """
        Find connections between patterns from different domains
        """
        insights = []
        
        if not source_patterns:
            return insights
            
        # Get source domain(s)
        source_domains = set()
        for p in source_patterns:
            if hasattr(p, 'domain'):
                source_domains.add(p.domain)
                
        # Find patterns from other domains that might connect
        all_patterns = self.db.get_all_patterns()
        
        for source_pattern in source_patterns[:3]:
            source_domain = getattr(source_pattern, 'domain', 'general')
            source_keywords = set(getattr(source_pattern, 'keywords', []))
            
            for target_pattern in all_patterns:
                target_domain = target_pattern.domain
                
                # Skip same domain
                if target_domain == source_domain:
                    continue
                    
                # Check for keyword overlap
                target_keywords = set(target_pattern.keywords)
                overlap = source_keywords & target_keywords
                
                if overlap or self._has_semantic_connection(source_pattern, target_pattern):
                    # Calculate connection strength
                    affinity = self._get_domain_affinity(source_domain, target_domain)
                    overlap_score = len(overlap) / max(len(source_keywords), 1)
                    
                    confidence = (affinity + overlap_score) / 2
                    
                    if confidence > 0.3:
                        insight = self._create_insight(
                            source_pattern, target_pattern,
                            list(overlap), confidence
                        )
                        insights.append(insight)
                        
        # Sort by confidence and novelty
        insights.sort(key=lambda x: (x.confidence, x.novelty_score), reverse=True)
        return insights[:limit]
        
    def _get_domain_affinity(self, domain1: str, domain2: str) -> float:
        """Get affinity between two domains"""
        key1 = (domain1, domain2)
        key2 = (domain2, domain1)
        
        if key1 in self.domain_affinities:
            return self.domain_affinities[key1]
        if key2 in self.domain_affinities:
            return self.domain_affinities[key2]
            
        return 0.5  # Default moderate affinity
        
    def _has_semantic_connection(self, pattern1: Any, pattern2: Any) -> bool:
        """Check for semantic connection between patterns"""
        # Check topic overlap
        topics1 = set(getattr(pattern1, 'topics', []))
        topics2 = set(getattr(pattern2, 'topics', []))
        
        if topics1 & topics2:
            return True
            
        # Check response text for shared concepts
        response1 = getattr(pattern1, 'response', '').lower()
        response2 = getattr(pattern2, 'response', '').lower()
        
        # Simple check for shared significant words
        words1 = set(response1.split())
        words2 = set(response2.split())
        
        shared = words1 & words2
        # Filter out common words
        significant_shared = {w for w in shared if len(w) > 5}
        
        return len(significant_shared) > 2
        
    def _create_insight(
        self,
        source: Any,
        target: Any,
        shared_keywords: List[str],
        confidence: float
    ) -> Insight:
        """Create an insight from pattern connection"""
        import uuid
        
        source_domain = getattr(source, 'domain', 'general')
        target_domain = getattr(target, 'domain', 'general')
        
        # Generate connection description
        if shared_keywords:
            connection = f"Shared concepts: {', '.join(shared_keywords[:3])}"
        else:
            connection = f"Structural similarity between domains"
            
        # Generate explanation
        template = random.choice(self.connection_templates)
        explanation = template.format(
            concept1=getattr(source, 'pattern', 'concept'),
            concept2=getattr(target, 'pattern', 'concept'),
            domain1=source_domain,
            domain2=target_domain,
            principle="fundamental patterns",
            shared_property="similar structural organization"
        )
        
        # Calculate novelty (inverse of how often we see this connection)
        novelty = 1.0 - self._get_domain_affinity(source_domain, target_domain)
        
        return Insight(
            id=str(uuid.uuid4()),
            source_domain=source_domain,
            target_domain=target_domain,
            connection=connection,
            explanation=explanation,
            confidence=confidence,
            novelty_score=novelty
        )
        
    def generate_aha_moment(
        self,
        query: str,
        patterns: List[Any],
        reasoning_result: Any
    ) -> Optional[Dict]:
        """
        Generate an "aha moment" - a surprising insight
        """
        # Look for unexpected connections
        insights = self.find_cross_domain_connections(query, patterns, limit=1)
        
        if not insights:
            return None
            
        best_insight = insights[0]
        
        # Only report as "aha moment" if sufficiently novel
        if best_insight.novelty_score < 0.4:
            return None
            
        return {
            'type': 'aha_moment',
            'insight': best_insight.to_dict(),
            'trigger': query,
            'message': f"💡 Interesting connection: {best_insight.explanation}"
        }
        
    def explore_analogies(
        self,
        concept: str,
        source_domain: str,
        target_domains: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Explore analogies for a concept across domains
        """
        analogies = []
        
        # Get patterns related to the concept
        source_patterns = self.db.search(concept, limit=5)
        
        if target_domains is None:
            target_domains = ['science', 'philosophy', 'technology', 'psychology', 'economics']
            
        for domain in target_domains:
            if domain == source_domain:
                continue
                
            # Find patterns in target domain
            target_patterns = self.db.get_patterns_by_domain(domain)
            
            for target in target_patterns[:5]:
                # Check for potential analogy
                connection_score = self._calculate_analogy_score(
                    source_patterns, target
                )
                
                if connection_score > 0.3:
                    analogies.append({
                        'source_concept': concept,
                        'target_concept': target.pattern,
                        'target_domain': domain,
                        'score': connection_score,
                        'explanation': f"{concept} relates to {target.pattern} through shared structural patterns"
                    })
                    
        # Sort by score
        analogies.sort(key=lambda x: x['score'], reverse=True)
        return analogies[:5]
        
    def _calculate_analogy_score(self, sources: List[Any], target: Any) -> float:
        """Calculate analogy score between source patterns and target"""
        if not sources:
            return 0.0
            
        max_score = 0.0
        target_keywords = set(getattr(target, 'keywords', []))
        target_topics = set(getattr(target, 'topics', []))
        
        for source in sources:
            source_keywords = set(getattr(source, 'keywords', []))
            source_topics = set(getattr(source, 'topics', []))
            
            keyword_overlap = len(source_keywords & target_keywords) / max(len(source_keywords | target_keywords), 1)
            topic_overlap = len(source_topics & target_topics) / max(len(source_topics | target_topics), 1)
            
            score = keyword_overlap * 0.6 + topic_overlap * 0.4
            max_score = max(max_score, score)
            
        return max_score
        
    def synthesize_insights(self, insights: List[Insight]) -> str:
        """Synthesize multiple insights into a coherent summary"""
        if not insights:
            return "No cross-domain insights available."
            
        if len(insights) == 1:
            return insights[0].explanation
            
        # Combine insights
        domains = set()
        for insight in insights:
            domains.add(insight.source_domain)
            domains.add(insight.target_domain)
            
        summary = f"Analysis reveals connections across {len(domains)} domains ({', '.join(list(domains)[:4])}). "
        summary += "Key patterns: "
        summary += "; ".join([i.connection for i in insights[:3]])
        
        return summary
