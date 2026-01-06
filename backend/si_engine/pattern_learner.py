"""
PatternLearner - Autonomous pattern generation and database expansion
Teaches the SI to learn and create its own patterns through supervised learning
"""

import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class LearnedPattern:
    """Represents a pattern learned by the SI"""
    query_pattern: str
    response_template: str
    domain: str
    confidence: float
    source: str  # 'web', 'interaction', 'synthesis'
    examples: List[str] = field(default_factory=list)
    success_rate: float = 0.5
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'query_pattern': self.query_pattern,
            'response_template': self.response_template,
            'domain': self.domain,
            'confidence': self.confidence,
            'source': self.source,
            'examples': self.examples,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'metadata': self.metadata
        }

class PatternLearner:
    """
    Autonomous pattern learning system
    Allows SI to expand its pattern database through:
    1. Web scraping and content analysis
    2. Interaction learning from conversations
    3. Pattern synthesis from existing patterns
    """
    
    def __init__(self, web_access=None, pattern_db=None, memory=None):
        self.web_access = web_access
        self.pattern_db = pattern_db
        self.memory = memory
        
        # Learning metrics
        self.patterns_generated = 0
        self.patterns_approved = 0
        self.patterns_rejected = 0
        
        # Pending patterns awaiting approval
        self.pending_patterns: List[LearnedPattern] = []
        
        logger.info("PatternLearner initialized")
    
    def learn_from_web(self, query: str, num_sources: int = 3) -> List[LearnedPattern]:
        """
        Learn patterns by scraping web content related to a query
        """
        if not self.web_access:
            logger.warning("No web access available")
            return []
        
        logger.info(f"Learning from web: {query}")
        
        # Search for relevant content
        search_results = self.web_access.search_web(query, num_results=num_sources)
        
        learned_patterns = []
        
        for result in search_results:
            # Extract potential patterns from content
            patterns = self._extract_patterns_from_content(
                query, 
                result.content,
                domain=self._infer_domain(query)
            )
            learned_patterns.extend(patterns)
        
        # Add to pending for approval
        self.pending_patterns.extend(learned_patterns)
        self.patterns_generated += len(learned_patterns)
        
        logger.info(f"Generated {len(learned_patterns)} patterns from web")
        return learned_patterns
    
    def learn_from_interaction(self, query: str, response: str, 
                              feedback: float = 0.5) -> Optional[LearnedPattern]:
        """
        Learn a pattern from a successful interaction
        Feedback: 0-1 scale (0=bad, 1=excellent)
        """
        logger.info(f"Learning from interaction (feedback: {feedback})")
        
        # Only learn from good interactions
        if feedback < 0.6:
            return None
        
        # Extract pattern structure
        query_pattern = self._generalize_query(query)
        response_template = self._create_response_template(response)
        domain = self._infer_domain(query)
        
        pattern = LearnedPattern(
            query_pattern=query_pattern,
            response_template=response_template,
            domain=domain,
            confidence=feedback,
            source='interaction',
            examples=[query],
            success_rate=feedback
        )
        
        self.pending_patterns.append(pattern)
        self.patterns_generated += 1
        
        logger.info(f"Created interaction pattern: {query_pattern[:50]}...")
        return pattern
    
    def synthesize_patterns(self, existing_patterns: List[Dict]) -> List[LearnedPattern]:
        """
        Create new patterns by combining and modifying existing ones
        """
        logger.info(f"Synthesizing from {len(existing_patterns)} patterns")
        
        synthesized = []
        
        # Group patterns by domain
        domain_groups = {}
        for p in existing_patterns:
            domain = p.get('domain', 'general')
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(p)
        
        # For each domain, create variations
        for domain, patterns in domain_groups.items():
            if len(patterns) >= 2:
                # Create hybrid patterns
                for i in range(min(3, len(patterns) - 1)):
                    hybrid = self._hybridize_patterns(patterns[i], patterns[i+1], domain)
                    if hybrid:
                        synthesized.append(hybrid)
        
        self.pending_patterns.extend(synthesized)
        self.patterns_generated += len(synthesized)
        
        logger.info(f"Synthesized {len(synthesized)} new patterns")
        return synthesized
    
    def _extract_patterns_from_content(self, query: str, content: str, 
                                      domain: str) -> List[LearnedPattern]:
        """
        Extract Q&A patterns from web content
        """
        patterns = []
        
        # Look for definition patterns
        definition_matches = re.findall(r'(?:is|are|means?)\s+(.{20,200})', content, re.I)
        if definition_matches and len(definition_matches) > 0:
            pattern = LearnedPattern(
                query_pattern=f"What is {{{domain}_topic}}?",
                response_template=definition_matches[0].strip(),
                domain=domain,
                confidence=0.6,
                source='web',
                examples=[query]
            )
            patterns.append(pattern)
        
        # Look for how-to patterns
        howto_matches = re.findall(r'(?:how to|steps to|way to)\s+(.{20,200})', content, re.I)
        if howto_matches:
            pattern = LearnedPattern(
                query_pattern=f"How to {{{domain}_action}}?",
                response_template=f"To accomplish this: {howto_matches[0].strip()}",
                domain=domain,
                confidence=0.6,
                source='web',
                examples=[query]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _generalize_query(self, query: str) -> str:
        """
        Convert specific query into generalized pattern
        Example: "What is Python?" -> "What is {programming_language}?"
        """
        # Simple generalization - replace specific terms with placeholders
        generalized = query
        
        # Detect and replace specific entities
        # (In production, use NER or entity detection)
        specific_terms = [
            (r'\bPython\b', '{programming_language}'),
            (r'\b\d+\b', '{number}'),
            (r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '{person_name}'),
            (r'\b202[0-9]\b', '{year}')
        ]
        
        for pattern, replacement in specific_terms:
            generalized = re.sub(pattern, replacement, generalized)
        
        return generalized
    
    def _create_response_template(self, response: str) -> str:
        """
        Convert specific response into reusable template
        """
        # For now, return as-is
        # In production, extract structure and create fill-in-the-blank templates
        return response[:500]  # Limit length
    
    def _infer_domain(self, query: str) -> str:
        """
        Infer the domain/topic of a query
        """
        query_lower = query.lower()
        
        domain_keywords = {
            'technology': ['computer', 'software', 'code', 'program', 'tech', 'ai', 'ml'],
            'science': ['atom', 'molecule', 'physics', 'chemistry', 'biology', 'research'],
            'math': ['equation', 'calculate', 'number', 'algebra', 'geometry'],
            'language': ['word', 'grammar', 'language', 'translate', 'meaning'],
            'history': ['war', 'ancient', 'century', 'historical', 'past'],
            'business': ['company', 'market', 'business', 'economy', 'trade']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        
        return 'general'
    
    def _hybridize_patterns(self, pattern1: Dict, pattern2: Dict, domain: str) -> Optional[LearnedPattern]:
        """
        Combine two patterns to create a hybrid
        """
        # Simple hybridization: combine query structures
        try:
            query1 = pattern1.get('pattern', pattern1.get('query_pattern', ''))
            query2 = pattern2.get('pattern', pattern2.get('query_pattern', ''))
            
            # Create a hybrid query pattern
            hybrid_query = f"{query1.split()[0]} {query2.split()[-1]}" if query1 and query2 else None
            
            if not hybrid_query:
                return None
            
            hybrid = LearnedPattern(
                query_pattern=hybrid_query,
                response_template="{synthesized_response}",
                domain=domain,
                confidence=0.5,
                source='synthesis',
                metadata={'parent_patterns': [query1, query2]}
            )
            
            return hybrid
        except Exception as e:
            logger.warning(f"Error hybridizing patterns: {e}")
            return None
    
    def get_pending_patterns(self, limit: int = 10) -> List[LearnedPattern]:
        """
        Get patterns awaiting approval
        """
        return self.pending_patterns[:limit]
    
    def approve_pattern(self, pattern: LearnedPattern) -> bool:
        """
        Approve a pattern and add it to the database
        """
        if pattern in self.pending_patterns:
            self.pending_patterns.remove(pattern)
            
            # Add to pattern database if available
            if self.pattern_db:
                # Convert to database format and store
                self.pattern_db.create(
                    pattern=pattern.query_pattern,
                    response=pattern.response_template,
                    domain=pattern.domain,
                    success_rate=pattern.success_rate
                )
            
            self.patterns_approved += 1
            logger.info(f"Pattern approved: {pattern.query_pattern[:50]}...")
            return True
        
        return False
    
    def reject_pattern(self, pattern: LearnedPattern) -> bool:
        """
        Reject a pattern
        """
        if pattern in self.pending_patterns:
            self.pending_patterns.remove(pattern)
            self.patterns_rejected += 1
            logger.info(f"Pattern rejected: {pattern.query_pattern[:50]}...")
            return True
        
        return False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics
        """
        return {
            'patterns_generated': self.patterns_generated,
            'patterns_approved': self.patterns_approved,
            'patterns_rejected': self.patterns_rejected,
            'pending_count': len(self.pending_patterns),
            'approval_rate': self.patterns_approved / max(1, self.patterns_generated)
        }
