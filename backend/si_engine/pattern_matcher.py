"""
PatternMatcher - Pattern recognition and similarity scoring
Uses statistical analysis for pattern effectiveness
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of a pattern match"""
    pattern_id: str
    pattern_text: str
    response: str
    confidence: float
    similarity_score: float
    match_type: str  # 'exact', 'semantic', 'keyword', 'fuzzy'
    matched_keywords: List[str]
    domain: str
    topics: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_text': self.pattern_text,
            'response': self.response,
            'confidence': self.confidence,
            'similarity_score': self.similarity_score,
            'match_type': self.match_type,
            'matched_keywords': self.matched_keywords,
            'domain': self.domain,
            'topics': self.topics
        }


class PatternMatcher:
    """
    Pattern recognition and matching engine
    Uses TF-IDF-like scoring and multiple matching strategies
    """
    
    def __init__(self, pattern_database):
        self.db = pattern_database
        self.idf_cache: Dict[str, float] = {}
        self._build_idf_cache()
        
    def _build_idf_cache(self):
        """Build IDF (Inverse Document Frequency) cache"""
        patterns = self.db.get_all_patterns()
        if not patterns:
            return
            
        # Count document frequency for each term
        doc_count = len(patterns)
        term_doc_freq: Dict[str, int] = Counter()
        
        for pattern in patterns:
            terms = set(self._tokenize(pattern.pattern + ' ' + ' '.join(pattern.keywords)))
            for term in terms:
                term_doc_freq[term] += 1
                
        # Calculate IDF
        for term, freq in term_doc_freq.items():
            self.idf_cache[term] = math.log((doc_count + 1) / (freq + 1)) + 1
            
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
        
    def _calculate_tf(self, terms: List[str]) -> Dict[str, float]:
        """Calculate term frequency"""
        counter = Counter(terms)
        total = len(terms) if terms else 1
        return {term: count / total for term, count in counter.items()}
        
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Get all terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values())) or 1
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values())) or 1
        
        return dot_product / (mag1 * mag2)
        
    def _get_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Get TF-IDF vector for text"""
        terms = self._tokenize(text)
        tf = self._calculate_tf(terms)
        
        tfidf = {}
        for term, freq in tf.items():
            idf = self.idf_cache.get(term, 1.0)
            tfidf[term] = freq * idf
            
        return tfidf
        
    def match(self, query: str, top_k: int = 5) -> List[MatchResult]:
        """
        Find best matching patterns for a query
        Uses multiple strategies: exact, keyword, semantic
        """
        results = []
        query_lower = query.lower().strip()
        query_terms = set(self._tokenize(query))
        query_vector = self._get_tfidf_vector(query)
        
        patterns = self.db.get_all_patterns()
        
        for pattern in patterns:
            # Try different matching strategies
            match_result = self._match_pattern(
                query_lower, query_terms, query_vector, pattern
            )
            if match_result:
                results.append(match_result)
                
        # Sort by confidence and similarity
        results.sort(key=lambda x: (x.confidence, x.similarity_score), reverse=True)
        return results[:top_k]
        
    def _match_pattern(
        self, 
        query_lower: str, 
        query_terms: set, 
        query_vector: Dict[str, float],
        pattern
    ) -> Optional[MatchResult]:
        """Match a single pattern against query"""
        
        pattern_text_lower = pattern.pattern.lower()
        
        # Strategy 1: Exact match
        if query_lower == pattern_text_lower:
            return MatchResult(
                pattern_id=pattern.id,
                pattern_text=pattern.pattern,
                response=pattern.response,
                confidence=0.99,
                similarity_score=1.0,
                match_type='exact',
                matched_keywords=pattern.keywords,
                domain=pattern.domain,
                topics=pattern.topics
            )
            
        # Strategy 2: Substring match
        if query_lower in pattern_text_lower or pattern_text_lower in query_lower:
            return MatchResult(
                pattern_id=pattern.id,
                pattern_text=pattern.pattern,
                response=pattern.response,
                confidence=0.85,
                similarity_score=0.9,
                match_type='substring',
                matched_keywords=pattern.keywords,
                domain=pattern.domain,
                topics=pattern.topics
            )
            
        # Strategy 3: Keyword matching
        pattern_keywords = set(k.lower() for k in pattern.keywords)
        matched_keywords = query_terms & pattern_keywords
        
        if matched_keywords:
            keyword_score = len(matched_keywords) / max(len(pattern_keywords), 1)
            if keyword_score >= 0.3:  # At least 30% keyword match
                return MatchResult(
                    pattern_id=pattern.id,
                    pattern_text=pattern.pattern,
                    response=pattern.response,
                    confidence=min(0.8, 0.5 + keyword_score * 0.4),
                    similarity_score=keyword_score,
                    match_type='keyword',
                    matched_keywords=list(matched_keywords),
                    domain=pattern.domain,
                    topics=pattern.topics
                )
                
        # Strategy 4: Semantic similarity (TF-IDF cosine)
        pattern_vector = self._get_tfidf_vector(pattern.pattern)
        similarity = self._cosine_similarity(query_vector, pattern_vector)
        
        if similarity >= 0.2:  # Minimum similarity threshold
            # Also consider keyword overlap
            all_pattern_terms = set(self._tokenize(pattern.pattern))
            term_overlap = len(query_terms & all_pattern_terms)
            
            combined_score = similarity * 0.7 + (term_overlap / max(len(query_terms), 1)) * 0.3
            
            if combined_score >= 0.25:
                return MatchResult(
                    pattern_id=pattern.id,
                    pattern_text=pattern.pattern,
                    response=pattern.response,
                    confidence=min(0.75, combined_score),
                    similarity_score=similarity,
                    match_type='semantic',
                    matched_keywords=list(query_terms & set(k.lower() for k in pattern.keywords)),
                    domain=pattern.domain,
                    topics=pattern.topics
                )
                
        return None
        
    def get_pattern_effectiveness(self, pattern_id: str) -> Dict[str, Any]:
        """Get effectiveness statistics for a pattern"""
        pattern = self.db.get_pattern(pattern_id)
        if not pattern:
            return {}
            
        return {
            'pattern_id': pattern_id,
            'success_rate': pattern.success_rate,
            'usage_count': pattern.usage_count,
            'confidence': pattern.confidence,
            'domain': pattern.domain,
            'effectiveness_score': pattern.success_rate * pattern.confidence
        }
        
    def find_similar_patterns(self, pattern_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find patterns similar to a given pattern"""
        source_pattern = self.db.get_pattern(pattern_id)
        if not source_pattern:
            return []
            
        source_vector = self._get_tfidf_vector(
            source_pattern.pattern + ' ' + ' '.join(source_pattern.keywords)
        )
        
        similarities = []
        for pattern in self.db.get_all_patterns():
            if pattern.id == pattern_id:
                continue
                
            target_vector = self._get_tfidf_vector(
                pattern.pattern + ' ' + ' '.join(pattern.keywords)
            )
            similarity = self._cosine_similarity(source_vector, target_vector)
            
            if similarity > 0.1:
                similarities.append((pattern.id, similarity))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def refresh_cache(self):
        """Refresh the IDF cache after pattern updates"""
        self._build_idf_cache()
