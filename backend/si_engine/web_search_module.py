"""
WebSearchModule - DuckDuckGo search with caching and auto-add to pattern DB
Provides real-time web search capabilities for the SI engine

Features:
- DuckDuckGo search (no API key required)
- Result caching with TTL
- Automatic pattern extraction from search results
- Integration with ScalablePatternDatabase
- Rate limiting to respect search engines
"""

import os
import re
import time
import json
import hashlib
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Warning: duckduckgo-search not available. Install with: pip install duckduckgo-search")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SearchResult:
    """A single search result"""
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'source': self.source,
            'timestamp': self.timestamp
        }


@dataclass
class CachedSearch:
    """Cached search results with TTL"""
    query: str
    results: List[SearchResult]
    created_at: float
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class ExtractedPattern:
    """Pattern extracted from web content"""
    query: str
    response: str
    source_url: str
    source_title: str
    domain: str
    topics: List[str]
    keywords: List[str]
    confidence: float
    extracted_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'response': self.response,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'domain': self.domain,
            'topics': self.topics,
            'keywords': self.keywords,
            'confidence': self.confidence,
            'extracted_at': self.extracted_at
        }


# ============================================================================
# SEARCH CACHE
# ============================================================================

class SearchCache:
    """LRU cache for search results with TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CachedSearch] = OrderedDict()
        self._lock = threading.RLock()
    
    def _get_key(self, query: str) -> str:
        """Generate cache key from query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached results if available and not expired"""
        key = self._get_key(query)
        
        with self._lock:
            if key not in self.cache:
                return None
            
            cached = self.cache[key]
            if cached.is_expired():
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return cached.results
    
    def put(self, query: str, results: List[SearchResult], ttl: Optional[int] = None):
        """Cache search results"""
        key = self._get_key(query)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            # Add new entry
            self.cache[key] = CachedSearch(
                query=query,
                results=results,
                created_at=time.time(),
                ttl_seconds=ttl
            )
    
    def clear(self):
        """Clear all cached results"""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            expired = sum(1 for c in self.cache.values() if c.is_expired())
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'expired': expired,
                'active': len(self.cache) - expired
            }


# ============================================================================
# KEYWORD EXTRACTOR
# ============================================================================

class KeywordExtractor:
    """Extract keywords and topics from text"""
    
    # Common stop words
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
        'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
        'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
        'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    }
    
    # Domain keywords mapping
    DOMAIN_KEYWORDS = {
        'technology': ['ai', 'machine learning', 'software', 'programming', 'computer', 'algorithm', 'data', 'code', 'developer', 'api', 'cloud', 'neural', 'model'],
        'science': ['research', 'study', 'experiment', 'theory', 'physics', 'biology', 'chemistry', 'scientific', 'discovery', 'quantum', 'particle'],
        'business': ['company', 'startup', 'market', 'revenue', 'investment', 'growth', 'ceo', 'funding', 'ipo', 'acquisition'],
        'health': ['medical', 'health', 'disease', 'treatment', 'drug', 'clinical', 'patient', 'vaccine', 'therapy'],
        'politics': ['government', 'policy', 'election', 'political', 'congress', 'senate', 'vote', 'law', 'regulation'],
        'general': []
    }
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words
        filtered = [w for w in words if w not in self.STOP_WORDS]
        
        # Count frequencies
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def detect_domain(self, text: str) -> str:
        """Detect domain from text content"""
        text_lower = text.lower()
        
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        
        # Return domain with highest score, or 'general' if no matches
        best_domain = max(scores.items(), key=lambda x: x[1])
        return best_domain[0] if best_domain[1] > 0 else 'general'
    
    def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract topic phrases from text"""
        # Simple approach: extract noun phrases (2-3 word combinations)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in self.STOP_WORDS]
        
        # Create bigrams
        bigrams = []
        for i in range(len(filtered) - 1):
            bigram = f"{filtered[i]} {filtered[i+1]}"
            bigrams.append(bigram)
        
        # Count and return top bigrams
        freq = {}
        for bg in bigrams:
            freq[bg] = freq.get(bg, 0) + 1
        
        sorted_bigrams = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [bg for bg, _ in sorted_bigrams[:max_topics]]


# ============================================================================
# WEB SEARCH MODULE
# ============================================================================

class WebSearchModule:
    """
    Web search module with DuckDuckGo integration
    
    Features:
    - Search with caching
    - Pattern extraction from results
    - Auto-add to pattern database
    - Rate limiting
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        rate_limit_delay: float = 1.0,
        auto_add_to_db: bool = True
    ):
        self.cache = SearchCache(max_size=cache_size, default_ttl=cache_ttl)
        self.rate_limit_delay = rate_limit_delay
        self.auto_add_to_db = auto_add_to_db
        self.keyword_extractor = KeywordExtractor()
        
        # Rate limiting
        self.last_search_time = 0
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'patterns_extracted': 0,
            'errors': 0
        }
        
        # Pattern database reference (set externally)
        self.pattern_db = None
    
    def set_pattern_database(self, db):
        """Set pattern database for auto-add functionality"""
        self.pattern_db = db
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        with self._lock:
            elapsed = time.time() - self.last_search_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            self.last_search_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        use_cache: bool = True,
        extract_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results
            use_cache: Whether to use cached results
            extract_patterns: Whether to extract patterns from results
            
        Returns:
            Dict with results, patterns, and metadata
        """
        self.stats['total_searches'] += 1
        
        # Check cache
        if use_cache:
            cached_results = self.cache.get(query)
            if cached_results:
                self.stats['cache_hits'] += 1
                return {
                    'query': query,
                    'results': [r.to_dict() for r in cached_results],
                    'cached': True,
                    'patterns': [],
                    'source': 'cache'
                }
        
        self.stats['cache_misses'] += 1
        
        # Perform search
        try:
            self._respect_rate_limit()
            results = self._search_duckduckgo(query, max_results)
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Search error: {e}")
            return {
                'query': query,
                'results': [],
                'cached': False,
                'error': str(e),
                'source': 'error'
            }
        
        # Cache results
        self.cache.put(query, results)
        
        # Extract patterns if requested
        patterns = []
        if extract_patterns and results:
            patterns = self._extract_patterns_from_results(query, results)
            self.stats['patterns_extracted'] += len(patterns)
            
            # Auto-add to database
            if self.auto_add_to_db and self.pattern_db and patterns:
                self._add_patterns_to_db(patterns)
        
        return {
            'query': query,
            'results': [r.to_dict() for r in results],
            'cached': False,
            'patterns': [p.to_dict() for p in patterns],
            'source': 'duckduckgo'
        }
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform DuckDuckGo search"""
        if not DDGS_AVAILABLE:
            logger.warning("DuckDuckGo search not available")
            return []
        
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get('title', ''),
                        url=r.get('href', r.get('link', '')),
                        snippet=r.get('body', r.get('snippet', '')),
                        source='duckduckgo'
                    ))
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        return results
    
    def _extract_patterns_from_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[ExtractedPattern]:
        """Extract patterns from search results"""
        patterns = []
        
        for result in results[:5]:  # Limit to top 5 results
            if not result.snippet or len(result.snippet) < 50:
                continue
            
            # Extract keywords and domain
            combined_text = f"{result.title} {result.snippet}"
            keywords = self.keyword_extractor.extract_keywords(combined_text)
            domain = self.keyword_extractor.detect_domain(combined_text)
            topics = self.keyword_extractor.extract_topics(combined_text)
            
            # Create pattern
            pattern = ExtractedPattern(
                query=query,
                response=result.snippet,
                source_url=result.url,
                source_title=result.title,
                domain=domain,
                topics=topics,
                keywords=keywords,
                confidence=0.6  # Lower confidence for web-sourced patterns
            )
            patterns.append(pattern)
        
        return patterns
    
    def _add_patterns_to_db(self, patterns: List[ExtractedPattern]):
        """Add extracted patterns to the pattern database"""
        if not self.pattern_db:
            return
        
        import uuid
        
        for ep in patterns:
            try:
                # Create pattern dict compatible with PatternDatabase
                pattern_data = {
                    'id': str(uuid.uuid4()),
                    'pattern': ep.query,
                    'response': ep.response,
                    'domain': ep.domain,
                    'topics': ep.topics,
                    'keywords': ep.keywords,
                    'confidence': ep.confidence,
                    'success_rate': 0.7,
                    'metadata': {
                        'source': 'web_search',
                        'source_url': ep.source_url,
                        'source_title': ep.source_title,
                        'extracted_at': ep.extracted_at
                    }
                }
                
                # Add to database (works with both PatternDatabase and ScalablePatternDatabase)
                if hasattr(self.pattern_db, 'add_pattern'):
                    self.pattern_db.add_pattern(pattern_data)
                    logger.info(f"Added web pattern: {ep.query[:50]}...")
                    
            except Exception as e:
                logger.error(f"Failed to add pattern: {e}")
    
    async def search_async(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Async version of search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query, max_results)
        )
    
    def search_multiple(
        self,
        queries: List[str],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search multiple queries"""
        results = []
        for query in queries:
            result = self.search(query, max_results)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            **self.stats,
            'cache_stats': self.cache.get_stats()
        }
    
    def clear_cache(self):
        """Clear search cache"""
        self.cache.clear()


# ============================================================================
# CONTENT FETCHER
# ============================================================================

class ContentFetcher:
    """Fetch and parse web page content"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from URL"""
        if not BS4_AVAILABLE:
            return None
        
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove scripts and styles
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                
                # Extract text
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                
                return text[:5000]  # Limit to 5000 chars
                
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create search module
    search = WebSearchModule(
        cache_size=100,
        cache_ttl=3600,
        auto_add_to_db=False  # Disable for standalone testing
    )
    
    # Test search
    print("Testing web search...")
    result = search.search("what is machine learning", max_results=5)
    
    print(f"\nQuery: {result['query']}")
    print(f"Source: {result['source']}")
    print(f"Cached: {result['cached']}")
    print(f"Results: {len(result['results'])}")
    
    for i, r in enumerate(result['results'][:3]):
        print(f"\n  {i+1}. {r['title'][:60]}...")
        print(f"     {r['snippet'][:100]}...")
    
    print(f"\nPatterns extracted: {len(result['patterns'])}")
    for p in result['patterns'][:2]:
        print(f"  - Domain: {p['domain']}, Keywords: {p['keywords'][:5]}")
    
    print(f"\nStatistics: {search.get_statistics()}")
