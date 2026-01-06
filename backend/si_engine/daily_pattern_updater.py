"""
DailyPatternUpdater - Automated pattern extraction from tech/AI news sources
Runs daily to keep the SI engine knowledge base current

Features:
- Scheduled daily runs at 2 AM
- Scrapes: Hacker News, ArXiv, GitHub Trending
- Extracts patterns using sentence-transformers
- Auto-adds to ScalablePatternDatabase
- Deduplication and quality filtering
"""

import os
import re
import time
import json
import uuid
import hashlib
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("Warning: schedule not available. Install with: pip install schedule")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NewsItem:
    """A news item from any source"""
    title: str
    url: str
    content: str
    source: str  # 'hackernews', 'arxiv', 'github'
    score: int = 0
    comments: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'content': self.content,
            'source': self.source,
            'score': self.score,
            'comments': self.comments,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ExtractedPattern:
    """Pattern extracted from news content"""
    id: str
    pattern: str  # The query/question
    response: str  # The answer/content
    domain: str
    topics: List[str]
    keywords: List[str]
    source_url: str
    source_title: str
    source_type: str
    confidence: float
    embedding: Optional[List[float]] = None
    extracted_at: float = field(default_factory=time.time)
    
    def to_pattern_dict(self) -> Dict:
        """Convert to pattern database format"""
        return {
            'id': self.id,
            'pattern': self.pattern,
            'response': self.response,
            'domain': self.domain,
            'topics': self.topics,
            'keywords': self.keywords,
            'confidence': self.confidence,
            'success_rate': 0.75,
            'metadata': {
                'source': self.source_type,
                'source_url': self.source_url,
                'source_title': self.source_title,
                'extracted_at': self.extracted_at,
                'auto_generated': True
            }
        }


# ============================================================================
# NEWS SCRAPERS
# ============================================================================

class HackerNewsScraper:
    """Scrape top stories from Hacker News"""
    
    API_BASE = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    
    async def get_top_stories(self, limit: int = 30) -> List[NewsItem]:
        """Get top stories from HN"""
        items = []
        
        try:
            # Get top story IDs
            async with self.session.get(f"{self.API_BASE}/topstories.json") as resp:
                if resp.status != 200:
                    return items
                story_ids = await resp.json()
            
            # Fetch story details (limit concurrency)
            story_ids = story_ids[:limit]
            tasks = [self._fetch_story(sid) for sid in story_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, NewsItem):
                    items.append(result)
                    
        except Exception as e:
            logger.error(f"HN scraper error: {e}")
        
        return items
    
    async def _fetch_story(self, story_id: int) -> Optional[NewsItem]:
        """Fetch a single story"""
        try:
            async with self.session.get(f"{self.API_BASE}/item/{story_id}.json") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            
            if not data or data.get('type') != 'story':
                return None
            
            return NewsItem(
                title=data.get('title', ''),
                url=data.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                content=data.get('text', data.get('title', '')),
                source='hackernews',
                score=data.get('score', 0),
                comments=data.get('descendants', 0),
                timestamp=data.get('time', time.time()),
                metadata={'hn_id': story_id}
            )
        except Exception as e:
            logger.error(f"Failed to fetch HN story {story_id}: {e}")
            return None


class ArxivScraper:
    """Scrape recent AI/ML papers from ArXiv"""
    
    API_BASE = "http://export.arxiv.org/api/query"
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    
    async def get_recent_papers(
        self, 
        categories: List[str] = None,
        limit: int = 30
    ) -> List[NewsItem]:
        """Get recent papers from ArXiv"""
        if categories is None:
            categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
        
        items = []
        
        try:
            # Build query
            cat_query = '+OR+'.join([f'cat:{cat}' for cat in categories])
            params = {
                'search_query': cat_query,
                'start': 0,
                'max_results': limit,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.API_BASE}?search_query={cat_query}&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
            
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return items
                xml_content = await resp.text()
            
            # Parse XML (basic parsing without lxml)
            items = self._parse_arxiv_xml(xml_content)
            
        except Exception as e:
            logger.error(f"ArXiv scraper error: {e}")
        
        return items
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[NewsItem]:
        """Parse ArXiv API response"""
        items = []
        
        if not BS4_AVAILABLE:
            return items
        
        soup = BeautifulSoup(xml_content, 'html.parser')
        entries = soup.find_all('entry')
        
        for entry in entries:
            try:
                title = entry.find('title')
                title = title.text.strip() if title else ''
                
                summary = entry.find('summary')
                summary = summary.text.strip() if summary else ''
                
                link = entry.find('id')
                url = link.text.strip() if link else ''
                
                published = entry.find('published')
                pub_time = published.text if published else ''
                
                # Extract categories
                categories = []
                for cat in entry.find_all('category'):
                    if cat.get('term'):
                        categories.append(cat.get('term'))
                
                items.append(NewsItem(
                    title=title,
                    url=url,
                    content=summary,
                    source='arxiv',
                    score=0,
                    comments=0,
                    metadata={'categories': categories, 'published': pub_time}
                ))
            except Exception as e:
                logger.error(f"Failed to parse ArXiv entry: {e}")
        
        return items


class GitHubTrendingScraper:
    """Scrape trending repositories from GitHub"""
    
    TRENDING_URL = "https://github.com/trending"
    API_BASE = "https://api.github.com"
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    
    async def get_trending_repos(
        self,
        language: Optional[str] = None,
        since: str = "daily",
        limit: int = 30
    ) -> List[NewsItem]:
        """Get trending repositories"""
        items = []
        
        try:
            # Use GitHub API to search for recently created/updated repos
            # (Trending page doesn't have an official API)
            query = "stars:>100 pushed:>2024-01-01"
            if language:
                query += f" language:{language}"
            
            headers = {'Accept': 'application/vnd.github.v3+json'}
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            url = f"{self.API_BASE}/search/repositories"
            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    return items
                data = await resp.json()
            
            for repo in data.get('items', [])[:limit]:
                items.append(NewsItem(
                    title=f"{repo.get('full_name', '')}: {repo.get('description', '')}",
                    url=repo.get('html_url', ''),
                    content=repo.get('description', '') or repo.get('full_name', ''),
                    source='github',
                    score=repo.get('stargazers_count', 0),
                    comments=repo.get('forks_count', 0),
                    metadata={
                        'language': repo.get('language'),
                        'topics': repo.get('topics', []),
                        'stars': repo.get('stargazers_count', 0)
                    }
                ))
                
        except Exception as e:
            logger.error(f"GitHub scraper error: {e}")
        
        return items


# ============================================================================
# PATTERN EXTRACTOR
# ============================================================================

class PatternExtractor:
    """Extract patterns from news items using NLP"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        # Domain keywords
        self.domain_map = {
            'technology': ['ai', 'machine learning', 'software', 'programming', 'api', 'cloud', 'neural', 'algorithm', 'model', 'llm', 'gpt'],
            'science': ['research', 'study', 'paper', 'experiment', 'physics', 'biology', 'quantum', 'discovery'],
            'business': ['startup', 'funding', 'company', 'market', 'revenue', 'growth', 'investment'],
            'general': []
        }
    
    def _load_model(self):
        """Load sentence transformer model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
    
    def extract_patterns(self, news_items: List[NewsItem]) -> List[ExtractedPattern]:
        """Extract patterns from news items"""
        patterns = []
        
        for item in news_items:
            try:
                pattern = self._extract_from_item(item)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.error(f"Pattern extraction failed: {e}")
        
        return patterns
    
    def _extract_from_item(self, item: NewsItem) -> Optional[ExtractedPattern]:
        """Extract pattern from a single news item"""
        # Skip if content is too short
        if len(item.content) < 50:
            return None
        
        # Generate question from title
        question = self._generate_question(item.title, item.source)
        
        # Clean and truncate content
        response = self._clean_content(item.content)
        if len(response) < 50:
            return None
        
        # Detect domain
        domain = self._detect_domain(item.title + ' ' + item.content)
        
        # Extract keywords
        keywords = self._extract_keywords(item.title + ' ' + item.content)
        
        # Extract topics
        topics = self._extract_topics(item, domain)
        
        # Generate embedding
        embedding = None
        if self.model:
            try:
                emb = self.model.encode(question + ' ' + response[:500])
                embedding = emb.tolist()
            except:
                pass
        
        # Calculate confidence based on source and engagement
        confidence = self._calculate_confidence(item)
        
        return ExtractedPattern(
            id=str(uuid.uuid4()),
            pattern=question,
            response=response,
            domain=domain,
            topics=topics,
            keywords=keywords,
            source_url=item.url,
            source_title=item.title,
            source_type=item.source,
            confidence=confidence,
            embedding=embedding
        )
    
    def _generate_question(self, title: str, source: str) -> str:
        """Generate a question from the title"""
        title = title.strip()
        
        # If title is already a question, use it
        if title.endswith('?'):
            return title
        
        # Generate appropriate question based on source
        if source == 'arxiv':
            return f"What is {title}?"
        elif source == 'github':
            return f"What is {title.split(':')[0] if ':' in title else title}?"
        else:
            # For news, try to make it informative
            return f"What is {title}?"
    
    def _clean_content(self, content: str) -> str:
        """Clean and truncate content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove URLs
        content = re.sub(r'http\S+', '', content)
        
        # Truncate to reasonable length
        if len(content) > 1000:
            # Try to cut at sentence boundary
            sentences = content[:1000].split('.')
            if len(sentences) > 1:
                content = '.'.join(sentences[:-1]) + '.'
            else:
                content = content[:1000] + '...'
        
        return content
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text"""
        text_lower = text.lower()
        
        scores = {}
        for domain, keywords in self.domain_map.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else 'technology'  # Default to technology for tech news
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with',
            'about', 'to', 'from', 'in', 'on', 'up', 'down', 'out', 'into'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _extract_topics(self, item: NewsItem, domain: str) -> List[str]:
        """Extract topics from news item"""
        topics = [domain]
        
        # Add source-specific topics
        if item.source == 'arxiv':
            categories = item.metadata.get('categories', [])
            topics.extend(categories[:3])
        elif item.source == 'github':
            gh_topics = item.metadata.get('topics', [])
            topics.extend(gh_topics[:3])
            if item.metadata.get('language'):
                topics.append(item.metadata['language'].lower())
        
        return list(set(topics))[:5]
    
    def _calculate_confidence(self, item: NewsItem) -> float:
        """Calculate confidence score based on source and engagement"""
        base_confidence = {
            'arxiv': 0.85,  # Academic papers are more reliable
            'hackernews': 0.70,
            'github': 0.75
        }.get(item.source, 0.65)
        
        # Boost based on engagement
        if item.score > 1000:
            base_confidence = min(base_confidence + 0.1, 0.95)
        elif item.score > 100:
            base_confidence = min(base_confidence + 0.05, 0.90)
        
        return base_confidence


# ============================================================================
# DEDUPLICATOR
# ============================================================================

class PatternDeduplicator:
    """Remove duplicate or near-duplicate patterns"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
    
    def deduplicate(self, patterns: List[ExtractedPattern]) -> List[ExtractedPattern]:
        """Remove duplicates from pattern list"""
        unique = []
        
        for pattern in patterns:
            # Check exact hash
            content_hash = self._hash_pattern(pattern)
            if content_hash in self.seen_hashes:
                continue
            
            # Check semantic similarity if embeddings available
            is_duplicate = False
            if pattern.embedding and SENTENCE_TRANSFORMERS_AVAILABLE:
                for existing in unique:
                    if existing.embedding:
                        sim = self._cosine_similarity(pattern.embedding, existing.embedding)
                        if sim > self.similarity_threshold:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                self.seen_hashes.add(content_hash)
                unique.append(pattern)
        
        return unique
    
    def _hash_pattern(self, pattern: ExtractedPattern) -> str:
        """Generate hash for pattern"""
        content = f"{pattern.pattern}|{pattern.response[:200]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================================
# DAILY PATTERN UPDATER
# ============================================================================

class DailyPatternUpdater:
    """
    Automated daily pattern updater
    
    Runs at 2 AM to:
    1. Scrape tech/AI news from HN, ArXiv, GitHub
    2. Extract patterns using sentence-transformers
    3. Deduplicate and filter
    4. Add to pattern database
    """
    
    def __init__(
        self,
        pattern_db=None,
        run_time: str = "02:00",
        max_patterns_per_run: int = 100
    ):
        self.pattern_db = pattern_db
        self.run_time = run_time
        self.max_patterns_per_run = max_patterns_per_run
        
        self.extractor = PatternExtractor()
        self.deduplicator = PatternDeduplicator()
        
        self._running = False
        self._thread = None
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'total_items_scraped': 0,
            'total_patterns_extracted': 0,
            'total_patterns_added': 0,
            'last_run': None,
            'last_run_duration': 0,
            'errors': []
        }
    
    def set_pattern_database(self, db):
        """Set pattern database"""
        self.pattern_db = db
    
    async def run_update(self) -> Dict[str, Any]:
        """Run a single update cycle"""
        start_time = time.time()
        self.stats['total_runs'] += 1
        
        logger.info("Starting daily pattern update...")
        
        result = {
            'success': True,
            'items_scraped': 0,
            'patterns_extracted': 0,
            'patterns_added': 0,
            'errors': []
        }
        
        try:
            # Create session
            async with aiohttp.ClientSession() as session:
                # Scrape all sources
                all_items = []
                
                # Hacker News
                logger.info("Scraping Hacker News...")
                hn_scraper = HackerNewsScraper(session)
                hn_items = await hn_scraper.get_top_stories(limit=30)
                all_items.extend(hn_items)
                logger.info(f"  Got {len(hn_items)} HN stories")
                
                # ArXiv
                logger.info("Scraping ArXiv...")
                arxiv_scraper = ArxivScraper(session)
                arxiv_items = await arxiv_scraper.get_recent_papers(limit=30)
                all_items.extend(arxiv_items)
                logger.info(f"  Got {len(arxiv_items)} ArXiv papers")
                
                # GitHub Trending
                logger.info("Scraping GitHub...")
                gh_scraper = GitHubTrendingScraper(session)
                gh_items = await gh_scraper.get_trending_repos(limit=30)
                all_items.extend(gh_items)
                logger.info(f"  Got {len(gh_items)} GitHub repos")
            
            result['items_scraped'] = len(all_items)
            self.stats['total_items_scraped'] += len(all_items)
            
            # Extract patterns
            logger.info(f"Extracting patterns from {len(all_items)} items...")
            patterns = self.extractor.extract_patterns(all_items)
            logger.info(f"  Extracted {len(patterns)} patterns")
            
            # Deduplicate
            logger.info("Deduplicating patterns...")
            patterns = self.deduplicator.deduplicate(patterns)
            logger.info(f"  {len(patterns)} unique patterns")
            
            result['patterns_extracted'] = len(patterns)
            self.stats['total_patterns_extracted'] += len(patterns)
            
            # Add to database
            if self.pattern_db and patterns:
                added = 0
                for pattern in patterns[:self.max_patterns_per_run]:
                    try:
                        pattern_dict = pattern.to_pattern_dict()
                        if hasattr(self.pattern_db, 'add_pattern'):
                            self.pattern_db.add_pattern(pattern_dict)
                            added += 1
                    except Exception as e:
                        logger.error(f"Failed to add pattern: {e}")
                        result['errors'].append(str(e))
                
                result['patterns_added'] = added
                self.stats['total_patterns_added'] += added
                logger.info(f"  Added {added} patterns to database")
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            result['success'] = False
            result['errors'].append(str(e))
            self.stats['errors'].append({
                'time': time.time(),
                'error': str(e)
            })
        
        # Update stats
        duration = time.time() - start_time
        self.stats['last_run'] = time.time()
        self.stats['last_run_duration'] = duration
        
        logger.info(f"Update completed in {duration:.2f}s")
        
        return result
    
    def run_update_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for run_update"""
        return asyncio.run(self.run_update())
    
    def start_scheduler(self):
        """Start the scheduled updater"""
        if not SCHEDULE_AVAILABLE:
            logger.error("Schedule library not available")
            return
        
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        # Schedule daily run
        schedule.every().day.at(self.run_time).do(self.run_update_sync)
        
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Scheduler started. Will run daily at {self.run_time}")
    
    def _scheduler_loop(self):
        """Scheduler loop running in background thread"""
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics"""
        return {
            **self.stats,
            'scheduler_running': self._running,
            'next_run': self.run_time
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Test the daily updater"""
    print("=" * 60)
    print("Daily Pattern Updater Test")
    print("=" * 60)
    
    # Create updater (without database for testing)
    updater = DailyPatternUpdater(
        pattern_db=None,
        max_patterns_per_run=50
    )
    
    # Run single update
    print("\nRunning update...")
    result = await updater.run_update()
    
    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Items scraped: {result['items_scraped']}")
    print(f"  Patterns extracted: {result['patterns_extracted']}")
    print(f"  Patterns added: {result['patterns_added']}")
    
    if result['errors']:
        print(f"  Errors: {result['errors']}")
    
    print(f"\nStatistics: {updater.get_statistics()}")


if __name__ == "__main__":
    asyncio.run(main())
