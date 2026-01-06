"""
WebAccess - Web scraping and search capabilities for SI engine
Provides the SI with ability to fetch real-time information from the web
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import re
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

@dataclass
class WebResult:
    """Represents a web search or scrape result"""
    url: str
    title: str
    content: str
    snippet: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'snippet': self.snippet,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

class WebAccess:
    """
    Web access capabilities for the SI engine
    Handles web scraping, content extraction, and search
    """
    
    def __init__(self, timeout: int = 10, max_content_length: int = 50000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def fetch_url(self, url: str) -> Optional[WebResult]:
        """
        Fetch and parse content from a URL
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else urlparse(url).netloc
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Create snippet (first 200 chars)
            snippet = main_content[:200] + '...' if len(main_content) > 200 else main_content
            
            # Truncate content if needed
            if len(main_content) > self.max_content_length:
                main_content = main_content[:self.max_content_length] + '...'
            
            return WebResult(
                url=url,
                title=title_text,
                content=main_content,
                snippet=snippet,
                metadata={'status_code': response.status_code}
            )
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from parsed HTML"""
        # Try to find main content areas
        main_selectors = ['main', 'article', '[role="main"]', '.content', '#content']
        
        for selector in main_selectors:
            main = soup.select_one(selector)
            if main:
                text = main.get_text(separator='\n', strip=True)
                if len(text) > 100:  # Ensure we got substantial content
                    return text
        
        # Fallback: get all text
        return soup.get_text(separator='\n', strip=True)
    
    def search_web(self, query: str, num_results: int = 5) -> List[WebResult]:
        """
        Search the web for a query
        Uses DuckDuckGo HTML search (no API key required)
        """
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            for result_div in soup.find_all('div', class_='result')[:num_results]:
                try:
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        url = title_elem.get('href', '')
                        title = title_elem.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True)
                        
                        results.append(WebResult(
                            url=url,
                            title=title,
                            content=snippet,
                            snippet=snippet,
                            metadata={'source': 'duckduckgo_search'}
                        ))
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return []
    
    def extract_links(self, url: str) -> List[str]:
        """Extract all valid links from a webpage"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                absolute_url = urljoin(url, href)
                
                # Only include http/https links
                if absolute_url.startswith(('http://', 'https://')):
                    links.append(absolute_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []
    
    def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a webpage (title, description, etc.)"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            metadata = {'url': url}
            
            # Title
            title = soup.find('title')
            if title:
                metadata['title'] = title.get_text().strip()
            
            # Meta description
            description = soup.find('meta', attrs={'name': 'description'})
            if description and description.get('content'):
                metadata['description'] = description['content']
            
            # Open Graph metadata
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                metadata['og_title'] = og_title['content']
            
            og_description = soup.find('meta', property='og:description')
            if og_description and og_description.get('content'):
                metadata['og_description'] = og_description['content']
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata from {url}: {e}")
            return {'url': url, 'error': str(e)}
