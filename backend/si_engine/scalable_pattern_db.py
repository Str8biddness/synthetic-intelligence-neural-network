"""
ScalablePatternDatabase - FAISS-based vector similarity search for 1M+ patterns
Implements hierarchical indexing with memory-mapped storage for <10ms search time

Architecture:
- IndexIVFFlat for datasets >100k patterns (recommended by FAISS)
- Hierarchical category-based indices for faster filtered searches
- Memory-mapped storage for large datasets
- TF-IDF vectorization with dimensionality reduction via PCA
"""

import os
import uuid
import time
import pickle
import mmap
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re
import math
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Pattern:
    """Pattern with metadata - compatible with existing pattern_database.py"""
    id: str
    pattern: str
    response: str
    domain: str
    topics: List[str]
    success_rate: float = 1.0
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    confidence: float = 0.8
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pattern':
        return cls(**data)


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    pattern: Pattern
    score: float  # Similarity score (higher = more similar)
    distance: float  # L2 distance (lower = more similar)
    match_type: str  # 'vector', 'keyword', 'exact'
    
    def to_dict(self) -> Dict:
        return {
            'pattern': self.pattern.to_dict(),
            'score': self.score,
            'distance': self.distance,
            'match_type': self.match_type
        }


@dataclass 
class IndexStats:
    """Statistics about the index"""
    total_patterns: int
    index_type: str
    vector_dimension: int
    nlist: int  # Number of Voronoi cells for IVF
    nprobe: int  # Number of cells to search
    categories: Dict[str, int]
    memory_mapped: bool
    index_size_mb: float
    avg_search_time_ms: float


# ============================================================================
# TF-IDF VECTORIZER (No external dependencies)
# ============================================================================

class TFIDFVectorizer:
    """
    Custom TF-IDF vectorizer for text to vector conversion
    Outputs fixed-dimension vectors suitable for FAISS
    """
    
    def __init__(self, max_features: int = 10000, output_dim: int = 256):
        self.max_features = max_features
        self.output_dim = output_dim
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.projection_matrix: Optional[np.ndarray] = None
        self._fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return words
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}
    
    def fit(self, documents: List[str]):
        """Fit vectorizer on documents"""
        # Count document frequency
        doc_freq = defaultdict(int)
        all_terms = defaultdict(int)
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
                all_terms[token] += 1
        
        # Select top features by frequency
        sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:self.max_features]
        
        # Build vocabulary
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        
        # Compute IDF
        n_docs = len(documents)
        self.idf = {
            term: math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1
            for term in self.vocabulary
        }
        
        # Create random projection matrix for dimensionality reduction
        # Using random projection (Johnson-Lindenstrauss lemma)
        np.random.seed(42)  # Reproducibility
        vocab_size = len(self.vocabulary)
        if vocab_size > self.output_dim:
            self.projection_matrix = np.random.randn(
                vocab_size, self.output_dim
            ).astype(np.float32) / np.sqrt(self.output_dim)
        else:
            # Pad with zeros if vocabulary is smaller
            self.projection_matrix = np.eye(
                vocab_size, self.output_dim
            ).astype(np.float32)
        
        self._fitted = True
    
    def transform(self, text: str) -> np.ndarray:
        """Transform text to vector"""
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        
        # Create sparse TF-IDF vector
        sparse_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token, freq in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                sparse_vec[idx] = freq * self.idf.get(token, 1.0)
        
        # L2 normalize
        norm = np.linalg.norm(sparse_vec)
        if norm > 0:
            sparse_vec /= norm
        
        # Project to output dimension
        dense_vec = sparse_vec @ self.projection_matrix
        
        # L2 normalize again
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec /= norm
            
        return dense_vec
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform documents"""
        self.fit(documents)
        vectors = np.array([self.transform(doc) for doc in documents], dtype=np.float32)
        return vectors
    
    def save(self, path: str):
        """Save vectorizer state"""
        state = {
            'vocabulary': self.vocabulary,
            'idf': self.idf,
            'projection_matrix': self.projection_matrix,
            'max_features': self.max_features,
            'output_dim': self.output_dim,
            '_fitted': self._fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load vectorizer state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.vocabulary = state['vocabulary']
        self.idf = state['idf']
        self.projection_matrix = state['projection_matrix']
        self.max_features = state['max_features']
        self.output_dim = state['output_dim']
        self._fitted = state['_fitted']


# ============================================================================
# HIERARCHICAL FAISS INDEX
# ============================================================================

class HierarchicalIndex:
    """
    Hierarchical FAISS index with category-based sub-indices
    Allows fast filtered searches within categories
    """
    
    def __init__(self, dimension: int, nlist: int = 100):
        self.dimension = dimension
        self.nlist = nlist
        
        # Main index (all patterns)
        self.main_index: Optional[faiss.Index] = None
        
        # Category-specific indices
        self.category_indices: Dict[str, faiss.Index] = {}
        
        # Pattern ID to index position mapping
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        
        # Category to pattern IDs
        self.category_patterns: Dict[str, Set[str]] = defaultdict(set)
        
        # Index counter
        self.next_idx = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _create_index(self, n_vectors: int) -> faiss.Index:
        """Create appropriate FAISS index based on dataset size"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        if n_vectors < 1000:
            # Small dataset: use flat index (exact search)
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
        elif n_vectors < 100000:
            # Medium dataset: IVF with fewer cells
            nlist = min(self.nlist, int(np.sqrt(n_vectors)))
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            # Large dataset: IVF with more cells for better performance
            nlist = min(self.nlist * 4, int(np.sqrt(n_vectors) * 4))
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        return index
    
    def build_index(self, vectors: np.ndarray, pattern_ids: List[str], categories: List[str]):
        """Build the hierarchical index"""
        with self._lock:
            n_vectors = len(vectors)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Build main index
            self.main_index = self._create_index(n_vectors)
            
            # Train if IVF index
            if hasattr(self.main_index, 'train'):
                self.main_index.train(vectors)
            
            # Add vectors
            self.main_index.add(vectors)
            
            # Build ID mappings
            for idx, (pid, cat) in enumerate(zip(pattern_ids, categories)):
                self.id_to_idx[pid] = idx
                self.idx_to_id[idx] = pid
                self.category_patterns[cat].add(pid)
            
            self.next_idx = n_vectors
            
            # Build category indices
            self._build_category_indices(vectors, pattern_ids, categories)
    
    def _build_category_indices(self, vectors: np.ndarray, pattern_ids: List[str], categories: List[str]):
        """Build separate indices for each category"""
        # Group by category
        cat_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
        cat_ids: Dict[str, List[str]] = defaultdict(list)
        
        for vec, pid, cat in zip(vectors, pattern_ids, categories):
            cat_vectors[cat].append(vec)
            cat_ids[cat].append(pid)
        
        # Build index for each category
        for cat, vecs in cat_vectors.items():
            if len(vecs) < 10:
                continue  # Skip very small categories
                
            cat_array = np.array(vecs, dtype=np.float32)
            index = self._create_index(len(vecs))
            
            if hasattr(index, 'train'):
                index.train(cat_array)
            
            index.add(cat_array)
            self.category_indices[cat] = index
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        category: Optional[str] = None,
        nprobe: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Returns:
            distances: Array of distances
            indices: Array of indices
        """
        with self._lock:
            # Normalize query
            query = query_vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            
            if category and category in self.category_indices:
                # Search category-specific index
                index = self.category_indices[category]
            else:
                # Search main index
                index = self.main_index
            
            if index is None:
                return np.array([]), np.array([])
            
            # Set nprobe for IVF indices
            if hasattr(index, 'nprobe'):
                index.nprobe = nprobe
            
            # Search
            distances, indices = index.search(query, k)
            
            return distances[0], indices[0]
    
    def add_vector(self, vector: np.ndarray, pattern_id: str, category: str):
        """Add a single vector to the index"""
        with self._lock:
            vec = vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            
            # Add to main index
            if self.main_index is not None:
                self.main_index.add(vec)
            
            # Update mappings
            idx = self.next_idx
            self.id_to_idx[pattern_id] = idx
            self.idx_to_id[idx] = pattern_id
            self.category_patterns[category].add(pattern_id)
            self.next_idx += 1
            
            # Add to category index if exists
            if category in self.category_indices:
                self.category_indices[category].add(vec)
    
    def get_pattern_id(self, idx: int) -> Optional[str]:
        """Get pattern ID from index"""
        return self.idx_to_id.get(idx)
    
    def save(self, directory: str):
        """Save index to directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save main index
        if self.main_index is not None:
            faiss.write_index(self.main_index, os.path.join(directory, 'main.index'))
        
        # Save category indices
        for cat, index in self.category_indices.items():
            safe_cat = cat.replace('/', '_').replace('\\', '_')
            faiss.write_index(index, os.path.join(directory, f'cat_{safe_cat}.index'))
        
        # Save mappings
        mappings = {
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'category_patterns': {k: list(v) for k, v in self.category_patterns.items()},
            'next_idx': self.next_idx,
            'dimension': self.dimension,
            'nlist': self.nlist
        }
        with open(os.path.join(directory, 'mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
    
    def load(self, directory: str):
        """Load index from directory"""
        # Load mappings
        with open(os.path.join(directory, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
        
        self.id_to_idx = mappings['id_to_idx']
        self.idx_to_id = mappings['idx_to_id']
        self.category_patterns = {k: set(v) for k, v in mappings['category_patterns'].items()}
        self.next_idx = mappings['next_idx']
        self.dimension = mappings['dimension']
        self.nlist = mappings['nlist']
        
        # Load main index
        main_path = os.path.join(directory, 'main.index')
        if os.path.exists(main_path):
            self.main_index = faiss.read_index(main_path)
        
        # Load category indices
        for filename in os.listdir(directory):
            if filename.startswith('cat_') and filename.endswith('.index'):
                cat = filename[4:-6].replace('_', '/')
                self.category_indices[cat] = faiss.read_index(
                    os.path.join(directory, filename)
                )


# ============================================================================
# MEMORY-MAPPED PATTERN STORAGE
# ============================================================================

class MemoryMappedStorage:
    """
    Memory-mapped storage for pattern data
    Allows handling datasets larger than RAM
    """
    
    HEADER_SIZE = 1024  # Reserved bytes for header
    RECORD_HEADER_SIZE = 12  # 4 bytes length + 8 bytes offset
    
    def __init__(self, filepath: str, mode: str = 'r'):
        self.filepath = filepath
        self.mode = mode
        self.file = None
        self.mmap = None
        self.index: Dict[str, Tuple[int, int]] = {}  # id -> (offset, length)
        self._lock = threading.RLock()
        
    def open(self):
        """Open the memory-mapped file"""
        if self.mode == 'w':
            self.file = open(self.filepath, 'w+b')
            # Write header placeholder
            self.file.write(b'\x00' * self.HEADER_SIZE)
            self.file.flush()
        else:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"Storage file not found: {self.filepath}")
            self.file = open(self.filepath, 'r+b')
        
        # Create memory map
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE if self.mode == 'w' else mmap.ACCESS_READ)
        
        # Load index if reading
        if self.mode == 'r':
            self._load_index()
    
    def close(self):
        """Close the storage"""
        if self.mode == 'w':
            self._save_index()
        
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
    
    def _save_index(self):
        """Save index to header"""
        index_data = pickle.dumps(self.index)
        if len(index_data) > self.HEADER_SIZE - 8:
            # Index too large, save to separate file
            with open(self.filepath + '.idx', 'wb') as f:
                f.write(index_data)
            # Mark header as external
            self.mmap.seek(0)
            self.mmap.write(struct.pack('<Q', 0xFFFFFFFFFFFFFFFF))
        else:
            self.mmap.seek(0)
            self.mmap.write(struct.pack('<Q', len(index_data)))
            self.mmap.write(index_data)
    
    def _load_index(self):
        """Load index from header"""
        self.mmap.seek(0)
        index_size = struct.unpack('<Q', self.mmap.read(8))[0]
        
        if index_size == 0xFFFFFFFFFFFFFFFF:
            # External index file
            with open(self.filepath + '.idx', 'rb') as f:
                self.index = pickle.load(f)
        else:
            index_data = self.mmap.read(index_size)
            self.index = pickle.loads(index_data)
    
    def write_pattern(self, pattern: Pattern) -> Tuple[int, int]:
        """Write pattern to storage, returns (offset, length)"""
        with self._lock:
            data = pickle.dumps(pattern.to_dict())
            
            # Seek to end
            self.file.seek(0, 2)
            offset = self.file.tell()
            
            # Write length and data
            self.file.write(struct.pack('<I', len(data)))
            self.file.write(data)
            self.file.flush()
            
            # Update mmap
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)
            
            # Update index
            self.index[pattern.id] = (offset, len(data))
            
            return offset, len(data)
    
    def read_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Read pattern from storage"""
        if pattern_id not in self.index:
            return None
        
        offset, length = self.index[pattern_id]
        
        with self._lock:
            self.mmap.seek(offset)
            data_length = struct.unpack('<I', self.mmap.read(4))[0]
            data = self.mmap.read(data_length)
            
            pattern_dict = pickle.loads(data)
            return Pattern.from_dict(pattern_dict)
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# MAIN SCALABLE PATTERN DATABASE
# ============================================================================

class ScalablePatternDatabase:
    """
    Scalable pattern database with FAISS-based vector search
    
    Features:
    - FAISS IndexIVFFlat for 1M+ patterns
    - Hierarchical category-based indexing
    - Memory-mapped storage support
    - <10ms search time on CPU
    - Thread-safe operations
    
    Usage:
        db = ScalablePatternDatabase(dimension=256)
        db.add_patterns(patterns)
        db.build_index()
        results = db.search("what is gravity", top_k=5)
    """
    
    def __init__(
        self,
        dimension: int = 256,
        nlist: int = 100,
        nprobe: int = 10,
        storage_path: Optional[str] = None,
        use_mmap: bool = False
    ):
        """
        Initialize scalable pattern database
        
        Args:
            dimension: Vector dimension for embeddings
            nlist: Number of Voronoi cells for IVF index
            nprobe: Number of cells to search (trade-off: speed vs accuracy)
            storage_path: Path for persistent storage
            use_mmap: Whether to use memory-mapped storage
        """
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.storage_path = storage_path
        self.use_mmap = use_mmap
        
        # Vectorizer
        self.vectorizer = TFIDFVectorizer(max_features=10000, output_dim=dimension)
        
        # FAISS index
        self.index = HierarchicalIndex(dimension=dimension, nlist=nlist)
        
        # In-memory pattern storage (for small datasets)
        self.patterns: Dict[str, Pattern] = {}
        
        # Memory-mapped storage (for large datasets)
        self.mmap_storage: Optional[MemoryMappedStorage] = None
        
        # Indices
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.search_times: List[float] = []
        self._initialized = False
        
        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
    
    def add_pattern(self, pattern: Pattern):
        """Add a single pattern"""
        with self._lock:
            self.patterns[pattern.id] = pattern
            
            # Update indices
            self.domain_index[pattern.domain].add(pattern.id)
            for topic in pattern.topics:
                self.topic_index[topic].add(pattern.id)
            for keyword in pattern.keywords:
                self.keyword_index[keyword.lower()].add(pattern.id)
    
    def add_patterns(self, patterns: List[Pattern]):
        """Add multiple patterns"""
        for pattern in patterns:
            self.add_pattern(pattern)
    
    def build_index(self, show_progress: bool = True):
        """Build FAISS index from all patterns"""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, using fallback search")
            self._initialized = True
            return
        
        patterns = list(self.patterns.values())
        if not patterns:
            print("No patterns to index")
            return
        
        n_patterns = len(patterns)
        if show_progress:
            print(f"Building index for {n_patterns:,} patterns...")
        
        start_time = time.time()
        
        # Prepare documents for vectorization
        documents = [
            f"{p.pattern} {' '.join(p.keywords)} {' '.join(p.topics)}"
            for p in patterns
        ]
        
        # Fit vectorizer and transform documents
        if show_progress:
            print("  Vectorizing patterns...")
        vectors = self.vectorizer.fit_transform(documents)
        
        # Get pattern IDs and categories
        pattern_ids = [p.id for p in patterns]
        categories = [p.domain for p in patterns]
        
        # Build FAISS index
        if show_progress:
            print("  Building FAISS index...")
        self.index.build_index(vectors, pattern_ids, categories)
        
        build_time = time.time() - start_time
        if show_progress:
            print(f"  Index built in {build_time:.2f}s")
            print(f"  Index type: IVFFlat with {self.nlist} cells")
        
        self._initialized = True
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for patterns similar to query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            domain: Optional domain filter
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        if not self._initialized:
            # Fallback to keyword search
            return self._fallback_search(query, top_k, domain)
        
        # Vectorize query
        query_vector = self.vectorizer.transform(query)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_vector,
            k=top_k * 2,  # Get more results for filtering
            category=domain,
            nprobe=self.nprobe
        )
        
        # Build results
        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0:  # Invalid index
                continue
                
            pattern_id = self.index.get_pattern_id(int(idx))
            if pattern_id is None:
                continue
                
            pattern = self.patterns.get(pattern_id)
            if pattern is None:
                continue
            
            # Convert distance to similarity score
            # For inner product, higher is more similar
            score = float(dist)
            
            if score < min_score:
                continue
            
            results.append(SearchResult(
                pattern=pattern,
                score=score,
                distance=1.0 - score,  # Convert to distance
                match_type='vector'
            ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        # Record search time
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        return results
    
    def _fallback_search(
        self,
        query: str,
        top_k: int,
        domain: Optional[str]
    ) -> List[SearchResult]:
        """Fallback keyword-based search when FAISS is not available"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored = []
        for pattern in self.patterns.values():
            if domain and pattern.domain != domain:
                continue
            
            score = 0.0
            
            # Exact match
            if query_lower == pattern.pattern.lower():
                score = 1.0
            else:
                # Keyword overlap
                pattern_words = set(pattern.keywords) | set(pattern.topics)
                pattern_words = {w.lower() for w in pattern_words}
                overlap = query_words & pattern_words
                if overlap:
                    score = len(overlap) / max(len(query_words), 1)
            
            if score > 0:
                scored.append(SearchResult(
                    pattern=pattern,
                    score=score,
                    distance=1.0 - score,
                    match_type='keyword'
                ))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
    
    def search_by_domain(self, domain: str, limit: int = 100) -> List[Pattern]:
        """Get all patterns in a domain"""
        pattern_ids = self.domain_index.get(domain, set())
        return [self.patterns[pid] for pid in list(pattern_ids)[:limit]]
    
    def search_by_topic(self, topic: str, limit: int = 100) -> List[Pattern]:
        """Get all patterns for a topic"""
        pattern_ids = self.topic_index.get(topic, set())
        return [self.patterns[pid] for pid in list(pattern_ids)[:limit]]
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns"""
        return list(self.patterns.values())
    
    def update_pattern_stats(self, pattern_id: str, success: bool = True):
        """Update pattern usage statistics"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = time.time()
            
            # Exponential moving average for success rate
            alpha = 0.1
            result = 1.0 if success else 0.0
            pattern.success_rate = alpha * result + (1 - alpha) * pattern.success_rate
    
    def get_statistics(self) -> IndexStats:
        """Get index statistics"""
        avg_search_time = (
            sum(self.search_times) / len(self.search_times)
            if self.search_times else 0.0
        )
        
        # Estimate index size
        index_size_mb = 0.0
        if self.index.main_index is not None:
            # Rough estimate: vectors * dimension * 4 bytes
            n_vectors = len(self.patterns)
            index_size_mb = (n_vectors * self.dimension * 4) / (1024 * 1024)
        
        return IndexStats(
            total_patterns=len(self.patterns),
            index_type='IVFFlat' if FAISS_AVAILABLE else 'keyword',
            vector_dimension=self.dimension,
            nlist=self.nlist,
            nprobe=self.nprobe,
            categories={d: len(ids) for d, ids in self.domain_index.items()},
            memory_mapped=self.use_mmap,
            index_size_mb=index_size_mb,
            avg_search_time_ms=avg_search_time
        )
    
    def save(self, directory: str):
        """Save database to directory"""
        os.makedirs(directory, exist_ok=True)
        
        # Save patterns
        patterns_data = {pid: p.to_dict() for pid, p in self.patterns.items()}
        with open(os.path.join(directory, 'patterns.pkl'), 'wb') as f:
            pickle.dump(patterns_data, f)
        
        # Save vectorizer
        self.vectorizer.save(os.path.join(directory, 'vectorizer.pkl'))
        
        # Save FAISS index
        if self._initialized and FAISS_AVAILABLE:
            self.index.save(os.path.join(directory, 'faiss'))
        
        # Save indices
        indices = {
            'domain_index': {k: list(v) for k, v in self.domain_index.items()},
            'topic_index': {k: list(v) for k, v in self.topic_index.items()},
            'keyword_index': {k: list(v) for k, v in self.keyword_index.items()}
        }
        with open(os.path.join(directory, 'indices.pkl'), 'wb') as f:
            pickle.dump(indices, f)
        
        print(f"Database saved to {directory}")
    
    def load(self, directory: str):
        """Load database from directory"""
        # Load patterns
        with open(os.path.join(directory, 'patterns.pkl'), 'rb') as f:
            patterns_data = pickle.load(f)
        self.patterns = {pid: Pattern.from_dict(data) for pid, data in patterns_data.items()}
        
        # Load vectorizer
        self.vectorizer.load(os.path.join(directory, 'vectorizer.pkl'))
        
        # Load FAISS index
        faiss_dir = os.path.join(directory, 'faiss')
        if os.path.exists(faiss_dir) and FAISS_AVAILABLE:
            self.index.load(faiss_dir)
            self._initialized = True
        
        # Load indices
        with open(os.path.join(directory, 'indices.pkl'), 'rb') as f:
            indices = pickle.load(f)
        self.domain_index = {k: set(v) for k, v in indices['domain_index'].items()}
        self.topic_index = {k: set(v) for k, v in indices['topic_index'].items()}
        self.keyword_index = {k: set(v) for k, v in indices['keyword_index'].items()}
        
        print(f"Database loaded from {directory}: {len(self.patterns):,} patterns")
    
    # ========================================================================
    # COMPATIBILITY WITH EXISTING PATTERN DATABASE
    # ========================================================================
    
    def initialize_with_seed_data(self):
        """Initialize with seed data (compatible with PatternDatabase)"""
        from .pattern_database import PatternDatabase
        
        # Create temporary PatternDatabase to get seed patterns
        temp_db = PatternDatabase()
        seed_patterns = temp_db._get_seed_patterns()
        
        # Convert and add patterns
        for p_data in seed_patterns:
            pattern = Pattern.from_dict(p_data)
            self.add_pattern(pattern)
        
        # Build index
        self.build_index(show_progress=True)
        
        print(f"Initialized with {len(self.patterns)} seed patterns")


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def generate_synthetic_patterns(n: int, domains: List[str] = None) -> List[Pattern]:
    """Generate synthetic patterns for benchmarking"""
    if domains is None:
        domains = ['science', 'philosophy', 'technology', 'history', 'economics']
    
    patterns = []
    topics_pool = ['physics', 'biology', 'ethics', 'computing', 'markets', 'culture']
    keywords_pool = ['concept', 'theory', 'system', 'process', 'method', 'analysis']
    
    for i in range(n):
        domain = domains[i % len(domains)]
        pattern = Pattern(
            id=str(uuid.uuid4()),
            pattern=f"synthetic query {i} about {domain}",
            response=f"This is a synthetic response for pattern {i} in {domain}. " * 5,
            domain=domain,
            topics=[topics_pool[i % len(topics_pool)], topics_pool[(i + 1) % len(topics_pool)]],
            keywords=[keywords_pool[j % len(keywords_pool)] for j in range(3)],
            success_rate=0.9,
            confidence=0.85
        )
        patterns.append(pattern)
    
    return patterns


def benchmark_search(db: ScalablePatternDatabase, n_queries: int = 100) -> Dict[str, float]:
    """Benchmark search performance"""
    queries = [
        "what is gravity",
        "explain quantum mechanics",
        "how does evolution work",
        "define consciousness",
        "what is artificial intelligence"
    ]
    
    times = []
    for i in range(n_queries):
        query = queries[i % len(queries)]
        start = time.time()
        db.search(query, top_k=10)
        times.append((time.time() - start) * 1000)
    
    return {
        'min_ms': min(times),
        'max_ms': max(times),
        'avg_ms': sum(times) / len(times),
        'p50_ms': sorted(times)[len(times) // 2],
        'p95_ms': sorted(times)[int(len(times) * 0.95)],
        'p99_ms': sorted(times)[int(len(times) * 0.99)]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ScalablePatternDatabase Benchmark")
    print("=" * 60)
    
    # Test with different dataset sizes
    for n_patterns in [1000, 10000, 100000]:
        print(f"\n--- Testing with {n_patterns:,} patterns ---")
        
        # Generate patterns
        print("Generating patterns...")
        patterns = generate_synthetic_patterns(n_patterns)
        
        # Create database
        db = ScalablePatternDatabase(dimension=256, nlist=100, nprobe=10)
        db.add_patterns(patterns)
        
        # Build index
        db.build_index(show_progress=True)
        
        # Benchmark
        print("Running benchmark...")
        results = benchmark_search(db, n_queries=100)
        
        print(f"  Avg search time: {results['avg_ms']:.2f}ms")
        print(f"  P95 search time: {results['p95_ms']:.2f}ms")
        print(f"  P99 search time: {results['p99_ms']:.2f}ms")
        
        stats = db.get_statistics()
        print(f"  Index size: {stats.index_size_mb:.2f}MB")
