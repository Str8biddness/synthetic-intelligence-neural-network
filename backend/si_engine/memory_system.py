"""
MemorySystem - Comprehensive memory and session management for SI engine
Implements short-term, long-term, episodic, and semantic memory
"""

import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    session_id: str
    content: str
    memory_type: str  # 'query', 'response', 'observation', 'pattern', 'entity'
    timestamp: float
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'content': self.content,
            'memory_type': self.memory_type,
            'timestamp': self.timestamp,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata
        }

@dataclass
class Session:
    """Represents a conversation session"""
    id: str
    user_id: Optional[str]
    created: float
    updated: float
    context: Dict[str, Any] = field(default_factory=dict)
    state: str = 'active'  # 'active', 'paused', 'archived'
    memory_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'created': self.created,
            'updated': self.updated,
            'context': self.context,
            'state': self.state,
            'memory_count': self.memory_count
        }

class MemorySystem:
    """
    Comprehensive memory management system for SI engine
    Implements multiple memory types and persistence
    """
    
    def __init__(self, db_connection=None, short_term_capacity: int = 50):
        self.db = db_connection
        self.short_term_capacity = short_term_capacity
        
        # Short-term memory (in-memory buffer)
        self.short_term: deque = deque(maxlen=short_term_capacity)
        
        # Active sessions
        self.sessions: Dict[str, Session] = {}
        
        # Memory indices for fast retrieval
        self.memory_index: Dict[str, MemoryEntry] = {}
        self.session_memories: Dict[str, List[str]] = {}
        
        logger.info(f"Memory system initialized with capacity: {short_term_capacity}")
    
    def create_session(self, user_id: Optional[str] = None) -> Session:
        """
        Create a new conversation session
        """
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            user_id=user_id,
            created=time.time(),
            updated=time.time()
        )
        
        self.sessions[session_id] = session
        self.session_memories[session_id] = []
        
        # Persist to database if available
        if self.db:
            self._persist_session(session)
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID
        """
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from database
        if self.db:
            session = self._load_session(session_id)
            if session:
                self.sessions[session_id] = session
                return session
        
        return None
    
    def store(self, session_id: str, content: str, memory_type: str,
             importance: float = 0.5, metadata: Optional[Dict] = None) -> MemoryEntry:
        """
        Store a new memory entry
        """
        memory_id = str(uuid.uuid4())
        
        memory = MemoryEntry(
            id=memory_id,
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # Add to short-term memory
        self.short_term.append(memory)
        
        # Add to indices
        self.memory_index[memory_id] = memory
        if session_id not in self.session_memories:
            self.session_memories[session_id] = []
        self.session_memories[session_id].append(memory_id)
        
        # Update session
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.updated = time.time()
            session.memory_count += 1
        
        # Persist to long-term storage
        if self.db and importance >= 0.6:  # Only persist important memories
            self._persist_memory(memory)
        
        logger.debug(f"Stored memory: {memory_id} (type: {memory_type})")
        return memory
    
    def retrieve(self, session_id: str, limit: int = 10,
                memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Retrieve recent memories from a session
        """
        if session_id not in self.session_memories:
            return []
        
        memory_ids = self.session_memories[session_id]
        memories = []
        
        for mem_id in reversed(memory_ids[-limit:]):
            if mem_id in self.memory_index:
                memory = self.memory_index[mem_id]
                if memory_type is None or memory.memory_type == memory_type:
                    memory.access_count += 1
                    memory.last_accessed = time.time()
                    memories.append(memory)
        
        return memories
    
    def search(self, query: str, session_id: Optional[str] = None,
              limit: int = 5) -> List[MemoryEntry]:
        """
        Search memories by content similarity
        Simple implementation using keyword matching
        """
        query_lower = query.lower()
        results = []
        
        # Search scope
        if session_id and session_id in self.session_memories:
            memory_ids = self.session_memories[session_id]
        else:
            memory_ids = list(self.memory_index.keys())
        
        for mem_id in memory_ids:
            if mem_id in self.memory_index:
                memory = self.memory_index[mem_id]
                if query_lower in memory.content.lower():
                    results.append((memory, self._calculate_relevance(query, memory)))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [mem for mem, _ in results[:limit]]
    
    def _calculate_relevance(self, query: str, memory: MemoryEntry) -> float:
        """
        Calculate relevance score between query and memory
        """
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words & memory_words)
        union = len(query_words | memory_words)
        similarity = intersection / union if union > 0 else 0.0
        
        # Weight by importance and recency
        recency_weight = 1.0 / (1.0 + (time.time() - memory.timestamp) / 3600)
        
        return similarity * 0.6 + memory.importance * 0.2 + recency_weight * 0.2
    
    def consolidate(self, session_id: str, threshold: float = 0.7):
        """
        Consolidate short-term memories into long-term storage
        Promotes important memories to persistent storage
        """
        if session_id not in self.session_memories:
            return
        
        memory_ids = self.session_memories[session_id]
        consolidated_count = 0
        
        for mem_id in memory_ids:
            if mem_id in self.memory_index:
                memory = self.memory_index[mem_id]
                
                # Calculate consolidation score
                score = self._calculate_consolidation_score(memory)
                
                if score >= threshold and self.db:
                    self._persist_memory(memory)
                    consolidated_count += 1
        
        logger.info(f"Consolidated {consolidated_count} memories for session {session_id}")
    
    def _calculate_consolidation_score(self, memory: MemoryEntry) -> float:
        """
        Calculate whether a memory should be consolidated to long-term
        """
        # Factors: importance, access frequency, recency
        access_score = min(memory.access_count / 10.0, 1.0)
        age_hours = (time.time() - memory.timestamp) / 3600
        recency_score = 1.0 / (1.0 + age_hours / 24)
        
        return memory.importance * 0.5 + access_score * 0.3 + recency_score * 0.2
    
    def get_context(self, session_id: str, window_size: int = 10) -> str:
        """
        Get recent conversation context as a formatted string
        """
        memories = self.retrieve(session_id, limit=window_size)
        
        context_parts = []
        for memory in memories:
            prefix = "User" if memory.memory_type == 'query' else "SI"
            context_parts.append(f"{prefix}: {memory.content}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """
        Clear all memories for a session
        """
        if session_id in self.session_memories:
            memory_ids = self.session_memories[session_id]
            for mem_id in memory_ids:
                if mem_id in self.memory_index:
                    del self.memory_index[mem_id]
            
            del self.session_memories[session_id]
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        logger.info(f"Cleared session: {session_id}")
    
    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory system statistics
        """
        if session_id:
            memory_count = len(self.session_memories.get(session_id, []))
            session = self.sessions.get(session_id)
            
            return {
                'session_id': session_id,
                'memory_count': memory_count,
                'session_duration': time.time() - session.created if session else 0,
                'short_term_utilization': len(self.short_term) / self.short_term_capacity
            }
        else:
            return {
                'total_memories': len(self.memory_index),
                'active_sessions': len(self.sessions),
                'short_term_count': len(self.short_term),
                'short_term_utilization': len(self.short_term) / self.short_term_capacity
            }
    
    # Database persistence methods (stubs - implement based on your DB)
    def _persist_session(self, session: Session):
        """Persist session to database"""
        if not self.db:
            return
        # TODO: Implement database persistence
        pass
    
    def _persist_memory(self, memory: MemoryEntry):
        """Persist memory to database"""
        if not self.db:
            return
        # TODO: Implement database persistence
        pass
    
    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load session from database"""
        if not self.db:
            return None
        # TODO: Implement database loading
        return None