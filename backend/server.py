from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone

# SI Engine imports
from si_engine import SyntheticIntelligence

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Synthetic Intelligence API")

# Create router with /api prefix
api_router = APIRouter(prefix="/api")

# Initialize SI Engine (singleton)
si_engine = SyntheticIntelligence()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    id: str
    query: str
    response: str
    confidence: float
    reasoning_strategy: str
    patterns_used: int
    domains_involved: List[str]
    reasoning_steps: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    response_time_ms: float
    consciousness_state: Dict[str, Any]

class SimulationRequest(BaseModel):
    query: str
    assumptions: Optional[List[str]] = None

class CausalRequest(BaseModel):
    query: str

class CounterfactualRequest(BaseModel):
    intervention: str
    target: str

class PatternCreate(BaseModel):
    pattern: str
    response: str
    domain: str
    topics: List[str]
    keywords: Optional[List[str]] = None

class FeedbackRequest(BaseModel):
    response_id: str
    success: bool

class ImageGenerateRequest(BaseModel):
    description: str
    use_optimizer: bool = True

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SessionCreate(BaseModel):
    user_id: Optional[str] = None

class Session(BaseModel):
    id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

# ==================== SI ENDPOINTS ====================

@api_router.get("/")
async def root():
    return {"message": "Synthetic Intelligence API", "status": "operational"}

@api_router.post("/si/ask", response_model=QueryResponse)
async def ask_si(request: QueryRequest):
    """Main query endpoint for SI"""
    try:
        response = si_engine.process_query(request.query, request.session_id)
        
        # Store in MongoDB for persistence
        message_doc = {
            'id': str(uuid.uuid4()),
            'session_id': request.session_id or 'anonymous',
            'query': request.query,
            'response': response.response,
            'confidence': response.confidence,
            'strategy': response.reasoning_strategy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await db.memory.insert_one(message_doc)
        
        return QueryResponse(**response.to_dict())
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/simulate")
async def simulate_realities(request: SimulationRequest):
    """Parallel reality simulation endpoint"""
    try:
        result = si_engine.simulate_realities(request.query, request.assumptions)
        return result
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/causal")
async def causal_reasoning(request: CausalRequest):
    """Causal reasoning endpoint"""
    try:
        result = si_engine.causal_reasoning(request.query)
        return result
    except Exception as e:
        logger.error(f"Error in causal reasoning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/counterfactual")
async def counterfactual_reasoning(request: CounterfactualRequest):
    """Counterfactual reasoning endpoint"""
    try:
        result = si_engine.counterfactual(request.intervention, request.target)
        return result
    except Exception as e:
        logger.error(f"Error in counterfactual: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/patterns")
async def get_patterns(domain: Optional[str] = None, limit: int = 20):
    """Get patterns from the database"""
    try:
        patterns = si_engine.get_patterns(domain=domain, limit=limit)
        return {"patterns": patterns, "count": len(patterns)}
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/patterns")
async def create_pattern(pattern: PatternCreate):
    """Add a new pattern to the database"""
    try:
        pattern_data = {
            'pattern': pattern.pattern,
            'response': pattern.response,
            'domain': pattern.domain,
            'topics': pattern.topics,
            'keywords': pattern.keywords or []
        }
        result = si_engine.add_pattern(pattern_data)
        return {"success": True, "pattern": result}
    except Exception as e:
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/stats")
async def get_statistics():
    """Get SI statistics"""
    try:
        stats = si_engine.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/self-observe")
async def self_observe():
    """Get self-observation report"""
    try:
        observation = si_engine.observe_self()
        return observation
    except Exception as e:
        logger.error(f"Error in self-observation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/self-improve")
async def self_improve():
    """Trigger self-improvement analysis"""
    try:
        improvement = si_engine.self_improve()
        return improvement
    except Exception as e:
        logger.error(f"Error in self-improvement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/feedback")
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback on a response"""
    try:
        si_engine.provide_feedback(request.response_id, request.success)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/strategies")
async def get_strategy_stats():
    """Get reasoning strategy statistics"""
    try:
        stats = si_engine.get_strategy_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting strategy stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/entities")
async def search_entities(query: str):
    """Search entities in knowledge base"""
    try:
        entities = si_engine.search_entities(query)
        return {"entities": entities, "count": len(entities)}
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE GENERATION ENDPOINTS ====================

@api_router.post("/generate-image")
async def generate_image(request: ImageGenerateRequest):
    """
    Generate an image from text description using pattern-based composition
    No neural networks - pure pattern matching and composition
    """
    try:
        result = si_engine.generate_image(
            request.description,
            use_optimizer=request.use_optimizer
        )
        return result
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/visual-patterns")
async def get_visual_patterns(tags: Optional[str] = None, limit: int = 20):
    """Get visual patterns from database"""
    try:
        tag_list = tags.split(',') if tags else None
        patterns = si_engine.get_visual_patterns(tags=tag_list, limit=limit)
        return {"patterns": patterns, "count": len(patterns)}
    except Exception as e:
        logger.error(f"Error getting visual patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/visual-patterns/{pattern_id}/preview")
async def get_visual_pattern_preview(pattern_id: str):
    """Get SVG preview of a visual pattern"""
    try:
        preview = si_engine.get_visual_pattern_preview(pattern_id)
        return preview
    except Exception as e:
        logger.error(f"Error getting pattern preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/image-generation/stats")
async def get_image_generation_stats():
    """Get image generation statistics"""
    try:
        stats = si_engine.get_image_generation_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting image generation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SESSION ENDPOINTS ====================

@api_router.post("/sessions", response_model=Session)
async def create_session(session_data: SessionCreate):
    """Create a new chat session"""
    session = {
        'id': str(uuid.uuid4()),
        'user_id': session_data.user_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'message_count': 0
    }
    await db.sessions.insert_one(session)
    return Session(
        id=session['id'],
        user_id=session['user_id'],
        created_at=datetime.fromisoformat(session['created_at']),
        updated_at=datetime.fromisoformat(session['updated_at']),
        message_count=0
    )

@api_router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """Get messages for a session"""
    messages = await db.memory.find(
        {'session_id': session_id},
        {'_id': 0}
    ).sort('timestamp', 1).limit(limit).to_list(limit)
    return {"messages": messages}

# ==================== WEB SEARCH ENDPOINTS ====================

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    extract_patterns: bool = True

@api_router.post("/si/web-search")
async def web_search(request: WebSearchRequest):
    """
    Search the web using DuckDuckGo and extract patterns
    Results are cached and patterns can be auto-added to the database
    """
    try:
        result = si_engine.web_search.search(
            query=request.query,
            max_results=request.max_results,
            extract_patterns=request.extract_patterns
        )
        return result
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/web-search/stats")
async def get_web_search_stats():
    """Get web search statistics"""
    try:
        return si_engine.get_web_search_stats()
    except Exception as e:
        logger.error(f"Error getting web search stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/web-search/clear-cache")
async def clear_web_search_cache():
    """Clear the web search cache"""
    try:
        si_engine.clear_web_search_cache()
        return {"success": True, "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DAILY UPDATER ENDPOINTS ====================

@api_router.post("/si/updater/run")
async def run_pattern_update():
    """
    Manually trigger a pattern update from news sources
    Scrapes HN, ArXiv, and GitHub for new patterns
    """
    try:
        result = await si_engine.run_pattern_update()
        return result
    except Exception as e:
        logger.error(f"Pattern update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/si/updater/stats")
async def get_updater_stats():
    """Get daily updater statistics"""
    try:
        return si_engine.get_updater_stats()
    except Exception as e:
        logger.error(f"Error getting updater stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/updater/start")
async def start_daily_updater():
    """Start the scheduled daily updater (runs at 2 AM)"""
    try:
        si_engine.start_daily_updater()
        return {"success": True, "message": "Daily updater started"}
    except Exception as e:
        logger.error(f"Error starting updater: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/si/updater/stop")
async def stop_daily_updater():
    """Stop the scheduled daily updater"""
    try:
        si_engine.stop_daily_updater()
        return {"success": True, "message": "Daily updater stopped"}
    except Exception as e:
        logger.error(f"Error stopping updater: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SCALABLE PATTERN DB ENDPOINTS ====================

@api_router.get("/si/scalable-db/stats")
async def get_scalable_db_stats():
    """Get scalable pattern database statistics (FAISS index info)"""
    try:
        return si_engine.get_scalable_db_stats()
    except Exception as e:
        logger.error(f"Error getting scalable DB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FastSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    domain: Optional[str] = None

@api_router.post("/si/fast-search")
async def fast_pattern_search(request: FastSearchRequest):
    """
    Fast pattern search using FAISS index
    Achieves <10ms search time even on 1M+ patterns
    """
    try:
        results = si_engine.search_patterns_fast(
            query=request.query,
            top_k=request.top_k,
            domain=request.domain
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Fast search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HEALTH CHECK ====================

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "si_engine": "operational",
        "patterns_loaded": si_engine.pattern_db._initialized,
        "entities_loaded": si_engine.entity_kb._initialized
    }

#
# 
# =================== TEST ENDPOINT ====================

@api_router.get("/si/test")
async def test_si_capabilities():
    """Test endpoint to demonstrate all SI capabilities"""
    
    # Get hardware info
    hw_info = si_engine.hardware.hardware_info.to_dict()
    
    # Get memory stats
    memory_stats = si_engine.memory.get_stats()
    
    # Test web access
    web_result = si_engine.web_access.search_web("Python programming", num_results=2)
    web_data = [r.to_dict() for r in web_result]
    
    # Create test session
    session = si_engine.memory.create_session()
    
    # Store test memory
    si_engine.memory.store(
        session.id, 
        "Test query about AI", 
        "query",
        importance=0.8
    )
    
    return {
        "status": "operational",
        "capabilities": {
            "hardware": hw_info,
            "memory": memory_stats,
            "web_access": {
                "enabled": True,
                "test_results": web_data[:2] if web_data else []
            },
            "session_id": session.id
        }
    }
# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
