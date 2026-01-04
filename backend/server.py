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
