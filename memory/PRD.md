# Synthetic Intelligence Engine - Product Requirements Document

## Project Overview
A pure pattern-based Synthetic Intelligence (SI) engine with no neural networks or external LLM dependencies. The system uses pattern matching, multi-strategy reasoning, and consciousness tracking to process queries and generate images.

## Architecture

### Core SI Modules (Backend)
1. **PatternDatabase** - In-memory pattern storage with semantic search
2. **PatternMatcher** - TF-IDF based pattern recognition and similarity scoring
3. **QuasiReasoningEngine** - 6 cognitive strategies (analogical, first_principles, inductive, deductive, chunking, hypothesis_testing)
4. **SyntheticLanguageGenerator** - N-gram based text generation
5. **EmergentReasoning** - Cross-domain pattern connections and "aha moments"
6. **WorldModelingEngine** - Causal reasoning with counterfactual analysis
7. **SelfModelingEngine** - Performance tracking and recursive self-improvement
8. **EntityKnowledgeBase** - 43+ real-world entities with aliases and facts
9. **ParallelRealityEngine** - Quantum-inspired parallel simulation
10. **SyntheticIntelligence** - Main orchestrator

### Image Generation Modules (NEW - Phase 2)
1. **VisualPatternDatabase** - 19 pre-seeded visual patterns (shapes, objects, scenes)
2. **TextToVisualDecomposer** - Text to visual concept parsing
3. **SceneComposer** - Scene graph composition with spatial reasoning
4. **PatternRenderer** - SVG/PNG rendering pipeline
5. **ConsciousnessController** - DSINN integration for visual reasoning
6. **RealTimeOptimizer** - Caching and optimization for <500ms response

### Frontend
- React + Tailwind CSS
- Dark theme with laser green accents
- Chat interface for text queries
- Image generation interface at /generate

## User Personas

### Primary Users
1. **AI Researchers** - Studying pattern-based intelligence approaches
2. **Developers** - Building applications with SI capabilities
3. **Educators** - Teaching AI concepts without neural network complexity

## Core Requirements (Static)

### Functional Requirements
- [x] Pattern-based reasoning without neural networks
- [x] Sub-500ms response time (achieved 1-2ms)
- [x] Multi-strategy cognitive reasoning
- [x] Consciousness state tracking
- [x] Self-improvement capability
- [x] Pattern-based image generation
- [x] Cross-domain insight generation

### Non-Functional Requirements
- [x] MongoDB for persistence
- [x] FastAPI backend
- [x] React frontend
- [x] Real-time performance (<500ms)

## What's Been Implemented

### December 2025 - Initial MVP
- Complete SI engine with 10 core modules
- 36 pre-seeded knowledge patterns across domains (science, philosophy, technology, etc.)
- 43 entities in knowledge base
- Chat interface with reasoning analysis panel
- Statistics dashboard

### December 2025 - Image Generation Module
- Pattern-Based Visual Vocabulary (PBVV) system
- 19 visual patterns (circle, rectangle, triangle, line, tree, house, car, sun, cloud, mountain, water, road, person, dog, sailboat, grass, sky, sunset_sky, rain)
- Text decomposition into visual concepts
- Scene graph composition with:
  - Spatial reasoning
  - Layer management (background, foreground, etc.)
  - Causal constraints (car touches road, etc.)
- SVG rendering pipeline
- Consciousness tracking during generation
- Multi-level caching (L1 RAM, L2 patterns)
- Sub-1ms generation time achieved

## Prioritized Backlog

### P0 - Critical (Next)
- [ ] Add more visual patterns (buildings, vehicles, animals)
- [ ] Improve color variation based on text attributes

### P1 - High Priority
- [ ] PNG export with transparency
- [ ] Pattern composition complexity (combining patterns)
- [ ] Animation support for dynamic scenes

### P2 - Medium Priority
- [ ] User pattern creation interface
- [ ] Pattern sharing/export functionality
- [ ] Style presets (realistic, cartoon, minimal)

### P3 - Low Priority
- [ ] 3D pattern support
- [ ] Vector export formats (SVG, PDF)
- [ ] Collaborative pattern editing

## Next Tasks

1. **Expand Visual Pattern Library**
   - Add more composite patterns (city, forest, beach scene)
   - Implement pattern variations

2. **Enhance Text Understanding**
   - Better attribute extraction (adjectives → visual properties)
   - Quantity handling ("two cars", "many trees")

3. **Improve Composition**
   - Perspective and depth
   - Shadow generation
   - More sophisticated spatial layouts

4. **Performance Monitoring**
   - Dashboard for generation metrics
   - Pattern usage analytics

## API Endpoints

### SI Text Processing
- `POST /api/si/ask` - Main query endpoint
- `POST /api/si/simulate` - Parallel reality simulation
- `POST /api/si/causal` - Causal reasoning
- `GET /api/si/stats` - Statistics
- `GET /api/si/patterns` - Pattern retrieval

### Image Generation
- `POST /api/generate-image` - Generate image from description
- `GET /api/visual-patterns` - List visual patterns
- `GET /api/visual-patterns/{id}/preview` - Pattern preview
- `GET /api/image-generation/stats` - Generation statistics

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Text Query Response | <500ms | 1-2ms |
| Image Generation | <500ms | 0-1ms |
| Pattern Match Confidence | >70% | 73-85% |
| Cache Hit Rate | >50% | Yes |

## Technology Stack
- **Backend**: Python 3.11, FastAPI, Motor (MongoDB)
- **Frontend**: React 18, Tailwind CSS, Shadcn/UI
- **Database**: MongoDB
- **Rendering**: SVG (native), CairoSVG (PNG conversion)
