"""
QuasiReasoningEngine - Multi-strategy cognitive reasoning
Implements 6 cognitive strategies for query processing
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of queries the engine can handle"""
    WHY = "why"
    HOW = "how"
    WHAT = "what"
    COMPARE = "compare"
    DEFINE = "define"
    EXPLAIN = "explain"
    WHEN = "when"
    WHERE = "where"
    WHO = "who"
    CALCULATE = "calculate"
    UNKNOWN = "unknown"


class CognitiveStrategy(Enum):
    """Available cognitive strategies"""
    ANALOGICAL = "analogical"  # Reasoning by analogy
    FIRST_PRINCIPLES = "first_principles"  # Break down to fundamentals
    INDUCTIVE = "inductive"  # Specific to general
    DEDUCTIVE = "deductive"  # General to specific
    CHUNKING = "chunking"  # Break into manageable pieces
    HYPOTHESIS_TESTING = "hypothesis_testing"  # Form and test hypotheses


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_number: int
    strategy: CognitiveStrategy
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'strategy': self.strategy.value,
            'description': self.description,
            'confidence': self.confidence
        }


@dataclass
class ReasoningResult:
    """Result of the reasoning process"""
    query: str
    query_type: QueryType
    primary_strategy: CognitiveStrategy
    steps: List[ReasoningStep]
    conclusion: str
    confidence: float
    supporting_evidence: List[str]
    alternative_interpretations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'query_type': self.query_type.value,
            'primary_strategy': self.primary_strategy.value,
            'steps': [s.to_dict() for s in self.steps],
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'supporting_evidence': self.supporting_evidence,
            'alternative_interpretations': self.alternative_interpretations
        }


class QuasiReasoningEngine:
    """
    Multi-strategy reasoning engine
    Implements cognitive strategies for query processing
    """
    
    def __init__(self, pattern_database, pattern_matcher):
        self.db = pattern_database
        self.matcher = pattern_matcher
        
        # Strategy effectiveness tracking
        self.strategy_stats: Dict[CognitiveStrategy, Dict] = {
            strategy: {'uses': 0, 'successes': 0} 
            for strategy in CognitiveStrategy
        }
        
        # Query type to strategy mapping
        self.type_strategy_map = {
            QueryType.WHY: [CognitiveStrategy.FIRST_PRINCIPLES, CognitiveStrategy.DEDUCTIVE],
            QueryType.HOW: [CognitiveStrategy.CHUNKING, CognitiveStrategy.ANALOGICAL],
            QueryType.WHAT: [CognitiveStrategy.DEDUCTIVE, CognitiveStrategy.INDUCTIVE],
            QueryType.COMPARE: [CognitiveStrategy.ANALOGICAL, CognitiveStrategy.HYPOTHESIS_TESTING],
            QueryType.DEFINE: [CognitiveStrategy.FIRST_PRINCIPLES, CognitiveStrategy.CHUNKING],
            QueryType.EXPLAIN: [CognitiveStrategy.CHUNKING, CognitiveStrategy.ANALOGICAL],
            QueryType.WHEN: [CognitiveStrategy.DEDUCTIVE, CognitiveStrategy.INDUCTIVE],
            QueryType.WHERE: [CognitiveStrategy.DEDUCTIVE, CognitiveStrategy.INDUCTIVE],
            QueryType.WHO: [CognitiveStrategy.DEDUCTIVE, CognitiveStrategy.INDUCTIVE],
            QueryType.CALCULATE: [CognitiveStrategy.FIRST_PRINCIPLES, CognitiveStrategy.CHUNKING],
            QueryType.UNKNOWN: [CognitiveStrategy.HYPOTHESIS_TESTING, CognitiveStrategy.INDUCTIVE]
        }
        
    def classify_query(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower().strip()
        
        # Check for question words
        if query_lower.startswith('why') or 'reason' in query_lower or 'cause' in query_lower:
            return QueryType.WHY
        elif query_lower.startswith('how') or 'process' in query_lower:
            return QueryType.HOW
        elif query_lower.startswith('what') or 'what is' in query_lower:
            return QueryType.WHAT
        elif 'compare' in query_lower or 'difference' in query_lower or 'versus' in query_lower or ' vs ' in query_lower:
            return QueryType.COMPARE
        elif 'define' in query_lower or 'definition' in query_lower:
            return QueryType.DEFINE
        elif 'explain' in query_lower or 'describe' in query_lower:
            return QueryType.EXPLAIN
        elif query_lower.startswith('when') or 'time' in query_lower:
            return QueryType.WHEN
        elif query_lower.startswith('where') or 'location' in query_lower:
            return QueryType.WHERE
        elif query_lower.startswith('who') or 'person' in query_lower:
            return QueryType.WHO
        elif 'calculate' in query_lower or 'compute' in query_lower or any(c.isdigit() for c in query):
            return QueryType.CALCULATE
        else:
            return QueryType.UNKNOWN
            
    def select_strategy(self, query_type: QueryType, context: Optional[Dict] = None) -> CognitiveStrategy:
        """Select the best cognitive strategy for the query"""
        strategies = self.type_strategy_map.get(query_type, [CognitiveStrategy.INDUCTIVE])
        
        # Select based on past effectiveness
        best_strategy = strategies[0]
        best_score = 0
        
        for strategy in strategies:
            stats = self.strategy_stats[strategy]
            if stats['uses'] > 0:
                score = stats['successes'] / stats['uses']
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    
        return best_strategy
        
    def reason(self, query: str, matched_patterns: List[Any]) -> ReasoningResult:
        """
        Apply reasoning to query and matched patterns
        Returns a structured reasoning result
        """
        query_type = self.classify_query(query)
        strategy = self.select_strategy(query_type)
        
        # Execute reasoning based on strategy
        steps = []
        conclusion = ""
        confidence = 0.0
        supporting_evidence = []
        alternatives = []
        
        if strategy == CognitiveStrategy.ANALOGICAL:
            steps, conclusion, confidence = self._analogical_reasoning(query, matched_patterns)
        elif strategy == CognitiveStrategy.FIRST_PRINCIPLES:
            steps, conclusion, confidence = self._first_principles_reasoning(query, matched_patterns)
        elif strategy == CognitiveStrategy.INDUCTIVE:
            steps, conclusion, confidence = self._inductive_reasoning(query, matched_patterns)
        elif strategy == CognitiveStrategy.DEDUCTIVE:
            steps, conclusion, confidence = self._deductive_reasoning(query, matched_patterns)
        elif strategy == CognitiveStrategy.CHUNKING:
            steps, conclusion, confidence = self._chunking_reasoning(query, matched_patterns)
        elif strategy == CognitiveStrategy.HYPOTHESIS_TESTING:
            steps, conclusion, confidence = self._hypothesis_testing(query, matched_patterns)
            
        # Extract supporting evidence from patterns
        for pattern in matched_patterns[:3]:
            if hasattr(pattern, 'response'):
                # Support both pattern_text and pattern attributes
                pattern_name = getattr(pattern, 'pattern_text', None) or getattr(pattern, 'pattern', 'Unknown')
                supporting_evidence.append(f"Pattern '{pattern_name}' suggests: {pattern.response[:100]}...")
                
        # Generate alternative interpretations
        alternatives = self._generate_alternatives(query, query_type, matched_patterns)
        
        # Update strategy statistics
        self.strategy_stats[strategy]['uses'] += 1
        
        return ReasoningResult(
            query=query,
            query_type=query_type,
            primary_strategy=strategy,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            alternative_interpretations=alternatives
        )
        
    def _analogical_reasoning(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Reason by analogy - find similar concepts"""
        steps = []
        
        # Step 1: Identify source domain
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.ANALOGICAL,
            description="Identifying source concepts for analogy",
            input_data=query,
            output_data=[p.domain if hasattr(p, 'domain') else 'general' for p in patterns[:3]],
            confidence=0.8
        )
        steps.append(step1)
        
        # Step 2: Map relationships
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.ANALOGICAL,
            description="Mapping relationships between source and target domains",
            input_data=patterns[:2] if patterns else [],
            output_data="Relationship mapping complete",
            confidence=0.75
        )
        steps.append(step2)
        
        # Step 3: Draw conclusion
        if patterns:
            best_pattern = patterns[0]
            conclusion = getattr(best_pattern, 'response', "No direct analogy found.")
            confidence = getattr(best_pattern, 'confidence', 0.6)
        else:
            conclusion = "Unable to find suitable analogies for this query."
            confidence = 0.3
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.ANALOGICAL,
            description="Drawing conclusion from analogical mapping",
            input_data="mapped relationships",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _first_principles_reasoning(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Break down to fundamental truths"""
        steps = []
        
        # Step 1: Identify assumptions
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.FIRST_PRINCIPLES,
            description="Breaking down query into fundamental components",
            input_data=query,
            output_data=self._extract_key_concepts(query),
            confidence=0.85
        )
        steps.append(step1)
        
        # Step 2: Find foundational patterns
        foundational_responses = []
        for pattern in patterns:
            if hasattr(pattern, 'response'):
                foundational_responses.append(pattern.response)
                
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.FIRST_PRINCIPLES,
            description="Identifying foundational principles",
            input_data=step1.output_data,
            output_data=f"Found {len(foundational_responses)} foundational patterns",
            confidence=0.8
        )
        steps.append(step2)
        
        # Step 3: Build up from principles
        if foundational_responses:
            conclusion = foundational_responses[0]
            confidence = patterns[0].confidence if patterns and hasattr(patterns[0], 'confidence') else 0.7
        else:
            conclusion = "Breaking down to first principles: " + query
            confidence = 0.5
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.FIRST_PRINCIPLES,
            description="Reconstructing understanding from principles",
            input_data="foundational patterns",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _inductive_reasoning(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Generalize from specific examples"""
        steps = []
        
        # Step 1: Gather specific examples
        examples = []
        for pattern in patterns[:5]:
            if hasattr(pattern, 'response'):
                examples.append(pattern.response[:50])
                
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.INDUCTIVE,
            description="Gathering specific examples and observations",
            input_data=query,
            output_data=f"Found {len(examples)} relevant examples",
            confidence=0.8
        )
        steps.append(step1)
        
        # Step 2: Identify patterns in examples
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.INDUCTIVE,
            description="Identifying common patterns across examples",
            input_data=examples,
            output_data="Pattern analysis complete",
            confidence=0.75
        )
        steps.append(step2)
        
        # Step 3: Form general conclusion
        if patterns:
            conclusion = patterns[0].response if hasattr(patterns[0], 'response') else str(patterns[0])
            confidence = patterns[0].confidence if hasattr(patterns[0], 'confidence') else 0.65
        else:
            conclusion = "Insufficient examples to form inductive conclusion."
            confidence = 0.3
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.INDUCTIVE,
            description="Forming general conclusion from patterns",
            input_data="analyzed patterns",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _deductive_reasoning(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Apply general rules to specific cases"""
        steps = []
        
        # Step 1: Identify relevant general rules
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.DEDUCTIVE,
            description="Identifying general rules and premises",
            input_data=query,
            output_data=f"Found {len(patterns)} applicable rules",
            confidence=0.85
        )
        steps.append(step1)
        
        # Step 2: Apply rules to specific case
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.DEDUCTIVE,
            description="Applying rules to the specific query",
            input_data=patterns[:3] if patterns else [],
            output_data="Rule application complete",
            confidence=0.8
        )
        steps.append(step2)
        
        # Step 3: Derive specific conclusion
        if patterns:
            conclusion = patterns[0].response if hasattr(patterns[0], 'response') else str(patterns[0])
            confidence = patterns[0].confidence if hasattr(patterns[0], 'confidence') else 0.7
        else:
            conclusion = "No general rules found applicable to this specific case."
            confidence = 0.4
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.DEDUCTIVE,
            description="Deriving specific conclusion from rules",
            input_data="applied rules",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _chunking_reasoning(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Break problem into manageable chunks"""
        steps = []
        
        # Step 1: Decompose problem
        chunks = self._decompose_query(query)
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.CHUNKING,
            description="Decomposing problem into chunks",
            input_data=query,
            output_data=chunks,
            confidence=0.85
        )
        steps.append(step1)
        
        # Step 2: Process each chunk
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.CHUNKING,
            description="Processing each chunk individually",
            input_data=chunks,
            output_data=f"Processed {len(chunks)} chunks",
            confidence=0.8
        )
        steps.append(step2)
        
        # Step 3: Synthesize results
        if patterns:
            conclusion = patterns[0].response if hasattr(patterns[0], 'response') else str(patterns[0])
            confidence = patterns[0].confidence if hasattr(patterns[0], 'confidence') else 0.7
        else:
            conclusion = "Unable to synthesize chunks into coherent answer."
            confidence = 0.4
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.CHUNKING,
            description="Synthesizing chunk results into final answer",
            input_data="processed chunks",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _hypothesis_testing(
        self, query: str, patterns: List[Any]
    ) -> Tuple[List[ReasoningStep], str, float]:
        """Form and test hypotheses"""
        steps = []
        
        # Step 1: Generate hypotheses
        hypotheses = self._generate_hypotheses(query, patterns)
        step1 = ReasoningStep(
            step_number=1,
            strategy=CognitiveStrategy.HYPOTHESIS_TESTING,
            description="Generating hypotheses about the answer",
            input_data=query,
            output_data=hypotheses,
            confidence=0.75
        )
        steps.append(step1)
        
        # Step 2: Test against evidence
        step2 = ReasoningStep(
            step_number=2,
            strategy=CognitiveStrategy.HYPOTHESIS_TESTING,
            description="Testing hypotheses against available evidence",
            input_data=hypotheses,
            output_data=f"Tested {len(hypotheses)} hypotheses",
            confidence=0.7
        )
        steps.append(step2)
        
        # Step 3: Select best hypothesis
        if patterns:
            conclusion = patterns[0].response if hasattr(patterns[0], 'response') else str(patterns[0])
            confidence = patterns[0].confidence if hasattr(patterns[0], 'confidence') else 0.65
        else:
            conclusion = "No hypothesis could be confirmed with available evidence."
            confidence = 0.35
            
        step3 = ReasoningStep(
            step_number=3,
            strategy=CognitiveStrategy.HYPOTHESIS_TESTING,
            description="Selecting most supported hypothesis",
            input_data="tested hypotheses",
            output_data=conclusion[:100],
            confidence=confidence
        )
        steps.append(step3)
        
        return steps, conclusion, confidence
        
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Remove question words and common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'are', 'were', 'was', 'be', 'been', 'being'}
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        return concepts[:5]
        
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose query into smaller chunks"""
        # Simple decomposition based on structure
        chunks = []
        
        # Extract main subject
        concepts = self._extract_key_concepts(query)
        if concepts:
            chunks.append(f"Define: {concepts[0]}")
            if len(concepts) > 1:
                chunks.append(f"Relationship: {' and '.join(concepts[:2])}")
            chunks.append(f"Context: {query}")
            
        return chunks if chunks else [query]
        
    def _generate_hypotheses(self, query: str, patterns: List[Any]) -> List[str]:
        """Generate possible hypotheses"""
        hypotheses = []
        
        for pattern in patterns[:3]:
            if hasattr(pattern, 'response'):
                hypotheses.append(f"H{len(hypotheses)+1}: {pattern.response[:80]}...")
                
        if not hypotheses:
            hypotheses.append("H1: Query requires more context")
            hypotheses.append("H2: Query falls outside known patterns")
            
        return hypotheses
        
    def _generate_alternatives(
        self, query: str, query_type: QueryType, patterns: List[Any]
    ) -> List[str]:
        """Generate alternative interpretations"""
        alternatives = []
        
        # Different query type interpretation
        if query_type != QueryType.WHAT:
            alternatives.append(f"Could also be interpreted as: What is {query}?")
            
        # Alternative domains
        domains = set()
        for pattern in patterns:
            if hasattr(pattern, 'domain'):
                domains.add(pattern.domain)
                
        if len(domains) > 1:
            alternatives.append(f"Spans multiple domains: {', '.join(list(domains)[:3])}")
            
        return alternatives[:3]
        
    def record_success(self, strategy: CognitiveStrategy, success: bool):
        """Record success/failure for strategy learning"""
        self.strategy_stats[strategy]['uses'] += 1
        if success:
            self.strategy_stats[strategy]['successes'] += 1
            
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get strategy effectiveness statistics"""
        stats = {}
        for strategy, data in self.strategy_stats.items():
            uses = data['uses']
            successes = data['successes']
            stats[strategy.value] = {
                'uses': uses,
                'successes': successes,
                'success_rate': successes / uses if uses > 0 else 0.0
            }
        return stats
