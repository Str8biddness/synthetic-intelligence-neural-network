"""
SyntheticLanguageGenerator - Token-by-token text generation
Uses statistical language modeling without neural networks
"""

import re
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math


class SyntheticLanguageGenerator:
    """
    Pure pattern-based language generation
    Uses n-gram statistics and template-based generation
    """
    
    def __init__(self, pattern_database):
        self.db = pattern_database
        
        # N-gram models
        self.unigrams: Dict[str, int] = defaultdict(int)
        self.bigrams: Dict[Tuple[str, str], int] = defaultdict(int)
        self.trigrams: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        # Transition probabilities
        self.bigram_probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.trigram_probs: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        
        # Response templates
        self.templates = self._init_templates()
        
        # Build language model from patterns
        self._build_language_model()
        
    def _init_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates by type"""
        return {
            'definition': [
                "{subject} is {definition}.",
                "{subject} refers to {definition}.",
                "In essence, {subject} can be understood as {definition}.",
                "{subject} represents {definition}."
            ],
            'explanation': [
                "{subject} works through {mechanism}. {details}",
                "The process of {subject} involves {mechanism}. {details}",
                "To understand {subject}: {mechanism}. {details}"
            ],
            'comparison': [
                "While {subject1} {property1}, {subject2} {property2}.",
                "{subject1} and {subject2} differ in that {difference}.",
                "Comparing {subject1} to {subject2}: {comparison}."
            ],
            'reason': [
                "{subject} occurs because {reason}.",
                "The reason for {subject} is {reason}.",
                "{reason} explains why {subject}."
            ],
            'synthesis': [
                "Based on pattern analysis: {content}",
                "Synthesizing available knowledge: {content}",
                "From the patterns observed: {content}"
            ],
            'uncertainty': [
                "Based on available patterns, {tentative_answer}. Confidence: {confidence}%",
                "Pattern matching suggests: {tentative_answer}",
                "Limited pattern data indicates: {tentative_answer}"
            ]
        }
        
    def _build_language_model(self):
        """Build n-gram language model from pattern database"""
        patterns = self.db.get_all_patterns()
        
        for pattern in patterns:
            # Process response text
            text = pattern.response
            tokens = self._tokenize(text)
            
            # Build n-grams
            for i, token in enumerate(tokens):
                self.unigrams[token] += 1
                
                if i > 0:
                    bigram = (tokens[i-1], token)
                    self.bigrams[bigram] += 1
                    
                if i > 1:
                    trigram = (tokens[i-2], tokens[i-1], token)
                    self.trigrams[trigram] += 1
                    
        # Calculate probabilities
        self._calculate_probabilities()
        
    def _calculate_probabilities(self):
        """Calculate transition probabilities"""
        # Bigram probabilities
        context_counts: Dict[str, int] = defaultdict(int)
        for (w1, w2), count in self.bigrams.items():
            context_counts[w1] += count
            
        for (w1, w2), count in self.bigrams.items():
            if context_counts[w1] > 0:
                self.bigram_probs[w1][w2] = count / context_counts[w1]
                
        # Trigram probabilities
        context_counts_tri: Dict[Tuple[str, str], int] = defaultdict(int)
        for (w1, w2, w3), count in self.trigrams.items():
            context_counts_tri[(w1, w2)] += count
            
        for (w1, w2, w3), count in self.trigrams.items():
            key = (w1, w2)
            if context_counts_tri[key] > 0:
                self.trigram_probs[key][w3] = count / context_counts_tri[key]
                
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[.,!?;:]', text)
        return ['<START>'] + tokens + ['<END>']
        
    def _sample_next_token(
        self, 
        context: List[str], 
        temperature: float = 1.0
    ) -> str:
        """Sample next token based on context"""
        candidates = {}
        
        # Try trigram first
        if len(context) >= 2:
            key = (context[-2], context[-1])
            if key in self.trigram_probs:
                candidates = self.trigram_probs[key].copy()
                
        # Fall back to bigram
        if not candidates and len(context) >= 1:
            last_word = context[-1]
            if last_word in self.bigram_probs:
                candidates = self.bigram_probs[last_word].copy()
                
        # Fall back to unigram
        if not candidates:
            total = sum(self.unigrams.values()) or 1
            candidates = {w: c/total for w, c in self.unigrams.items()}
            
        if not candidates:
            return '<END>'
            
        # Apply temperature
        if temperature != 1.0:
            candidates = {
                w: p ** (1.0 / temperature) 
                for w, p in candidates.items()
            }
            
        # Normalize
        total = sum(candidates.values())
        if total > 0:
            candidates = {w: p/total for w, p in candidates.items()}
            
        # Sample
        r = random.random()
        cumulative = 0
        for word, prob in sorted(candidates.items(), key=lambda x: -x[1]):
            cumulative += prob
            if r <= cumulative:
                return word
                
        return list(candidates.keys())[0] if candidates else '<END>'
        
    def generate_tokens(
        self, 
        seed: Optional[List[str]] = None, 
        max_tokens: int = 50,
        temperature: float = 1.0
    ) -> List[str]:
        """Generate tokens using the language model"""
        if seed:
            context = ['<START>'] + [s.lower() for s in seed]
        else:
            context = ['<START>']
            
        generated = []
        
        for _ in range(max_tokens):
            next_token = self._sample_next_token(context, temperature)
            
            if next_token == '<END>' or next_token in ['<START>']:
                break
                
            generated.append(next_token)
            context.append(next_token)
            
            # Stop at sentence end
            if next_token in ['.', '!', '?'] and len(generated) > 10:
                break
                
        return generated
        
    def generate_response(
        self,
        query: str,
        matched_response: str,
        response_type: str = 'synthesis',
        confidence: float = 0.8
    ) -> str:
        """
        Generate a response by combining pattern matching with language generation
        """
        # If we have a good matched response, use it with minimal modification
        if matched_response and confidence > 0.7:
            return self._enhance_response(matched_response, query)
            
        # Otherwise, use template-based generation
        templates = self.templates.get(response_type, self.templates['synthesis'])
        template = random.choice(templates)
        
        # Extract key concepts from query
        concepts = self._extract_concepts(query)
        subject = concepts[0] if concepts else query
        content = matched_response or self._generate_fallback(query)
        
        # Fill template with safe formatting
        try:
            if response_type == 'definition':
                return template.format(
                    subject=subject,
                    definition=content
                )
            elif response_type == 'uncertainty':
                return template.format(
                    tentative_answer=content,
                    confidence=int(confidence * 100)
                )
            elif response_type == 'explanation':
                return template.format(
                    subject=subject,
                    mechanism=content[:200] if len(content) > 200 else content,
                    details=content[200:] if len(content) > 200 else ""
                )
            elif response_type == 'reason':
                return template.format(
                    subject=subject,
                    reason=content
                )
            elif response_type == 'comparison':
                return template.format(
                    subject1=subject,
                    subject2=concepts[1] if len(concepts) > 1 else "the alternative",
                    property1="exhibits certain characteristics",
                    property2="shows different traits",
                    difference=content,
                    comparison=content
                )
            else:
                return template.format(content=content)
        except KeyError:
            # Fallback if template formatting fails
            return content
            
    def _enhance_response(self, response: str, query: str) -> str:
        """Enhance a response based on query context"""
        # Check if response already addresses the query type
        query_lower = query.lower()
        
        if query_lower.startswith('why') and not response.lower().startswith('because'):
            # Add causal framing if answering a 'why' question
            return response
            
        if query_lower.startswith('how') and 'steps' not in response.lower():
            # Response already explains the process
            return response
            
        return response
        
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'are', 'were', 'was', 'be', 'been', 'being', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from'}
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        return concepts
        
    def _generate_fallback(self, query: str) -> str:
        """Generate fallback response when patterns are insufficient"""
        concepts = self._extract_concepts(query)
        
        if concepts:
            # Try to generate something from the language model
            generated = self.generate_tokens(seed=concepts[:2], max_tokens=30, temperature=0.8)
            if len(generated) > 5:
                return ' '.join(generated).capitalize() + '.'
                
        return f"I can analyze patterns related to '{query}', but my current pattern database may not have sufficient coverage for a complete answer."
        
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text under the model"""
        tokens = self._tokenize(text)
        
        if len(tokens) < 3:
            return float('inf')
            
        log_prob_sum = 0
        count = 0
        
        for i in range(2, len(tokens)):
            trigram_key = (tokens[i-2], tokens[i-1])
            token = tokens[i]
            
            if trigram_key in self.trigram_probs and token in self.trigram_probs[trigram_key]:
                prob = self.trigram_probs[trigram_key][token]
                log_prob_sum += math.log2(prob)
            elif tokens[i-1] in self.bigram_probs and token in self.bigram_probs[tokens[i-1]]:
                prob = self.bigram_probs[tokens[i-1]][token]
                log_prob_sum += math.log2(prob)
            else:
                # Smoothing for unseen n-grams
                log_prob_sum += math.log2(1e-6)
                
            count += 1
            
        if count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / count
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity
        
    def refresh_model(self):
        """Refresh the language model with updated patterns"""
        self.unigrams.clear()
        self.bigrams.clear()
        self.trigrams.clear()
        self.bigram_probs.clear()
        self.trigram_probs.clear()
        self._build_language_model()
