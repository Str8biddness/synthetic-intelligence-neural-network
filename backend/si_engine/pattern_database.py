"""
PatternDatabase - In-memory pattern storage with MongoDB persistence
Handles pattern indexing, CRUD operations, and metadata tracking
"""

import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re


@dataclass
class Pattern:
    """Represents a pattern with metadata"""
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


class PatternDatabase:
    """
    In-memory pattern storage with indexing capabilities
    Supports pattern retrieval by domain, topic, and keywords
    """
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.domain_index: Dict[str, List[str]] = defaultdict(list)
        self.topic_index: Dict[str, List[str]] = defaultdict(list)
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
        self._initialized = False
        
    def initialize_with_seed_data(self):
        """Pre-seed with patterns across domains"""
        if self._initialized:
            return
            
        seed_patterns = self._get_seed_patterns()
        for pattern_data in seed_patterns:
            self.add_pattern(pattern_data)
        self._initialized = True
        
    def _get_seed_patterns(self) -> List[Dict]:
        """Generate comprehensive seed patterns"""
        patterns = []
        
        # Science patterns
        science_patterns = [
            {"pattern": "what is gravity", "response": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to physical objects and causes dropped items to fall. The strength of gravitational attraction depends on the masses involved and the distance between them, following Newton's law of universal gravitation: F = G(m1*m2)/r².", "domain": "science", "topics": ["physics", "forces", "newton"], "keywords": ["gravity", "force", "mass", "attraction", "weight"]},
            {"pattern": "explain photosynthesis", "response": "Photosynthesis is the process by which plants convert light energy into chemical energy. Using chlorophyll in their leaves, plants absorb sunlight and combine carbon dioxide from air with water from soil to produce glucose (sugar) and oxygen. The equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2. This process is fundamental to life on Earth.", "domain": "science", "topics": ["biology", "plants", "energy"], "keywords": ["photosynthesis", "plants", "chlorophyll", "glucose", "oxygen", "sunlight"]},
            {"pattern": "what is quantum mechanics", "response": "Quantum mechanics is a fundamental theory in physics describing nature at atomic and subatomic scales. Key principles include: wave-particle duality (matter exhibits both wave and particle properties), superposition (particles exist in multiple states simultaneously until observed), and entanglement (particles can be correlated regardless of distance). It differs radically from classical physics.", "domain": "science", "topics": ["physics", "quantum", "atoms"], "keywords": ["quantum", "particles", "wave", "superposition", "entanglement", "physics"]},
            {"pattern": "how does dna work", "response": "DNA (deoxyribonucleic acid) is a molecule carrying genetic instructions for living organisms. It consists of two strands forming a double helix, with four bases: Adenine, Thymine, Guanine, and Cytosine. Base pairs (A-T, G-C) encode information. During replication, strands separate and new complementary strands form. Genes are DNA segments that encode proteins.", "domain": "science", "topics": ["biology", "genetics", "molecules"], "keywords": ["dna", "genes", "genetics", "helix", "bases", "replication"]},
            {"pattern": "what is relativity", "response": "Einstein's Theory of Relativity consists of Special Relativity (1905) and General Relativity (1915). Special Relativity states that the speed of light is constant and time dilates at high velocities (E=mc²). General Relativity describes gravity as curvature of spacetime caused by mass. Massive objects bend spacetime, causing what we perceive as gravitational attraction.", "domain": "science", "topics": ["physics", "einstein", "spacetime"], "keywords": ["relativity", "einstein", "spacetime", "light", "gravity", "time"]},
            {"pattern": "explain evolution", "response": "Evolution is the process by which species change over generations through natural selection. Key mechanisms: 1) Genetic variation exists in populations, 2) Individuals compete for resources, 3) Those with advantageous traits survive and reproduce more (fitness), 4) Beneficial traits become more common over time. Evidence includes fossils, DNA similarities, and observed speciation.", "domain": "science", "topics": ["biology", "darwin", "species"], "keywords": ["evolution", "natural selection", "darwin", "species", "adaptation", "genetics"]},
            {"pattern": "what is entropy", "response": "Entropy is a measure of disorder or randomness in a system. In thermodynamics, the Second Law states that entropy in isolated systems tends to increase over time. This explains why heat flows from hot to cold, why ice melts at room temperature, and why perpetual motion machines are impossible. Entropy is also used in information theory to measure uncertainty.", "domain": "science", "topics": ["physics", "thermodynamics", "information"], "keywords": ["entropy", "disorder", "thermodynamics", "heat", "energy", "randomness"]},
            {"pattern": "how do black holes work", "response": "Black holes are regions of spacetime where gravity is so strong that nothing, including light, can escape. They form when massive stars collapse at end of life. Key features: Event horizon (point of no return), singularity (center of infinite density), Hawking radiation (quantum effect causing slow evaporation). Black holes warp spacetime dramatically and can be detected by effects on nearby matter.", "domain": "science", "topics": ["physics", "astronomy", "spacetime"], "keywords": ["black hole", "gravity", "singularity", "event horizon", "spacetime", "stars"]},
            {"pattern": "what is the scientific method", "response": "The scientific method is a systematic approach to understanding phenomena: 1) Observation - notice something interesting, 2) Question - formulate a research question, 3) Hypothesis - propose a testable explanation, 4) Experiment - test the hypothesis with controlled experiments, 5) Analysis - examine results, 6) Conclusion - accept, reject, or modify hypothesis, 7) Peer review - share findings for validation.", "domain": "science", "topics": ["methodology", "research", "epistemology"], "keywords": ["scientific method", "hypothesis", "experiment", "observation", "research", "evidence"]},
            {"pattern": "explain the periodic table", "response": "The periodic table organizes all known elements by atomic number (protons). Elements in columns (groups) share similar properties due to electron configurations. Periods (rows) show electron shells filling. Key groups: Alkali metals (reactive), Noble gases (stable), Halogens (highly reactive). The table predicts element behavior and was created by Mendeleev in 1869.", "domain": "science", "topics": ["chemistry", "elements", "atoms"], "keywords": ["periodic table", "elements", "atoms", "chemistry", "electrons", "mendeleev"]},
        ]
        
        # Philosophy patterns
        philosophy_patterns = [
            {"pattern": "what is consciousness", "response": "Consciousness is subjective experience and awareness - the 'what it's like' to be something. Major theories include: Dualism (mind separate from body), Materialism (consciousness emerges from brain), Panpsychism (consciousness is fundamental). The Hard Problem asks why physical processes produce subjective experience. Consciousness remains one of philosophy's deepest mysteries.", "domain": "philosophy", "topics": ["mind", "awareness", "metaphysics"], "keywords": ["consciousness", "mind", "awareness", "experience", "brain", "qualia"]},
            {"pattern": "what is free will", "response": "Free will is the ability to choose between actions without external constraint. Major positions: Determinism (all events are caused, no free will), Libertarian free will (genuine choice exists), Compatibilism (free will compatible with determinism). Key questions: If brain activity precedes conscious decisions, are we truly free? Does moral responsibility require free will?", "domain": "philosophy", "topics": ["mind", "ethics", "metaphysics"], "keywords": ["free will", "determinism", "choice", "responsibility", "agency", "causation"]},
            {"pattern": "what is the meaning of life", "response": "The meaning of life has been addressed differently across traditions: Nihilism (no inherent meaning), Existentialism (we create our own meaning), Religious views (divine purpose), Absurdism (embrace meaninglessness), Hedonism (pleasure), Eudaimonia (flourishing). Perhaps meaning comes from relationships, achievement, contribution, or the journey itself.", "domain": "philosophy", "topics": ["existentialism", "ethics", "purpose"], "keywords": ["meaning", "life", "purpose", "existence", "nihilism", "absurdism"]},
            {"pattern": "what is truth", "response": "Truth theories: Correspondence (statements match reality), Coherence (truth fits within belief system), Pragmatic (truth is what works), Deflationary (saying 'X is true' just means X). Epistemology asks: Can we know truth? Challenges include skepticism, relativism, and the problem of infinite regress in justification.", "domain": "philosophy", "topics": ["epistemology", "logic", "knowledge"], "keywords": ["truth", "knowledge", "belief", "reality", "epistemology", "justification"]},
            {"pattern": "explain ethics and morality", "response": "Ethics studies right and wrong conduct. Major frameworks: Consequentialism (outcomes matter - utilitarianism), Deontology (duties/rules matter - Kant), Virtue Ethics (character matters - Aristotle), Care Ethics (relationships matter). Meta-ethics asks if moral facts exist. Applied ethics addresses specific issues like bioethics, business ethics, and environmental ethics.", "domain": "philosophy", "topics": ["ethics", "morality", "values"], "keywords": ["ethics", "morality", "right", "wrong", "virtue", "duty", "consequences"]},
            {"pattern": "what is knowledge", "response": "Traditional definition: Knowledge is justified true belief. Gettier problems challenged this with cases of justified true beliefs that aren't knowledge. Responses include adding conditions (no false lemmas, reliable process) or abandoning JTB. Types: A priori (independent of experience), A posteriori (from experience), Propositional (that), Procedural (how), Acquaintance (direct).", "domain": "philosophy", "topics": ["epistemology", "belief", "justification"], "keywords": ["knowledge", "belief", "truth", "justification", "epistemology", "gettier"]},
            {"pattern": "what is reality", "response": "Metaphysics studies the nature of reality. Key positions: Materialism (only physical exists), Idealism (only mind exists), Dualism (both exist), Pluralism (many types exist). Questions include: What exists fundamentally? Is reality mind-independent? Are abstract objects real? What is the nature of time and space? Is change real or illusion?", "domain": "philosophy", "topics": ["metaphysics", "existence", "ontology"], "keywords": ["reality", "existence", "metaphysics", "ontology", "materialism", "idealism"]},
            {"pattern": "explain socratic method", "response": "The Socratic method is a form of cooperative argumentative dialogue. Named after Socrates, it involves asking and answering questions to stimulate critical thinking and illuminate ideas. Key features: Claim ignorance, ask probing questions, expose contradictions, seek definitions. Goal is to reach better understanding through dialectic rather than lecture.", "domain": "philosophy", "topics": ["logic", "socrates", "dialogue"], "keywords": ["socratic", "questions", "dialogue", "critical thinking", "socrates", "dialectic"]},
        ]
        
        # General Knowledge patterns
        general_patterns = [
            {"pattern": "how does the internet work", "response": "The internet is a global network of interconnected computers using standardized protocols (TCP/IP). Data travels in packets through routers, following optimal paths. Key components: DNS (translates domains to IPs), HTTP/HTTPS (web protocols), servers (host content), ISPs (provide access). The web (websites) runs on top of the internet infrastructure.", "domain": "technology", "topics": ["networks", "computing", "web"], "keywords": ["internet", "network", "web", "tcp", "ip", "servers", "protocols"]},
            {"pattern": "what is artificial intelligence", "response": "Artificial Intelligence is the simulation of human intelligence by machines. Types: Narrow AI (specific tasks), General AI (human-level), Super AI (beyond human). Approaches: Machine Learning (learns from data), Deep Learning (neural networks), Symbolic AI (logic-based). Applications include image recognition, NLP, robotics, game playing, and autonomous vehicles.", "domain": "technology", "topics": ["computing", "machine learning", "automation"], "keywords": ["artificial intelligence", "ai", "machine learning", "neural networks", "automation", "robots"]},
            {"pattern": "explain democracy", "response": "Democracy is a system of government where power rests with the people. Types: Direct democracy (citizens vote on issues), Representative democracy (elected officials decide). Key principles: Popular sovereignty, political equality, rule of law, protection of rights, free elections, separation of powers. Originated in ancient Athens; modern democracies vary widely in structure.", "domain": "politics", "topics": ["government", "society", "rights"], "keywords": ["democracy", "government", "voting", "elections", "rights", "citizens", "representation"]},
            {"pattern": "what is economics", "response": "Economics studies how societies allocate scarce resources. Microeconomics focuses on individual decisions (supply/demand, pricing). Macroeconomics examines whole economies (GDP, inflation, unemployment). Key concepts: Opportunity cost, marginal analysis, incentives, market equilibrium. Schools include Classical, Keynesian, Monetarist, Austrian, and Behavioral economics.", "domain": "economics", "topics": ["markets", "society", "resources"], "keywords": ["economics", "supply", "demand", "markets", "gdp", "inflation", "resources"]},
            {"pattern": "how does memory work", "response": "Memory involves encoding, storing, and retrieving information. Types: Sensory (brief), Short-term/Working (limited capacity, ~7 items), Long-term (unlimited, permanent). Long-term divides into Explicit (declarative - facts/events) and Implicit (procedural - skills). Memory formation involves hippocampus; storage distributes across cortex. Sleep aids consolidation.", "domain": "psychology", "topics": ["brain", "cognition", "learning"], "keywords": ["memory", "brain", "learning", "encoding", "hippocampus", "recall", "cognition"]},
            {"pattern": "what is climate change", "response": "Climate change refers to long-term shifts in global temperatures and weather patterns. Current warming is primarily caused by human activities, especially burning fossil fuels (CO2 emissions). Effects include rising sea levels, extreme weather, ecosystem disruption, and biodiversity loss. Solutions involve reducing emissions, renewable energy, and adaptation strategies.", "domain": "science", "topics": ["environment", "weather", "earth"], "keywords": ["climate change", "global warming", "carbon", "emissions", "environment", "temperature", "weather"]},
            {"pattern": "explain the renaissance", "response": "The Renaissance (14th-17th century) was a cultural rebirth originating in Italy. Key features: Revival of classical learning, humanism (focus on human potential), artistic innovation (perspective, realism), scientific advances. Major figures: Leonardo da Vinci, Michelangelo, Galileo, Machiavelli. It marked transition from medieval to modern thinking.", "domain": "history", "topics": ["culture", "art", "europe"], "keywords": ["renaissance", "art", "humanism", "italy", "culture", "history", "science"]},
            {"pattern": "what is language", "response": "Language is a structured system of communication using symbols (words/signs) governed by rules (grammar). Key components: Phonology (sounds), Morphology (word structure), Syntax (sentence structure), Semantics (meaning), Pragmatics (context). Unique human features: Recursion, displacement, productivity. Theories debate whether it's innate (Chomsky) or learned.", "domain": "linguistics", "topics": ["communication", "cognition", "culture"], "keywords": ["language", "grammar", "communication", "syntax", "semantics", "linguistics", "words"]},
            {"pattern": "how do computers work", "response": "Computers process information using binary (0s and 1s). Key components: CPU (processes instructions), RAM (temporary memory), Storage (permanent data), I/O devices. Programs are instructions translated to machine code. Operating systems manage hardware and software. Modern computers execute billions of operations per second through transistors in integrated circuits.", "domain": "technology", "topics": ["computing", "hardware", "software"], "keywords": ["computer", "cpu", "memory", "binary", "hardware", "software", "processing"]},
            {"pattern": "what is mathematics", "response": "Mathematics is the study of quantity, structure, space, and change using logic and abstraction. Branches include: Arithmetic, Algebra, Geometry, Calculus, Statistics, Number Theory. Math is foundational to science, engineering, and technology. Debate exists whether math is discovered (Platonism) or invented (formalism). It provides a universal language for precise reasoning.", "domain": "mathematics", "topics": ["logic", "numbers", "patterns"], "keywords": ["mathematics", "math", "numbers", "algebra", "geometry", "logic", "patterns"]},
        ]
        
        # Reasoning and logic patterns
        reasoning_patterns = [
            {"pattern": "what is logic", "response": "Logic is the systematic study of valid reasoning. Deductive logic guarantees conclusions from premises (syllogisms). Inductive logic infers general rules from specific cases (probable but not certain). Formal logic uses symbolic notation. Key concepts: Validity (structure), Soundness (valid + true premises), Fallacies (reasoning errors).", "domain": "philosophy", "topics": ["reasoning", "arguments", "truth"], "keywords": ["logic", "reasoning", "deduction", "induction", "arguments", "validity", "fallacies"]},
            {"pattern": "explain cause and effect", "response": "Causation is the relationship where one event (cause) produces another (effect). Criteria: Temporal precedence (cause before effect), Correlation, Mechanism, Elimination of alternatives. Challenges: Correlation isn't causation, confounding variables, reverse causation. Counterfactual analysis asks: Would the effect occur without the cause?", "domain": "philosophy", "topics": ["metaphysics", "science", "reasoning"], "keywords": ["cause", "effect", "causation", "correlation", "mechanism", "counterfactual"]},
            {"pattern": "what is probability", "response": "Probability measures likelihood of events (0 to 1). Interpretations: Frequentist (long-run frequency), Bayesian (degree of belief), Classical (equally likely outcomes). Key rules: Addition (P(A or B)), Multiplication (P(A and B)), Conditional probability (P(A|B)), Bayes' theorem. Applications span statistics, science, gambling, decision-making.", "domain": "mathematics", "topics": ["statistics", "uncertainty", "chance"], "keywords": ["probability", "chance", "statistics", "likelihood", "bayes", "uncertainty", "random"]},
            {"pattern": "how to think critically", "response": "Critical thinking is disciplined analysis and evaluation of information. Key skills: 1) Question assumptions, 2) Evaluate evidence, 3) Consider alternatives, 4) Identify biases, 5) Draw reasoned conclusions, 6) Recognize fallacies. Apply: What's the source? What's the evidence? What's assumed? What are counterarguments? What follows logically?", "domain": "philosophy", "topics": ["reasoning", "epistemology", "skills"], "keywords": ["critical thinking", "analysis", "reasoning", "evidence", "bias", "evaluation"]},
            {"pattern": "what are cognitive biases", "response": "Cognitive biases are systematic thinking errors. Examples: Confirmation bias (favor supporting evidence), Anchoring (over-rely on first information), Availability heuristic (judge by ease of recall), Dunning-Kruger (incompetent overestimate ability), Sunk cost fallacy (continue failing investments). Awareness and deliberate analysis can mitigate biases.", "domain": "psychology", "topics": ["cognition", "reasoning", "errors"], "keywords": ["cognitive bias", "bias", "thinking", "heuristics", "psychology", "errors", "judgment"]},
        ]
        
        # Analogical patterns for reasoning
        analogy_patterns = [
            {"pattern": "atom is like solar system", "response": "The atom-solar system analogy: electrons orbit the nucleus like planets orbit the sun. Both have a massive center with smaller objects in orbit due to attractive forces. Limitations: Electrons don't have fixed orbits (probability clouds), quantum effects have no solar analog, electrons can jump energy levels.", "domain": "science", "topics": ["physics", "analogies", "atoms"], "keywords": ["atom", "solar system", "analogy", "electrons", "planets", "orbit"]},
            {"pattern": "brain is like computer", "response": "Brain-computer analogy: Both process information, have memory (storage), and produce outputs from inputs. Brain has neurons (transistors), synapses (connections), patterns (programs). Limitations: Brain is analog/parallel vs digital/serial, brain is plastic and self-organizing, consciousness has no clear computational analog.", "domain": "technology", "topics": ["cognition", "analogies", "computing"], "keywords": ["brain", "computer", "analogy", "neurons", "processing", "memory"]},
            {"pattern": "evolution is like algorithm", "response": "Evolution as optimization algorithm: Random variation generates candidates, selection filters for fitness, iteration improves solutions over generations. Genetic algorithms explicitly use this. Limitations: Evolution has no goal, 'fitness' is contextual, local optima can trap populations, historical contingency shapes outcomes.", "domain": "science", "topics": ["biology", "computing", "analogies"], "keywords": ["evolution", "algorithm", "optimization", "selection", "fitness", "genetic"]},
        ]
        
        # Combine all patterns
        all_patterns = (
            science_patterns + 
            philosophy_patterns + 
            general_patterns + 
            reasoning_patterns +
            analogy_patterns
        )
        
        # Add IDs
        for p in all_patterns:
            p['id'] = str(uuid.uuid4())
            p['success_rate'] = 0.9
            p['usage_count'] = 0
            p['last_used'] = time.time()
            p['confidence'] = 0.85
            
        return all_patterns
        
    def add_pattern(self, pattern_data: Dict) -> Pattern:
        """Add a new pattern to the database"""
        if 'id' not in pattern_data:
            pattern_data['id'] = str(uuid.uuid4())
            
        # Extract keywords from pattern if not provided
        if 'keywords' not in pattern_data or not pattern_data['keywords']:
            pattern_data['keywords'] = self._extract_keywords(
                pattern_data.get('pattern', '') + ' ' + pattern_data.get('response', '')
            )
            
        pattern = Pattern.from_dict(pattern_data)
        self.patterns[pattern.id] = pattern
        
        # Update indices
        self.domain_index[pattern.domain].append(pattern.id)
        for topic in pattern.topics:
            self.topic_index[topic].append(pattern.id)
        for keyword in pattern.keywords:
            self.keyword_index[keyword.lower()].append(pattern.id)
            
        return pattern
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter common words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'what', 'when', 'where', 'which', 'while', 'being', 'about', 'would', 'could', 'should', 'there', 'these', 'those', 'such', 'more', 'some', 'other', 'than', 'into', 'only', 'very', 'also', 'just', 'most', 'both', 'each', 'same'}
        keywords = [w for w in set(words) if w not in stop_words][:10]
        return keywords
        
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
        
    def get_patterns_by_domain(self, domain: str) -> List[Pattern]:
        """Get all patterns in a domain"""
        pattern_ids = self.domain_index.get(domain, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        
    def get_patterns_by_topic(self, topic: str) -> List[Pattern]:
        """Get all patterns for a topic"""
        pattern_ids = self.topic_index.get(topic, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        
    def get_patterns_by_keywords(self, keywords: List[str]) -> List[Pattern]:
        """Get patterns matching any of the keywords"""
        pattern_ids = set()
        for kw in keywords:
            pattern_ids.update(self.keyword_index.get(kw.lower(), []))
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        
    def update_pattern_stats(self, pattern_id: str, success: bool = True):
        """Update pattern usage statistics"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = time.time()
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            result = 1.0 if success else 0.0
            pattern.success_rate = alpha * result + (1 - alpha) * pattern.success_rate
            
    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns"""
        return list(self.patterns.values())
        
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        patterns = list(self.patterns.values())
        if not patterns:
            return {"total": 0, "domains": {}, "avg_success_rate": 0}
            
        return {
            "total_patterns": len(patterns),
            "domains": {d: len(ids) for d, ids in self.domain_index.items()},
            "topics": len(self.topic_index),
            "avg_success_rate": sum(p.success_rate for p in patterns) / len(patterns),
            "total_usages": sum(p.usage_count for p in patterns),
            "most_used": sorted(patterns, key=lambda p: p.usage_count, reverse=True)[:5]
        }
        
    def search(self, query: str, limit: int = 10) -> List[Pattern]:
        """Search patterns by query string"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored_patterns = []
        for pattern in self.patterns.values():
            score = 0
            pattern_text = (pattern.pattern + ' ' + ' '.join(pattern.keywords)).lower()
            
            # Exact match in pattern
            if query_lower in pattern.pattern.lower():
                score += 10
                
            # Keyword matches
            for word in query_words:
                if word in pattern_text:
                    score += 2
                if word in pattern.keywords:
                    score += 3
                    
            # Domain/topic relevance
            for word in query_words:
                if word == pattern.domain:
                    score += 5
                if word in pattern.topics:
                    score += 3
                    
            if score > 0:
                scored_patterns.append((pattern, score))
                
        # Sort by score and return
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_patterns[:limit]]
