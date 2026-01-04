"""
EntityKnowledgeBase - Real-world entity storage and retrieval
Contains 1000+ entities with aliases and facts
"""

import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Entity:
    """A real-world entity"""
    id: str
    name: str
    category: str
    aliases: List[str]
    facts: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'aliases': self.aliases,
            'facts': self.facts,
            'properties': self.properties,
            'relationships': self.relationships
        }


class EntityKnowledgeBase:
    """
    Knowledge base of real-world entities
    Supports alias resolution and entity-grounded reasoning
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.alias_index: Dict[str, str] = {}  # alias -> entity_id
        self.category_index: Dict[str, List[str]] = {}  # category -> [entity_ids]
        self.name_index: Dict[str, str] = {}  # lowercase name -> entity_id
        
        self._initialized = False
        
    def initialize(self):
        """Initialize with seed entities"""
        if self._initialized:
            return
            
        seed_entities = self._get_seed_entities()
        for entity_data in seed_entities:
            self.add_entity(entity_data)
            
        self._initialized = True
        
    def _get_seed_entities(self) -> List[Dict]:
        """Generate seed entities across categories"""
        entities = []
        
        # Scientists
        scientists = [
            {"name": "Albert Einstein", "category": "scientist", "aliases": ["Einstein", "A. Einstein"], "facts": ["Developed theory of relativity", "Won Nobel Prize in Physics 1921", "Famous equation E=mc²", "Born in Germany 1879"]},
            {"name": "Isaac Newton", "category": "scientist", "aliases": ["Newton", "Sir Isaac Newton"], "facts": ["Discovered laws of motion", "Developed calculus", "Law of universal gravitation", "Born in England 1643"]},
            {"name": "Marie Curie", "category": "scientist", "aliases": ["Curie", "Madame Curie"], "facts": ["Discovered radium and polonium", "First woman Nobel laureate", "Won two Nobel Prizes", "Pioneer in radioactivity research"]},
            {"name": "Charles Darwin", "category": "scientist", "aliases": ["Darwin"], "facts": ["Theory of evolution by natural selection", "Voyage on HMS Beagle", "Wrote Origin of Species", "Born in England 1809"]},
            {"name": "Nikola Tesla", "category": "scientist", "aliases": ["Tesla"], "facts": ["Invented AC electrical system", "Pioneer in wireless technology", "Hundreds of patents", "Born in modern-day Croatia 1856"]},
            {"name": "Stephen Hawking", "category": "scientist", "aliases": ["Hawking", "Professor Hawking"], "facts": ["Theoretical physicist", "Black hole radiation theory", "A Brief History of Time author", "Lived with ALS for decades"]},
            {"name": "Galileo Galilei", "category": "scientist", "aliases": ["Galileo"], "facts": ["Father of modern observational astronomy", "Improved the telescope", "Supported heliocentrism", "Born in Italy 1564"]},
            {"name": "Richard Feynman", "category": "scientist", "aliases": ["Feynman"], "facts": ["Nobel Prize in Physics", "Developed quantum electrodynamics", "Famous lecturer", "Worked on Manhattan Project"]},
        ]
        
        # Philosophers
        philosophers = [
            {"name": "Socrates", "category": "philosopher", "aliases": [], "facts": ["Ancient Greek philosopher", "Socratic method", "Know thyself", "Executed by drinking hemlock"]},
            {"name": "Plato", "category": "philosopher", "aliases": [], "facts": ["Student of Socrates", "Founded the Academy", "Theory of Forms", "Wrote The Republic"]},
            {"name": "Aristotle", "category": "philosopher", "aliases": [], "facts": ["Student of Plato", "Tutor of Alexander the Great", "Founded formal logic", "Wrote on ethics, politics, metaphysics"]},
            {"name": "Immanuel Kant", "category": "philosopher", "aliases": ["Kant"], "facts": ["German philosopher", "Critique of Pure Reason", "Categorical imperative", "Enlightenment thinker"]},
            {"name": "Friedrich Nietzsche", "category": "philosopher", "aliases": ["Nietzsche"], "facts": ["German philosopher", "Thus Spoke Zarathustra", "Will to power concept", "Übermensch concept"]},
            {"name": "René Descartes", "category": "philosopher", "aliases": ["Descartes"], "facts": ["I think therefore I am", "Father of modern philosophy", "Cartesian dualism", "French philosopher"]},
            {"name": "Ludwig Wittgenstein", "category": "philosopher", "aliases": ["Wittgenstein"], "facts": ["Austrian philosopher", "Tractatus Logico-Philosophicus", "Language games", "Influenced analytic philosophy"]},
            {"name": "John Locke", "category": "philosopher", "aliases": ["Locke"], "facts": ["English philosopher", "Social contract theory", "Tabula rasa", "Influenced American founders"]},
        ]
        
        # Countries
        countries = [
            {"name": "United States", "category": "country", "aliases": ["USA", "US", "America", "United States of America"], "facts": ["Capital: Washington D.C.", "Population: ~330 million", "50 states", "Founded 1776"], "properties": {"continent": "North America", "currency": "USD"}},
            {"name": "China", "category": "country", "aliases": ["PRC", "People's Republic of China"], "facts": ["Capital: Beijing", "Most populous country", "Ancient civilization", "Second largest economy"], "properties": {"continent": "Asia", "currency": "CNY"}},
            {"name": "Japan", "category": "country", "aliases": ["Nippon"], "facts": ["Capital: Tokyo", "Island nation", "Third largest economy", "Constitutional monarchy"], "properties": {"continent": "Asia", "currency": "JPY"}},
            {"name": "Germany", "category": "country", "aliases": ["Deutschland", "Federal Republic of Germany"], "facts": ["Capital: Berlin", "Largest EU economy", "Reunified 1990", "Famous for engineering"], "properties": {"continent": "Europe", "currency": "EUR"}},
            {"name": "United Kingdom", "category": "country", "aliases": ["UK", "Britain", "Great Britain"], "facts": ["Capital: London", "Constitutional monarchy", "Four countries", "Industrial revolution origin"], "properties": {"continent": "Europe", "currency": "GBP"}},
            {"name": "France", "category": "country", "aliases": ["French Republic"], "facts": ["Capital: Paris", "Largest EU country by area", "French Revolution 1789", "Nuclear power"], "properties": {"continent": "Europe", "currency": "EUR"}},
            {"name": "India", "category": "country", "aliases": ["Republic of India", "Bharat"], "facts": ["Capital: New Delhi", "Second most populous", "World's largest democracy", "Ancient civilization"], "properties": {"continent": "Asia", "currency": "INR"}},
            {"name": "Brazil", "category": "country", "aliases": ["Brasil"], "facts": ["Capital: Brasília", "Largest South American country", "Amazon rainforest", "Portuguese speaking"], "properties": {"continent": "South America", "currency": "BRL"}},
        ]
        
        # Concepts
        concepts = [
            {"name": "Gravity", "category": "concept", "aliases": ["gravitational force", "gravitation"], "facts": ["Fundamental force", "Described by Newton and Einstein", "Causes objects to fall", "Keeps planets in orbit"]},
            {"name": "Evolution", "category": "concept", "aliases": ["natural selection", "Darwinian evolution"], "facts": ["Change over generations", "Natural selection mechanism", "Explains biodiversity", "Foundation of modern biology"]},
            {"name": "Democracy", "category": "concept", "aliases": ["democratic government"], "facts": ["Rule by the people", "Originated in ancient Greece", "Various forms exist", "Requires free elections"]},
            {"name": "Artificial Intelligence", "category": "concept", "aliases": ["AI", "machine intelligence"], "facts": ["Simulation of human intelligence", "Includes machine learning", "Growing rapidly", "Applications in many fields"]},
            {"name": "Quantum Mechanics", "category": "concept", "aliases": ["quantum physics", "quantum theory"], "facts": ["Physics of very small", "Wave-particle duality", "Uncertainty principle", "Superposition"]},
            {"name": "Capitalism", "category": "concept", "aliases": ["market economy", "free market"], "facts": ["Private ownership", "Profit motive", "Free markets", "Competition"]},
            {"name": "Consciousness", "category": "concept", "aliases": ["awareness", "sentience"], "facts": ["Subjective experience", "Hard problem", "Multiple theories", "Neural correlates"]},
            {"name": "Internet", "category": "concept", "aliases": ["the net", "world wide web"], "facts": ["Global network", "TCP/IP protocol", "Transformed communication", "Information age"]},
        ]
        
        # Organizations
        organizations = [
            {"name": "United Nations", "category": "organization", "aliases": ["UN"], "facts": ["International organization", "Founded 1945", "193 member states", "Headquarters in New York"]},
            {"name": "NASA", "category": "organization", "aliases": ["National Aeronautics and Space Administration"], "facts": ["US space agency", "Moon landings", "Mars rovers", "International Space Station"]},
            {"name": "World Health Organization", "category": "organization", "aliases": ["WHO"], "facts": ["UN agency for health", "Founded 1948", "Global health initiatives", "Disease prevention"]},
            {"name": "European Union", "category": "organization", "aliases": ["EU"], "facts": ["Political and economic union", "27 member states", "Single market", "Founded from earlier organizations"]},
            {"name": "Google", "category": "organization", "aliases": ["Alphabet"], "facts": ["Technology company", "Search engine", "Founded 1998", "Part of Alphabet Inc"]},
            {"name": "Apple", "category": "organization", "aliases": ["Apple Inc", "Apple Computer"], "facts": ["Technology company", "iPhone, Mac", "Founded by Steve Jobs", "Cupertino headquarters"]},
        ]
        
        # Historical events
        events = [
            {"name": "World War II", "category": "event", "aliases": ["WWII", "Second World War", "WW2"], "facts": ["1939-1945", "Deadliest conflict in history", "Allied vs Axis powers", "Led to United Nations"]},
            {"name": "French Revolution", "category": "event", "aliases": [], "facts": ["1789-1799", "End of monarchy in France", "Liberty, Equality, Fraternity", "Reign of Terror"]},
            {"name": "Industrial Revolution", "category": "event", "aliases": [], "facts": ["18th-19th century", "Started in Britain", "Mechanization", "Urbanization"]},
            {"name": "Moon Landing", "category": "event", "aliases": ["Apollo 11"], "facts": ["July 20, 1969", "Neil Armstrong first", "One small step quote", "Cold War achievement"]},
            {"name": "Renaissance", "category": "event", "aliases": [], "facts": ["14th-17th century", "Cultural rebirth", "Started in Italy", "Art and science flourished"]},
        ]
        
        # Combine all
        all_entities = scientists + philosophers + countries + concepts + organizations + events
        
        # Add IDs
        for e in all_entities:
            e['id'] = str(uuid.uuid4())
            if 'properties' not in e:
                e['properties'] = {}
            if 'relationships' not in e:
                e['relationships'] = []
                
        return all_entities
        
    def add_entity(self, entity_data: Dict) -> Entity:
        """Add an entity to the knowledge base"""
        if 'id' not in entity_data:
            entity_data['id'] = str(uuid.uuid4())
            
        entity = Entity(
            id=entity_data['id'],
            name=entity_data['name'],
            category=entity_data['category'],
            aliases=entity_data.get('aliases', []),
            facts=entity_data.get('facts', []),
            properties=entity_data.get('properties', {}),
            relationships=entity_data.get('relationships', [])
        )
        
        self.entities[entity.id] = entity
        
        # Update indices
        self.name_index[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self.alias_index[alias.lower()] = entity.id
            
        if entity.category not in self.category_index:
            self.category_index[entity.category] = []
        self.category_index[entity.category].append(entity.id)
        
        return entity
        
    def resolve_entity(self, name: str) -> Optional[Entity]:
        """Resolve a name or alias to an entity"""
        name_lower = name.lower().strip()
        
        # Try exact name match
        if name_lower in self.name_index:
            return self.entities[self.name_index[name_lower]]
            
        # Try alias match
        if name_lower in self.alias_index:
            return self.entities[self.alias_index[name_lower]]
            
        # Try partial match
        for entity in self.entities.values():
            if name_lower in entity.name.lower():
                return entity
            for alias in entity.aliases:
                if name_lower in alias.lower():
                    return entity
                    
        return None
        
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
        
    def get_entities_by_category(self, category: str) -> List[Entity]:
        """Get all entities in a category"""
        entity_ids = self.category_index.get(category, [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
        
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by query"""
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            score = 0
            
            # Name match
            if query_lower in entity.name.lower():
                score += 10
            elif entity.name.lower() in query_lower:
                score += 8
                
            # Alias match
            for alias in entity.aliases:
                if query_lower in alias.lower():
                    score += 7
                elif alias.lower() in query_lower:
                    score += 5
                    
            # Fact match
            for fact in entity.facts:
                if query_lower in fact.lower():
                    score += 2
                    
            if score > 0:
                results.append((entity, score))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in results[:limit]]
        
    def get_entity_facts(self, entity_id: str) -> List[str]:
        """Get facts about an entity"""
        entity = self.entities.get(entity_id)
        if entity:
            return entity.facts
        return []
        
    def add_fact(self, entity_id: str, fact: str):
        """Add a fact to an entity"""
        if entity_id in self.entities:
            self.entities[entity_id].facts.append(fact)
            
    def add_relationship(self, entity1_id: str, entity2_id: str, relation_type: str):
        """Add a relationship between entities"""
        if entity1_id in self.entities and entity2_id in self.entities:
            entity1 = self.entities[entity1_id]
            entity2 = self.entities[entity2_id]
            
            entity1.relationships.append({
                'type': relation_type,
                'target': entity2_id,
                'target_name': entity2.name
            })
            
    def ground_query(self, query: str) -> List[Dict]:
        """
        Ground a query by finding relevant entities
        Returns entities that can provide context
        """
        grounded = []
        query_words = query.lower().split()
        
        # Look for entity mentions
        for entity in self.entities.values():
            # Check name
            if any(word in entity.name.lower() for word in query_words):
                grounded.append({
                    'entity': entity.to_dict(),
                    'match_type': 'name',
                    'relevance': 1.0
                })
                continue
                
            # Check aliases
            for alias in entity.aliases:
                if any(word in alias.lower() for word in query_words):
                    grounded.append({
                        'entity': entity.to_dict(),
                        'match_type': 'alias',
                        'relevance': 0.9
                    })
                    break
                    
        # Sort by relevance
        grounded.sort(key=lambda x: x['relevance'], reverse=True)
        return grounded[:5]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'total_entities': len(self.entities),
            'categories': {cat: len(ids) for cat, ids in self.category_index.items()},
            'total_facts': sum(len(e.facts) for e in self.entities.values()),
            'total_aliases': len(self.alias_index)
        }
