"""
Text-to-Visual Decomposition Engine
Decomposes text descriptions into visual concepts using SI reasoning strategies
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class VisualConcept:
    """A concept extracted from text with visual attributes"""
    entity: str
    attributes: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str]] = field(default_factory=tuple)  # (relation_type, target_entity)
    position_hint: Optional[str] = None  # top, bottom, left, right, center
    size_hint: Optional[str] = None  # small, medium, large
    importance: float = 1.0  # 0-1, how central to the scene
    
    def to_dict(self) -> Dict:
        return {
            'entity': self.entity,
            'attributes': self.attributes,
            'relations': list(self.relations),
            'position_hint': self.position_hint,
            'size_hint': self.size_hint,
            'importance': self.importance
        }


@dataclass
class SceneDecomposition:
    """Complete decomposition of a scene description"""
    original_text: str
    concepts: List[VisualConcept]
    background_type: str = "default"
    mood: str = "neutral"  # happy, sad, dramatic, peaceful, etc.
    time_of_day: Optional[str] = None  # day, night, sunset, sunrise
    weather: Optional[str] = None  # sunny, rainy, cloudy, snowy
    
    def to_dict(self) -> Dict:
        return {
            'original_text': self.original_text,
            'concepts': [c.to_dict() for c in self.concepts],
            'background_type': self.background_type,
            'mood': self.mood,
            'time_of_day': self.time_of_day,
            'weather': self.weather
        }


class TextToVisualDecomposer:
    """
    Decomposes text descriptions into structured visual concepts
    Uses pattern-based NLP without neural networks
    """
    
    def __init__(self, pattern_database=None, entity_knowledge_base=None):
        self.pattern_db = pattern_database
        self.entity_kb = entity_knowledge_base
        
        # Color mappings
        self.color_keywords = {
            'red': '#DC2626', 'blue': '#3B82F6', 'green': '#22C55E',
            'yellow': '#FBBF24', 'orange': '#F97316', 'purple': '#8B5CF6',
            'pink': '#EC4899', 'black': '#1F2937', 'white': '#FFFFFF',
            'gray': '#6B7280', 'grey': '#6B7280', 'brown': '#8B4513',
            'gold': '#F59E0B', 'silver': '#9CA3AF', 'cyan': '#06B6D4',
            'teal': '#14B8A6', 'indigo': '#6366F1', 'violet': '#8B5CF6'
        }
        
        # Size keywords
        self.size_keywords = {
            'tiny': 'small', 'small': 'small', 'little': 'small',
            'medium': 'medium', 'normal': 'medium', 'average': 'medium',
            'big': 'large', 'large': 'large', 'huge': 'large', 'giant': 'large'
        }
        
        # Position keywords
        self.position_keywords = {
            'top': 'top', 'above': 'top', 'upper': 'top', 'over': 'top',
            'bottom': 'bottom', 'below': 'bottom', 'lower': 'bottom', 'under': 'bottom',
            'left': 'left', 'right': 'right',
            'center': 'center', 'middle': 'center',
            'background': 'background', 'foreground': 'foreground',
            'behind': 'background', 'front': 'foreground'
        }
        
        # Relation patterns
        self.relation_patterns = [
            (r'(\w+)\s+on\s+(?:a\s+|the\s+)?(\w+)', 'on'),
            (r'(\w+)\s+in\s+(?:a\s+|the\s+)?(\w+)', 'in'),
            (r'(\w+)\s+near\s+(?:a\s+|the\s+)?(\w+)', 'near'),
            (r'(\w+)\s+next\s+to\s+(?:a\s+|the\s+)?(\w+)', 'beside'),
            (r'(\w+)\s+beside\s+(?:a\s+|the\s+)?(\w+)', 'beside'),
            (r'(\w+)\s+under\s+(?:a\s+|the\s+)?(\w+)', 'under'),
            (r'(\w+)\s+over\s+(?:a\s+|the\s+)?(\w+)', 'over'),
            (r'(\w+)\s+with\s+(?:a\s+|the\s+)?(\w+)', 'with'),
            (r'(\w+)\s+and\s+(?:a\s+|the\s+)?(\w+)', 'with')
        ]
        
        # Time of day keywords
        self.time_keywords = {
            'day': 'day', 'daytime': 'day', 'sunny': 'day', 'bright': 'day',
            'night': 'night', 'nighttime': 'night', 'dark': 'night', 'midnight': 'night',
            'sunset': 'sunset', 'dusk': 'sunset', 'evening': 'sunset',
            'sunrise': 'sunrise', 'dawn': 'sunrise', 'morning': 'sunrise'
        }
        
        # Weather keywords
        self.weather_keywords = {
            'sunny': 'sunny', 'clear': 'sunny', 'bright': 'sunny',
            'rainy': 'rainy', 'rain': 'rainy', 'raining': 'rainy', 'wet': 'rainy',
            'cloudy': 'cloudy', 'clouds': 'cloudy', 'overcast': 'cloudy',
            'snowy': 'snowy', 'snow': 'snowy', 'winter': 'snowy',
            'stormy': 'stormy', 'storm': 'stormy', 'thunder': 'stormy'
        }
        
        # Entity to visual pattern mapping
        self.entity_pattern_map = {
            'car': ['car', 'vehicle', 'automobile'],
            'house': ['house', 'home', 'building', 'cottage'],
            'tree': ['tree', 'plant', 'forest', 'pine'],
            'person': ['person', 'man', 'woman', 'human', 'people'],
            'dog': ['dog', 'puppy', 'pet', 'canine'],
            'sun': ['sun', 'sunny'],
            'cloud': ['cloud', 'clouds', 'cloudy'],
            'mountain': ['mountain', 'hill', 'peak'],
            'water': ['water', 'ocean', 'sea', 'lake', 'river'],
            'road': ['road', 'street', 'highway', 'path'],
            'boat': ['boat', 'ship', 'sailboat', 'yacht'],
            'grass': ['grass', 'lawn', 'field', 'meadow', 'park'],
            'sky': ['sky', 'atmosphere'],
            'rain': ['rain', 'rainy', 'raining'],
            'sunset_sky': ['sunset', 'dusk', 'evening sky']
        }
        
    def decompose(self, text: str) -> SceneDecomposition:
        """
        Decompose text description into visual concepts
        
        Input: "a red car on a rainy mountain road"
        Output: SceneDecomposition with entities and relations
        """
        text_lower = text.lower().strip()
        
        # Extract concepts
        concepts = self._extract_concepts(text_lower)
        
        # Extract relations between concepts
        self._extract_relations(text_lower, concepts)
        
        # Determine scene properties
        background = self._determine_background(text_lower, concepts)
        mood = self._determine_mood(text_lower)
        time_of_day = self._extract_time_of_day(text_lower)
        weather = self._extract_weather(text_lower)
        
        # Assign positions based on semantic understanding
        self._assign_positions(concepts, background)
        
        return SceneDecomposition(
            original_text=text,
            concepts=concepts,
            background_type=background,
            mood=mood,
            time_of_day=time_of_day,
            weather=weather
        )
    
    def _extract_concepts(self, text: str) -> List[VisualConcept]:
        """Extract visual concepts from text"""
        concepts = []
        words = text.split()
        
        # Find entity words
        entity_words = set()
        for pattern_name, keywords in self.entity_pattern_map.items():
            for keyword in keywords:
                if keyword in text:
                    entity_words.add(keyword)
        
        # Also check for direct pattern matches
        if self.pattern_db:
            for pattern in self.pattern_db.get_all_patterns():
                if pattern.name.lower() in text:
                    entity_words.add(pattern.name.lower())
                for tag in pattern.semantic_tags:
                    if tag.lower() in text:
                        entity_words.add(tag.lower())
        
        # Create concepts for each entity
        processed = set()
        for entity in entity_words:
            if entity in processed:
                continue
            processed.add(entity)
            
            # Find attributes for this entity
            attributes = self._find_attributes_for_entity(text, entity)
            
            # Determine importance (entities mentioned first are more important)
            position = text.find(entity)
            importance = 1.0 - (position / max(len(text), 1)) * 0.3
            
            concept = VisualConcept(
                entity=self._normalize_entity(entity),
                attributes=attributes,
                importance=importance
            )
            concepts.append(concept)
        
        # Sort by importance
        concepts.sort(key=lambda c: c.importance, reverse=True)
        
        return concepts
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity name to pattern name"""
        for pattern_name, keywords in self.entity_pattern_map.items():
            if entity in keywords:
                return pattern_name
        return entity
    
    def _find_attributes_for_entity(self, text: str, entity: str) -> List[str]:
        """Find attributes (colors, sizes) associated with an entity"""
        attributes = []
        
        # Look for color words near entity
        entity_pos = text.find(entity)
        if entity_pos == -1:
            return attributes
        
        # Check words before entity (typically adjectives)
        before_text = text[:entity_pos].split()[-3:]  # Last 3 words before entity
        
        for word in before_text:
            # Check colors
            if word in self.color_keywords:
                attributes.append(f"color:{self.color_keywords[word]}")
            # Check sizes
            if word in self.size_keywords:
                attributes.append(f"size:{self.size_keywords[word]}")
        
        return attributes
    
    def _extract_relations(self, text: str, concepts: List[VisualConcept]):
        """Extract relations between concepts"""
        # Build entity name lookup
        entity_names = {c.entity for c in concepts}
        entity_lookup = {c.entity: c for c in concepts}
        
        # Check each relation pattern
        for pattern, relation_type in self.relation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                source_word, target_word = match
                
                # Normalize to entity names
                source = self._normalize_entity(source_word)
                target = self._normalize_entity(target_word)
                
                if source in entity_lookup and target in entity_lookup:
                    entity_lookup[source].relations = list(entity_lookup[source].relations) + [(relation_type, target)]
    
    def _determine_background(self, text: str, concepts: List[VisualConcept]) -> str:
        """Determine background type based on text and concepts"""
        entity_names = {c.entity for c in concepts}
        
        # Check for specific backgrounds
        if 'ocean' in text or 'sea' in text or 'beach' in text:
            return 'ocean'
        if 'mountain' in text or 'hill' in text:
            return 'mountain'
        if 'forest' in text or 'wood' in text:
            return 'forest'
        if 'city' in text or 'urban' in text:
            return 'city'
        if 'park' in text or 'garden' in text:
            return 'park'
        if 'road' in entity_names or 'street' in text:
            return 'road'
        if 'sky' in entity_names:
            return 'sky'
        
        # Default to outdoor scene
        return 'outdoor'
    
    def _determine_mood(self, text: str) -> str:
        """Determine mood from text"""
        mood_keywords = {
            'happy': ['happy', 'joyful', 'cheerful', 'bright', 'sunny', 'playful'],
            'sad': ['sad', 'gloomy', 'dark', 'lonely', 'melancholy'],
            'dramatic': ['dramatic', 'intense', 'stormy', 'powerful'],
            'peaceful': ['peaceful', 'calm', 'serene', 'quiet', 'tranquil'],
            'romantic': ['romantic', 'sunset', 'love', 'couple']
        }
        
        for mood, keywords in mood_keywords.items():
            if any(kw in text for kw in keywords):
                return mood
        
        return 'neutral'
    
    def _extract_time_of_day(self, text: str) -> Optional[str]:
        """Extract time of day from text"""
        for keyword, time in self.time_keywords.items():
            if keyword in text:
                return time
        return None
    
    def _extract_weather(self, text: str) -> Optional[str]:
        """Extract weather from text"""
        for keyword, weather in self.weather_keywords.items():
            if keyword in text:
                return weather
        return None
    
    def _assign_positions(self, concepts: List[VisualConcept], background: str):
        """Assign position hints to concepts based on semantic understanding"""
        for concept in concepts:
            # Use relations to determine position
            for relation_type, target in concept.relations:
                if relation_type == 'on':
                    concept.position_hint = 'on_' + target
                elif relation_type == 'under':
                    concept.position_hint = 'top'
                elif relation_type == 'over':
                    concept.position_hint = 'bottom'
            
            # Use entity knowledge for default positions
            if concept.position_hint is None:
                if concept.entity in ['sky', 'sun', 'cloud', 'sunset_sky']:
                    concept.position_hint = 'top'
                elif concept.entity in ['road', 'water', 'grass']:
                    concept.position_hint = 'bottom'
                elif concept.entity in ['mountain']:
                    concept.position_hint = 'middle'
                elif concept.entity in ['person', 'car', 'dog', 'house', 'tree', 'boat']:
                    concept.position_hint = 'center'
    
    def match_concepts_to_patterns(
        self, 
        decomposition: SceneDecomposition, 
        visual_pattern_db
    ) -> List[Dict[str, Any]]:
        """
        Match decomposed concepts to visual patterns
        Returns list of matched patterns with transformation info
        """
        matched = []
        
        for concept in decomposition.concepts:
            # Search patterns by entity name and tags
            patterns = visual_pattern_db.search_by_tags([concept.entity] + concept.attributes[:2])
            
            if patterns:
                best_pattern = patterns[0]
                
                # Apply attribute transformations
                color_override = None
                size_modifier = 1.0
                
                for attr in concept.attributes:
                    if attr.startswith('color:'):
                        color_override = attr.split(':')[1]
                    elif attr.startswith('size:'):
                        size = attr.split(':')[1]
                        size_modifier = {'small': 0.5, 'medium': 1.0, 'large': 1.5}.get(size, 1.0)
                
                matched.append({
                    'concept': concept.to_dict(),
                    'pattern': best_pattern.to_dict(),
                    'pattern_id': best_pattern.id,
                    'color_override': color_override,
                    'size_modifier': size_modifier,
                    'position_hint': concept.position_hint,
                    'importance': concept.importance
                })
            else:
                # No pattern found, use placeholder
                matched.append({
                    'concept': concept.to_dict(),
                    'pattern': None,
                    'pattern_id': None,
                    'color_override': None,
                    'size_modifier': 1.0,
                    'position_hint': concept.position_hint,
                    'importance': concept.importance
                })
        
        return matched
