"""
Scene Graph Composition Engine
Builds and arranges visual elements using reasoning strategies
"""

import uuid
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints between scene elements"""
    SPATIAL = "spatial"  # Position-based
    CAUSAL = "causal"  # Cause-effect
    TEMPORAL = "temporal"  # Time-based
    AESTHETIC = "aesthetic"  # Visual harmony


class SpatialRelationType(Enum):
    """Spatial relationships"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    OVERLAPS = "overlaps"
    TOUCHES = "touches"
    NEAR = "near"
    FAR = "far"
    ON = "on"


@dataclass
class AffineTransform:
    """2D affine transformation"""
    translate_x: float = 0.0
    translate_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0  # Degrees
    
    def to_svg_transform(self) -> str:
        """Convert to SVG transform string"""
        transforms = []
        if self.translate_x != 0 or self.translate_y != 0:
            transforms.append(f"translate({self.translate_x},{self.translate_y})")
        if self.scale_x != 1 or self.scale_y != 1:
            transforms.append(f"scale({self.scale_x},{self.scale_y})")
        if self.rotation != 0:
            transforms.append(f"rotate({self.rotation})")
        return " ".join(transforms) if transforms else ""
    
    def to_dict(self) -> Dict:
        return {
            'translate_x': self.translate_x,
            'translate_y': self.translate_y,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'rotation': self.rotation
        }


@dataclass
class Constraint:
    """A constraint between scene elements"""
    type: ConstraintType
    relation: str
    source_id: str
    target_id: Optional[str] = None
    strength: float = 1.0  # 0-1
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'relation': self.relation,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'strength': self.strength,
            'parameters': self.parameters
        }


@dataclass
class SceneGraphNode:
    """A node in the scene graph representing a visual element"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_id: Optional[str] = None
    name: str = "unnamed"
    children: List['SceneGraphNode'] = field(default_factory=list)
    transform: AffineTransform = field(default_factory=AffineTransform)
    attributes: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    layer: int = 0  # Z-index for rendering order
    visible: bool = True
    
    def add_child(self, child: 'SceneGraphNode'):
        """Add a child node"""
        self.children.append(child)
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint"""
        self.constraints.append(constraint)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'name': self.name,
            'children': [c.to_dict() for c in self.children],
            'transform': self.transform.to_dict(),
            'attributes': self.attributes,
            'constraints': [c.to_dict() for c in self.constraints],
            'layer': self.layer,
            'visible': self.visible
        }


class SceneComposer:
    """
    Composes visual elements into a coherent scene graph
    Uses reasoning strategies for intelligent placement
    """
    
    def __init__(self, visual_pattern_db, reasoning_engine=None, self_model=None):
        self.pattern_db = visual_pattern_db
        self.reasoning_engine = reasoning_engine
        self.self_model = self_model
        
        # Canvas settings
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Layer definitions
        self.layer_map = {
            'background': 0,
            'far_background': 1,
            'mid_background': 2,
            'foreground': 3,
            'main_subject': 4,
            'overlay': 5
        }
        
        # Position presets (normalized 0-1)
        self.position_presets = {
            'top': (0.5, 0.15),
            'top_left': (0.2, 0.15),
            'top_right': (0.8, 0.15),
            'middle': (0.5, 0.5),
            'center': (0.5, 0.5),
            'bottom': (0.5, 0.85),
            'bottom_left': (0.2, 0.85),
            'bottom_right': (0.8, 0.85),
            'left': (0.2, 0.5),
            'right': (0.8, 0.5)
        }
        
    def compose_scene(
        self, 
        matched_patterns: List[Dict[str, Any]],
        decomposition: 'SceneDecomposition'
    ) -> SceneGraphNode:
        """
        Build a scene graph from matched patterns and decomposition
        """
        # Create root node
        root = SceneGraphNode(
            name="scene_root",
            layer=0
        )
        
        # Add background based on decomposition
        background_node = self._create_background_node(decomposition)
        if background_node:
            root.add_child(background_node)
        
        # Add weather effects if applicable
        if decomposition.weather:
            weather_node = self._create_weather_node(decomposition.weather)
            if weather_node:
                weather_node.layer = self.layer_map['overlay']
                root.add_child(weather_node)
        
        # Sort patterns by importance and layer
        sorted_patterns = self._sort_patterns_by_layer(matched_patterns)
        
        # Create nodes for each pattern
        placed_nodes = {}
        for match_data in sorted_patterns:
            if match_data['pattern_id'] is None:
                continue
                
            node = self._create_pattern_node(match_data)
            if node:
                placed_nodes[node.id] = {
                    'node': node,
                    'match_data': match_data
                }
                root.add_child(node)
        
        # Apply spatial constraints
        self._apply_spatial_constraints(placed_nodes, matched_patterns)
        
        # Apply causal reasoning
        if self.reasoning_engine:
            self._apply_causal_reasoning(root, placed_nodes)
        
        # Self-critique and adjust
        if self.self_model:
            self._self_critique_and_adjust(root)
        
        return root
    
    def _create_background_node(self, decomposition) -> Optional[SceneGraphNode]:
        """Create background node based on scene type"""
        bg_type = decomposition.background_type
        time_of_day = decomposition.time_of_day
        
        # Determine sky color based on time
        sky_pattern_name = 'sky'
        if time_of_day == 'sunset' or time_of_day == 'sunrise':
            sky_pattern_name = 'sunset_sky'
        elif time_of_day == 'night':
            sky_pattern_name = 'sky'  # Will override color
        
        # Create sky layer
        sky_pattern = self.pattern_db.search_by_name(sky_pattern_name)
        if sky_pattern:
            sky_node = SceneGraphNode(
                name="background_sky",
                pattern_id=sky_pattern.id,
                layer=self.layer_map['background'],
                transform=AffineTransform(
                    translate_x=0,
                    translate_y=0,
                    scale_x=self.canvas_width / 100,
                    scale_y=self.canvas_height * 0.6 / 100
                )
            )
            
            if time_of_day == 'night':
                sky_node.attributes['color_override'] = '#1E3A5F'
            
            return sky_node
        
        return None
    
    def _create_weather_node(self, weather: str) -> Optional[SceneGraphNode]:
        """Create weather effect overlay"""
        if weather == 'rainy':
            rain_pattern = self.pattern_db.search_by_name('rain')
            if rain_pattern:
                return SceneGraphNode(
                    name="weather_rain",
                    pattern_id=rain_pattern.id,
                    layer=self.layer_map['overlay'],
                    transform=AffineTransform(
                        translate_x=0,
                        translate_y=0,
                        scale_x=self.canvas_width / 100,
                        scale_y=self.canvas_height / 100
                    ),
                    attributes={'opacity': 0.6}
                )
        elif weather == 'cloudy':
            cloud_pattern = self.pattern_db.search_by_name('cloud')
            if cloud_pattern:
                # Create multiple clouds
                cloud_node = SceneGraphNode(
                    name="weather_clouds",
                    layer=self.layer_map['mid_background']
                )
                for i in range(3):
                    child = SceneGraphNode(
                        name=f"cloud_{i}",
                        pattern_id=cloud_pattern.id,
                        transform=AffineTransform(
                            translate_x=100 + i * 250,
                            translate_y=50 + (i % 2) * 30,
                            scale_x=1.5 + i * 0.3,
                            scale_y=1.5 + i * 0.3
                        )
                    )
                    cloud_node.add_child(child)
                return cloud_node
        
        return None
    
    def _sort_patterns_by_layer(self, matched_patterns: List[Dict]) -> List[Dict]:
        """Sort patterns by rendering layer"""
        layer_assignments = {
            'sky': 0, 'sunset_sky': 0,
            'mountain': 1,
            'water': 2, 'grass': 2, 'road': 2,
            'house': 3, 'tree': 3,
            'car': 4, 'person': 4, 'dog': 4, 'boat': 4,
            'sun': 1, 'cloud': 1,
            'rain': 5
        }
        
        def get_layer(match_data):
            concept = match_data.get('concept', {})
            entity = concept.get('entity', '')
            return layer_assignments.get(entity, 3)
        
        return sorted(matched_patterns, key=get_layer)
    
    def _create_pattern_node(self, match_data: Dict) -> Optional[SceneGraphNode]:
        """Create a scene node from matched pattern data"""
        pattern_id = match_data.get('pattern_id')
        if not pattern_id:
            return None
        
        pattern = self.pattern_db.get_pattern(pattern_id)
        if not pattern:
            return None
        
        concept = match_data.get('concept', {})
        entity = concept.get('entity', 'unknown')
        position_hint = match_data.get('position_hint')
        size_modifier = match_data.get('size_modifier', 1.0)
        color_override = match_data.get('color_override')
        importance = match_data.get('importance', 1.0)
        
        # Calculate position
        x, y = self._calculate_position(entity, position_hint)
        
        # Calculate scale based on pattern bounding box and importance
        base_scale = self._calculate_base_scale(pattern, entity)
        scale = base_scale * size_modifier * (0.8 + importance * 0.4)
        
        # Determine layer
        layer = self._get_layer_for_entity(entity)
        
        # Create node
        node = SceneGraphNode(
            name=entity,
            pattern_id=pattern_id,
            layer=layer,
            transform=AffineTransform(
                translate_x=x - (pattern.bounding_box['width'] * scale / 2),
                translate_y=y - (pattern.bounding_box['height'] * scale / 2),
                scale_x=scale,
                scale_y=scale
            ),
            attributes={
                'entity': entity,
                'color_override': color_override,
                'importance': importance
            }
        )
        
        return node
    
    def _calculate_position(self, entity: str, position_hint: Optional[str]) -> Tuple[float, float]:
        """Calculate pixel position for entity"""
        # Default position mapping
        entity_positions = {
            'sky': (0.5, 0.3),
            'sunset_sky': (0.5, 0.3),
            'sun': (0.8, 0.15),
            'cloud': (0.3, 0.2),
            'mountain': (0.5, 0.4),
            'water': (0.5, 0.75),
            'grass': (0.5, 0.85),
            'road': (0.5, 0.8),
            'house': (0.3, 0.55),
            'tree': (0.7, 0.55),
            'car': (0.5, 0.7),
            'person': (0.4, 0.65),
            'dog': (0.55, 0.7),
            'boat': (0.5, 0.6)
        }
        
        # Use position hint if available
        if position_hint and position_hint in self.position_presets:
            norm_x, norm_y = self.position_presets[position_hint]
        elif position_hint and position_hint.startswith('on_'):
            # Position relative to target
            target = position_hint[3:]
            target_pos = entity_positions.get(target, (0.5, 0.5))
            norm_x = target_pos[0]
            norm_y = target_pos[1] - 0.1  # Slightly above
        else:
            norm_x, norm_y = entity_positions.get(entity, (0.5, 0.5))
        
        # Add some randomness for natural look
        import random
        random.seed(hash(entity))
        norm_x += random.uniform(-0.05, 0.05)
        
        return norm_x * self.canvas_width, norm_y * self.canvas_height
    
    def _calculate_base_scale(self, pattern, entity: str) -> float:
        """Calculate appropriate scale for entity"""
        # Desired sizes relative to canvas
        size_map = {
            'sky': 8.0,
            'sunset_sky': 8.0,
            'sun': 1.0,
            'cloud': 1.5,
            'mountain': 3.0,
            'water': 4.0,
            'grass': 4.0,
            'road': 4.0,
            'house': 1.5,
            'tree': 1.2,
            'car': 1.0,
            'person': 0.8,
            'dog': 0.6,
            'boat': 1.2,
            'rain': 8.0
        }
        
        return size_map.get(entity, 1.0)
    
    def _get_layer_for_entity(self, entity: str) -> int:
        """Get rendering layer for entity type"""
        layer_map = {
            'sky': 0, 'sunset_sky': 0,
            'sun': 1, 'cloud': 1, 'mountain': 1,
            'water': 2, 'grass': 2, 'road': 2,
            'house': 3, 'tree': 3, 'boat': 3,
            'car': 4, 'person': 4, 'dog': 4,
            'rain': 5
        }
        return layer_map.get(entity, 3)
    
    def _apply_spatial_constraints(self, placed_nodes: Dict, matched_patterns: List[Dict]):
        """Apply spatial constraints between nodes"""
        for match_data in matched_patterns:
            concept = match_data.get('concept', {})
            relations = concept.get('relations', [])
            
            for relation_type, target in relations:
                # Find source and target nodes
                source_entity = concept.get('entity')
                source_node = None
                target_node = None
                
                for node_data in placed_nodes.values():
                    if node_data['match_data'].get('concept', {}).get('entity') == source_entity:
                        source_node = node_data['node']
                    if node_data['match_data'].get('concept', {}).get('entity') == target:
                        target_node = node_data['node']
                
                if source_node and target_node:
                    # Apply constraint
                    self._apply_relation_constraint(source_node, target_node, relation_type)
    
    def _apply_relation_constraint(self, source: SceneGraphNode, target: SceneGraphNode, relation: str):
        """Apply a spatial relation constraint"""
        if relation == 'on':
            # Source should be on top of target
            target_y = target.transform.translate_y
            source.transform.translate_y = target_y - 50  # Above
            source.transform.translate_x = target.transform.translate_x
        elif relation == 'beside':
            # Source beside target
            source.transform.translate_x = target.transform.translate_x + 100
            source.transform.translate_y = target.transform.translate_y
        elif relation == 'in':
            # Source inside target
            source.transform.translate_x = target.transform.translate_x
            source.transform.translate_y = target.transform.translate_y
            source.layer = target.layer + 1
    
    def _apply_causal_reasoning(self, root: SceneGraphNode, placed_nodes: Dict):
        """Apply causal reasoning for physical plausibility"""
        # Ensure ground-touching objects are properly placed
        ground_entities = {'car', 'person', 'dog', 'house', 'tree'}
        
        for node_data in placed_nodes.values():
            node = node_data['node']
            entity = node.attributes.get('entity', '')
            
            if entity in ground_entities:
                # Make sure they're not floating
                ground_level = self.canvas_height * 0.75
                pattern = self.pattern_db.get_pattern(node.pattern_id)
                if pattern:
                    obj_bottom = node.transform.translate_y + pattern.bounding_box['height'] * node.transform.scale_y
                    if obj_bottom < ground_level:
                        # Adjust to touch ground
                        node.transform.translate_y = ground_level - pattern.bounding_box['height'] * node.transform.scale_y
    
    def _self_critique_and_adjust(self, root: SceneGraphNode):
        """Use self-modeling to critique and adjust composition"""
        # Check for overlapping elements
        # Check for aesthetic balance
        # This would integrate with SelfModelingEngine
        pass
    
    def get_all_nodes(self, root: SceneGraphNode) -> List[SceneGraphNode]:
        """Flatten scene graph into list of nodes"""
        nodes = [root]
        for child in root.children:
            nodes.extend(self.get_all_nodes(child))
        return nodes
