"""
Visual Patterns - Core data structures for pattern-based visual vocabulary (PBVV)
Pure pattern matching without neural networks
"""

import uuid
import math
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class ShapeType(Enum):
    """Primitive shape types"""
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    LINE = "line"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    BEZIER_CURVE = "bezier_curve"
    ARC = "arc"
    PATH = "path"


@dataclass
class ShapePrimitive:
    """
    Atomic visual building block
    Can be: circle, line, bezier_curve, polygon, etc.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ShapeType = ShapeType.RECTANGLE
    parameters: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default attributes if not provided
        if 'fill' not in self.attributes:
            self.attributes['fill'] = '#000000'
        if 'stroke' not in self.attributes:
            self.attributes['stroke'] = 'none'
        if 'stroke_width' not in self.attributes:
            self.attributes['stroke_width'] = 1
        if 'opacity' not in self.attributes:
            self.attributes['opacity'] = 1.0
    
    def to_svg_element(self, x_offset: float = 0, y_offset: float = 0, scale: float = 1.0) -> str:
        """Convert primitive to SVG element string"""
        attrs = self._get_common_attrs()
        
        if self.type == ShapeType.CIRCLE:
            cx = (self.parameters.get('cx', 50) + x_offset) * scale
            cy = (self.parameters.get('cy', 50) + y_offset) * scale
            r = self.parameters.get('r', 25) * scale
            return f'<circle cx="{cx}" cy="{cy}" r="{r}" {attrs}/>'
            
        elif self.type == ShapeType.ELLIPSE:
            cx = (self.parameters.get('cx', 50) + x_offset) * scale
            cy = (self.parameters.get('cy', 50) + y_offset) * scale
            rx = self.parameters.get('rx', 30) * scale
            ry = self.parameters.get('ry', 20) * scale
            return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {attrs}/>'
            
        elif self.type == ShapeType.RECTANGLE:
            x = (self.parameters.get('x', 0) + x_offset) * scale
            y = (self.parameters.get('y', 0) + y_offset) * scale
            width = self.parameters.get('width', 100) * scale
            height = self.parameters.get('height', 50) * scale
            rx = self.parameters.get('rx', 0) * scale
            return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" {attrs}/>'
            
        elif self.type == ShapeType.LINE:
            x1 = (self.parameters.get('x1', 0) + x_offset) * scale
            y1 = (self.parameters.get('y1', 0) + y_offset) * scale
            x2 = (self.parameters.get('x2', 100) + x_offset) * scale
            y2 = (self.parameters.get('y2', 100) + y_offset) * scale
            return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {attrs}/>'
            
        elif self.type == ShapeType.POLYGON:
            points = self.parameters.get('points', [(0, 0), (50, 100), (100, 0)])
            scaled_points = ' '.join([f"{(p[0]+x_offset)*scale},{(p[1]+y_offset)*scale}" for p in points])
            return f'<polygon points="{scaled_points}" {attrs}/>'
            
        elif self.type == ShapeType.POLYLINE:
            points = self.parameters.get('points', [(0, 0), (50, 50), (100, 0)])
            scaled_points = ' '.join([f"{(p[0]+x_offset)*scale},{(p[1]+y_offset)*scale}" for p in points])
            return f'<polyline points="{scaled_points}" {attrs}/>'
            
        elif self.type == ShapeType.PATH:
            d = self.parameters.get('d', 'M 0 0 L 100 100')
            return f'<path d="{d}" {attrs}/>'
            
        elif self.type == ShapeType.BEZIER_CURVE:
            points = self.parameters.get('points', [(0, 0), (25, 50), (75, 50), (100, 0)])
            if len(points) >= 4:
                p0, p1, p2, p3 = [(p[0]+x_offset, p[1]+y_offset) for p in points[:4]]
                d = f"M {p0[0]*scale} {p0[1]*scale} C {p1[0]*scale} {p1[1]*scale}, {p2[0]*scale} {p2[1]*scale}, {p3[0]*scale} {p3[1]*scale}"
                return f'<path d="{d}" {attrs}/>'
            return ''
            
        return ''
    
    def _get_common_attrs(self) -> str:
        """Get common SVG attributes as string"""
        attrs = []
        attrs.append(f'fill="{self.attributes.get("fill", "none")}"')
        attrs.append(f'stroke="{self.attributes.get("stroke", "black")}"')
        attrs.append(f'stroke-width="{self.attributes.get("stroke_width", 1)}"')
        if self.attributes.get('opacity', 1.0) < 1.0:
            attrs.append(f'opacity="{self.attributes.get("opacity", 1.0)}"')
        return ' '.join(attrs)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'parameters': self.parameters,
            'attributes': self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ShapePrimitive':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=ShapeType(data.get('type', 'rectangle')),
            parameters=data.get('parameters', {}),
            attributes=data.get('attributes', {})
        )


@dataclass
class ColorDistribution:
    """Color palette with weights"""
    primary: str = "#3B82F6"  # Main color
    secondary: str = "#10B981"  # Accent color
    background: str = "#FFFFFF"
    accent: str = "#F59E0B"
    weights: Dict[str, float] = field(default_factory=lambda: {
        'primary': 0.5,
        'secondary': 0.3,
        'background': 0.15,
        'accent': 0.05
    })
    
    def get_color_by_weight(self, seed: float = 0.5) -> str:
        """Get color based on weight distribution"""
        cumulative = 0
        for color_name, weight in self.weights.items():
            cumulative += weight
            if seed <= cumulative:
                return getattr(self, color_name)
        return self.primary
    
    def to_dict(self) -> Dict:
        return {
            'primary': self.primary,
            'secondary': self.secondary,
            'background': self.background,
            'accent': self.accent,
            'weights': self.weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColorDistribution':
        return cls(**data)


@dataclass
class TextureSignature:
    """Texture pattern template"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "solid"
    pattern_type: str = "solid"  # solid, striped, dotted, gradient, noise
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_svg_defs(self, pattern_id: str) -> str:
        """Generate SVG pattern definition"""
        if self.pattern_type == "striped":
            spacing = self.parameters.get('spacing', 10)
            color = self.parameters.get('color', '#000000')
            angle = self.parameters.get('angle', 45)
            return f'''
            <pattern id="{pattern_id}" patternUnits="userSpaceOnUse" width="{spacing}" height="{spacing}" patternTransform="rotate({angle})">
                <line x1="0" y1="0" x2="0" y2="{spacing}" stroke="{color}" stroke-width="2"/>
            </pattern>
            '''
        elif self.pattern_type == "dotted":
            spacing = self.parameters.get('spacing', 10)
            radius = self.parameters.get('radius', 2)
            color = self.parameters.get('color', '#000000')
            return f'''
            <pattern id="{pattern_id}" patternUnits="userSpaceOnUse" width="{spacing}" height="{spacing}">
                <circle cx="{spacing/2}" cy="{spacing/2}" r="{radius}" fill="{color}"/>
            </pattern>
            '''
        elif self.pattern_type == "gradient":
            color1 = self.parameters.get('color1', '#000000')
            color2 = self.parameters.get('color2', '#FFFFFF')
            direction = self.parameters.get('direction', 'vertical')
            if direction == 'vertical':
                return f'''
                <linearGradient id="{pattern_id}" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:{color1}"/>
                    <stop offset="100%" style="stop-color:{color2}"/>
                </linearGradient>
                '''
            else:
                return f'''
                <linearGradient id="{pattern_id}" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:{color1}"/>
                    <stop offset="100%" style="stop-color:{color2}"/>
                </linearGradient>
                '''
        return ''
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'pattern_type': self.pattern_type,
            'parameters': self.parameters
        }


@dataclass
class SpatialRelation:
    """Relationship between visual components"""
    source: str  # Source component ID
    target: str  # Target component ID
    relation_type: str  # above, below, left_of, right_of, inside, overlaps, touches
    strength: float = 1.0  # 0-1, how strict the relation is
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'strength': self.strength
        }


@dataclass
class RelationGraph:
    """Graph of spatial relations between components"""
    relations: List[SpatialRelation] = field(default_factory=list)
    
    def add_relation(self, source: str, target: str, relation_type: str, strength: float = 1.0):
        self.relations.append(SpatialRelation(source, target, relation_type, strength))
    
    def get_relations_for(self, component_id: str) -> List[SpatialRelation]:
        return [r for r in self.relations if r.source == component_id or r.target == component_id]
    
    def to_dict(self) -> Dict:
        return {'relations': [r.to_dict() for r in self.relations]}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RelationGraph':
        graph = cls()
        for r in data.get('relations', []):
            graph.relations.append(SpatialRelation(**r))
        return graph


@dataclass
class VisualPattern:
    """
    Complete visual pattern with all components
    Abstraction levels: 1=atomic (line, circle), 5=complex (house, car)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "unnamed"
    shape_descriptors: List[ShapePrimitive] = field(default_factory=list)
    color_palette: ColorDistribution = field(default_factory=ColorDistribution)
    texture_templates: List[TextureSignature] = field(default_factory=list)
    spatial_relations: RelationGraph = field(default_factory=RelationGraph)
    abstraction_level: int = 1  # 1-5
    semantic_tags: Set[str] = field(default_factory=set)
    usage_contexts: List[str] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=dict)  # Named slots for composition
    bounding_box: Dict[str, float] = field(default_factory=lambda: {'width': 100, 'height': 100})
    
    def to_svg(self, width: int = 200, height: int = 200) -> str:
        """Render pattern to SVG string"""
        svg_parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {self.bounding_box["width"]} {self.bounding_box["height"]}">']
        
        # Add defs for textures
        if self.texture_templates:
            svg_parts.append('<defs>')
            for i, texture in enumerate(self.texture_templates):
                svg_parts.append(texture.to_svg_defs(f'texture_{i}'))
            svg_parts.append('</defs>')
        
        # Add shapes
        for shape in self.shape_descriptors:
            svg_parts.append(shape.to_svg_element())
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'shape_descriptors': [s.to_dict() for s in self.shape_descriptors],
            'color_palette': self.color_palette.to_dict(),
            'texture_templates': [t.to_dict() for t in self.texture_templates],
            'spatial_relations': self.spatial_relations.to_dict(),
            'abstraction_level': self.abstraction_level,
            'semantic_tags': list(self.semantic_tags),
            'usage_contexts': self.usage_contexts,
            'slots': self.slots,
            'bounding_box': self.bounding_box
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VisualPattern':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', 'unnamed'),
            shape_descriptors=[ShapePrimitive.from_dict(s) for s in data.get('shape_descriptors', [])],
            color_palette=ColorDistribution.from_dict(data.get('color_palette', {})),
            texture_templates=[TextureSignature(**t) for t in data.get('texture_templates', [])],
            spatial_relations=RelationGraph.from_dict(data.get('spatial_relations', {})),
            abstraction_level=data.get('abstraction_level', 1),
            semantic_tags=set(data.get('semantic_tags', [])),
            usage_contexts=data.get('usage_contexts', []),
            slots=data.get('slots', {}),
            bounding_box=data.get('bounding_box', {'width': 100, 'height': 100})
        )


class VisualPatternDatabase:
    """
    In-memory visual pattern database with semantic search
    """
    
    def __init__(self):
        self.patterns: Dict[str, VisualPattern] = {}
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> pattern_ids
        self.level_index: Dict[int, Set[str]] = {}  # abstraction_level -> pattern_ids
        self._initialized = False
    
    def initialize(self):
        """Initialize with base patterns"""
        if self._initialized:
            return
        
        base_patterns = self._get_base_patterns()
        for pattern in base_patterns:
            self.add_pattern(pattern)
        
        self._initialized = True
    
    def _get_base_patterns(self) -> List[VisualPattern]:
        """Generate base visual patterns"""
        patterns = []
        
        # Level 1: Atomic shapes
        # Circle
        circle = VisualPattern(
            name="circle",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 50, 'cy': 50, 'r': 40},
                    attributes={'fill': '#3B82F6', 'stroke': '#1E40AF', 'stroke_width': 2}
                )
            ],
            abstraction_level=1,
            semantic_tags={'circle', 'round', 'ball', 'wheel', 'dot', 'sun', 'moon'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(circle)
        
        # Rectangle
        rectangle = VisualPattern(
            name="rectangle",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 10, 'y': 20, 'width': 80, 'height': 60},
                    attributes={'fill': '#10B981', 'stroke': '#047857', 'stroke_width': 2}
                )
            ],
            abstraction_level=1,
            semantic_tags={'rectangle', 'box', 'block', 'square', 'building', 'window'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(rectangle)
        
        # Triangle
        triangle = VisualPattern(
            name="triangle",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 10), (90, 90), (10, 90)]},
                    attributes={'fill': '#F59E0B', 'stroke': '#D97706', 'stroke_width': 2}
                )
            ],
            abstraction_level=1,
            semantic_tags={'triangle', 'arrow', 'roof', 'mountain', 'peak'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(triangle)
        
        # Line
        line = VisualPattern(
            name="line",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 10, 'y1': 50, 'x2': 90, 'y2': 50},
                    attributes={'fill': 'none', 'stroke': '#6B7280', 'stroke_width': 3}
                )
            ],
            abstraction_level=1,
            semantic_tags={'line', 'edge', 'border', 'horizon', 'road', 'path'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(line)
        
        # Level 2: Simple composites
        # Tree
        tree = VisualPattern(
            name="tree",
            shape_descriptors=[
                # Trunk
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 40, 'y': 60, 'width': 20, 'height': 40},
                    attributes={'fill': '#8B4513', 'stroke': '#5D3A1A', 'stroke_width': 1}
                ),
                # Foliage
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 5), (85, 65), (15, 65)]},
                    attributes={'fill': '#22C55E', 'stroke': '#15803D', 'stroke_width': 2}
                )
            ],
            abstraction_level=2,
            semantic_tags={'tree', 'plant', 'nature', 'forest', 'pine', 'evergreen'},
            slots={'trunk': 'rectangle', 'leaves': 'triangle'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(tree)
        
        # House
        house = VisualPattern(
            name="house",
            shape_descriptors=[
                # Body
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 15, 'y': 45, 'width': 70, 'height': 50},
                    attributes={'fill': '#F3F4F6', 'stroke': '#6B7280', 'stroke_width': 2}
                ),
                # Roof
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 10), (95, 50), (5, 50)]},
                    attributes={'fill': '#EF4444', 'stroke': '#B91C1C', 'stroke_width': 2}
                ),
                # Door
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 40, 'y': 65, 'width': 20, 'height': 30},
                    attributes={'fill': '#854D0E', 'stroke': '#713F12', 'stroke_width': 1}
                ),
                # Window
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 25, 'y': 55, 'width': 12, 'height': 12},
                    attributes={'fill': '#BFDBFE', 'stroke': '#1E40AF', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 63, 'y': 55, 'width': 12, 'height': 12},
                    attributes={'fill': '#BFDBFE', 'stroke': '#1E40AF', 'stroke_width': 1}
                )
            ],
            abstraction_level=3,
            semantic_tags={'house', 'home', 'building', 'structure', 'cottage', 'dwelling'},
            slots={'roof': 'triangle', 'walls': 'rectangle', 'door': 'rectangle', 'windows': 'rectangle'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(house)
        
        # Car
        car = VisualPattern(
            name="car",
            shape_descriptors=[
                # Body
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 10, 'y': 40, 'width': 80, 'height': 30, 'rx': 5},
                    attributes={'fill': '#DC2626', 'stroke': '#991B1B', 'stroke_width': 2}
                ),
                # Cabin
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(25, 40), (35, 20), (65, 20), (75, 40)]},
                    attributes={'fill': '#DC2626', 'stroke': '#991B1B', 'stroke_width': 2}
                ),
                # Windows
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(30, 38), (38, 23), (48, 23), (48, 38)]},
                    attributes={'fill': '#BFDBFE', 'stroke': '#1E40AF', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(52, 38), (52, 23), (62, 23), (70, 38)]},
                    attributes={'fill': '#BFDBFE', 'stroke': '#1E40AF', 'stroke_width': 1}
                ),
                # Wheels
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 28, 'cy': 70, 'r': 12},
                    attributes={'fill': '#1F2937', 'stroke': '#111827', 'stroke_width': 2}
                ),
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 72, 'cy': 70, 'r': 12},
                    attributes={'fill': '#1F2937', 'stroke': '#111827', 'stroke_width': 2}
                ),
                # Wheel centers
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 28, 'cy': 70, 'r': 5},
                    attributes={'fill': '#9CA3AF', 'stroke': 'none'}
                ),
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 72, 'cy': 70, 'r': 5},
                    attributes={'fill': '#9CA3AF', 'stroke': 'none'}
                )
            ],
            abstraction_level=3,
            semantic_tags={'car', 'vehicle', 'automobile', 'transport', 'sedan', 'auto'},
            slots={'body': 'rectangle', 'wheels': 'circle', 'windows': 'polygon'},
            bounding_box={'width': 100, 'height': 85}
        )
        patterns.append(car)
        
        # Sun
        sun = VisualPattern(
            name="sun",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 50, 'cy': 50, 'r': 25},
                    attributes={'fill': '#FBBF24', 'stroke': '#F59E0B', 'stroke_width': 2}
                ),
                # Rays
                *[ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={
                        'x1': 50 + 30 * math.cos(math.radians(i * 45)),
                        'y1': 50 + 30 * math.sin(math.radians(i * 45)),
                        'x2': 50 + 45 * math.cos(math.radians(i * 45)),
                        'y2': 50 + 45 * math.sin(math.radians(i * 45))
                    },
                    attributes={'fill': 'none', 'stroke': '#F59E0B', 'stroke_width': 3}
                ) for i in range(8)]
            ],
            abstraction_level=2,
            semantic_tags={'sun', 'sunny', 'daylight', 'bright', 'warm', 'summer'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(sun)
        
        # Cloud
        cloud = VisualPattern(
            name="cloud",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 35, 'cy': 55, 'r': 20},
                    attributes={'fill': '#E5E7EB', 'stroke': '#D1D5DB', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 50, 'cy': 45, 'r': 25},
                    attributes={'fill': '#E5E7EB', 'stroke': '#D1D5DB', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 70, 'cy': 50, 'r': 22},
                    attributes={'fill': '#E5E7EB', 'stroke': '#D1D5DB', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 25, 'y': 50, 'width': 55, 'height': 25},
                    attributes={'fill': '#E5E7EB', 'stroke': 'none'}
                )
            ],
            abstraction_level=2,
            semantic_tags={'cloud', 'sky', 'weather', 'fluffy', 'cloudy'},
            bounding_box={'width': 100, 'height': 80}
        )
        patterns.append(cloud)
        
        # Mountain
        mountain = VisualPattern(
            name="mountain",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 10), (100, 90), (0, 90)]},
                    attributes={'fill': '#6B7280', 'stroke': '#4B5563', 'stroke_width': 2}
                ),
                # Snow cap
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 10), (65, 35), (35, 35)]},
                    attributes={'fill': '#FFFFFF', 'stroke': '#E5E7EB', 'stroke_width': 1}
                )
            ],
            abstraction_level=2,
            semantic_tags={'mountain', 'peak', 'hill', 'landscape', 'nature', 'terrain'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(mountain)
        
        # Water/Ocean
        water = VisualPattern(
            name="water",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 0, 'width': 100, 'height': 100},
                    attributes={'fill': '#3B82F6', 'stroke': 'none'}
                ),
                # Waves
                *[ShapePrimitive(
                    type=ShapeType.PATH,
                    parameters={'d': f'M 0 {20 + i*25} Q 25 {10 + i*25} 50 {20 + i*25} T 100 {20 + i*25}'},
                    attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2, 'opacity': 0.5}
                ) for i in range(3)]
            ],
            abstraction_level=2,
            semantic_tags={'water', 'ocean', 'sea', 'lake', 'river', 'wave', 'blue'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(water)
        
        # Road
        road = VisualPattern(
            name="road",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 30, 'width': 100, 'height': 40},
                    attributes={'fill': '#374151', 'stroke': 'none'}
                ),
                # Lane markings
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 10, 'y1': 50, 'x2': 30, 'y2': 50},
                    attributes={'fill': 'none', 'stroke': '#FBBF24', 'stroke_width': 3}
                ),
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 40, 'y1': 50, 'x2': 60, 'y2': 50},
                    attributes={'fill': 'none', 'stroke': '#FBBF24', 'stroke_width': 3}
                ),
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 70, 'y1': 50, 'x2': 90, 'y2': 50},
                    attributes={'fill': 'none', 'stroke': '#FBBF24', 'stroke_width': 3}
                )
            ],
            abstraction_level=2,
            semantic_tags={'road', 'street', 'highway', 'path', 'asphalt', 'pavement'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(road)
        
        # Person (stick figure style)
        person = VisualPattern(
            name="person",
            shape_descriptors=[
                # Head
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 50, 'cy': 20, 'r': 12},
                    attributes={'fill': '#FCD34D', 'stroke': '#F59E0B', 'stroke_width': 2}
                ),
                # Body
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 50, 'y1': 32, 'x2': 50, 'y2': 60},
                    attributes={'fill': 'none', 'stroke': '#1F2937', 'stroke_width': 4}
                ),
                # Arms
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 30, 'y1': 45, 'x2': 70, 'y2': 45},
                    attributes={'fill': 'none', 'stroke': '#1F2937', 'stroke_width': 4}
                ),
                # Legs
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 50, 'y1': 60, 'x2': 35, 'y2': 90},
                    attributes={'fill': 'none', 'stroke': '#1F2937', 'stroke_width': 4}
                ),
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 50, 'y1': 60, 'x2': 65, 'y2': 90},
                    attributes={'fill': 'none', 'stroke': '#1F2937', 'stroke_width': 4}
                )
            ],
            abstraction_level=3,
            semantic_tags={'person', 'human', 'man', 'woman', 'people', 'figure', 'walking'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(person)
        
        # Dog
        dog = VisualPattern(
            name="dog",
            shape_descriptors=[
                # Body
                ShapePrimitive(
                    type=ShapeType.ELLIPSE,
                    parameters={'cx': 50, 'cy': 55, 'rx': 30, 'ry': 18},
                    attributes={'fill': '#D97706', 'stroke': '#B45309', 'stroke_width': 2}
                ),
                # Head
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 25, 'cy': 40, 'r': 15},
                    attributes={'fill': '#D97706', 'stroke': '#B45309', 'stroke_width': 2}
                ),
                # Ear
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(15, 35), (10, 20), (25, 30)]},
                    attributes={'fill': '#92400E', 'stroke': '#78350F', 'stroke_width': 1}
                ),
                # Eye
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 20, 'cy': 38, 'r': 3},
                    attributes={'fill': '#1F2937', 'stroke': 'none'}
                ),
                # Nose
                ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': 12, 'cy': 45, 'r': 3},
                    attributes={'fill': '#1F2937', 'stroke': 'none'}
                ),
                # Legs
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 30, 'y': 65, 'width': 8, 'height': 20},
                    attributes={'fill': '#D97706', 'stroke': '#B45309', 'stroke_width': 1}
                ),
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 60, 'y': 65, 'width': 8, 'height': 20},
                    attributes={'fill': '#D97706', 'stroke': '#B45309', 'stroke_width': 1}
                ),
                # Tail
                ShapePrimitive(
                    type=ShapeType.PATH,
                    parameters={'d': 'M 80 55 Q 95 45 90 30'},
                    attributes={'fill': 'none', 'stroke': '#D97706', 'stroke_width': 6}
                )
            ],
            abstraction_level=3,
            semantic_tags={'dog', 'puppy', 'pet', 'animal', 'canine', 'hound'},
            bounding_box={'width': 100, 'height': 90}
        )
        patterns.append(dog)
        
        # Boat/Sailboat
        boat = VisualPattern(
            name="sailboat",
            shape_descriptors=[
                # Hull
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(10, 70), (30, 90), (70, 90), (90, 70)]},
                    attributes={'fill': '#8B4513', 'stroke': '#5D3A1A', 'stroke_width': 2}
                ),
                # Mast
                ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': 50, 'y1': 70, 'x2': 50, 'y2': 15},
                    attributes={'fill': 'none', 'stroke': '#5D3A1A', 'stroke_width': 3}
                ),
                # Sail
                ShapePrimitive(
                    type=ShapeType.POLYGON,
                    parameters={'points': [(50, 15), (50, 65), (85, 50)]},
                    attributes={'fill': '#FFFFFF', 'stroke': '#D1D5DB', 'stroke_width': 2}
                )
            ],
            abstraction_level=3,
            semantic_tags={'boat', 'sailboat', 'ship', 'vessel', 'sailing', 'yacht'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(boat)
        
        # Grass
        grass = VisualPattern(
            name="grass",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 50, 'width': 100, 'height': 50},
                    attributes={'fill': '#22C55E', 'stroke': 'none'}
                ),
                # Grass blades
                *[ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={'x1': i * 10, 'y1': 50, 'x2': i * 10 + 3, 'y2': 40},
                    attributes={'fill': 'none', 'stroke': '#15803D', 'stroke_width': 2}
                ) for i in range(10)]
            ],
            abstraction_level=2,
            semantic_tags={'grass', 'lawn', 'ground', 'green', 'field', 'meadow', 'park'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(grass)
        
        # Sky
        sky = VisualPattern(
            name="sky",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 0, 'width': 100, 'height': 100},
                    attributes={'fill': '#93C5FD', 'stroke': 'none'}
                )
            ],
            abstraction_level=1,
            semantic_tags={'sky', 'blue', 'day', 'background', 'atmosphere'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(sky)
        
        # Sunset sky
        sunset_sky = VisualPattern(
            name="sunset_sky",
            shape_descriptors=[
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 0, 'width': 100, 'height': 30},
                    attributes={'fill': '#FCD34D', 'stroke': 'none'}
                ),
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 30, 'width': 100, 'height': 30},
                    attributes={'fill': '#F97316', 'stroke': 'none'}
                ),
                ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': 0, 'y': 60, 'width': 100, 'height': 40},
                    attributes={'fill': '#DC2626', 'stroke': 'none'}
                )
            ],
            abstraction_level=2,
            semantic_tags={'sunset', 'sky', 'evening', 'dusk', 'orange', 'red', 'golden'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(sunset_sky)
        
        # Rain
        rain = VisualPattern(
            name="rain",
            shape_descriptors=[
                *[ShapePrimitive(
                    type=ShapeType.LINE,
                    parameters={
                        'x1': 10 + i * 12 + (j % 2) * 6,
                        'y1': 10 + j * 20,
                        'x2': 10 + i * 12 + (j % 2) * 6 + 5,
                        'y2': 25 + j * 20
                    },
                    attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2, 'opacity': 0.7}
                ) for i in range(8) for j in range(4)]
            ],
            abstraction_level=2,
            semantic_tags={'rain', 'rainy', 'weather', 'wet', 'storm', 'drops'},
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(rain)
        
        return patterns
    
    def add_pattern(self, pattern: VisualPattern):
        """Add pattern to database and update indices"""
        self.patterns[pattern.id] = pattern
        
        # Update tag index
        for tag in pattern.semantic_tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(pattern.id)
        
        # Update level index
        if pattern.abstraction_level not in self.level_index:
            self.level_index[pattern.abstraction_level] = set()
        self.level_index[pattern.abstraction_level].add(pattern.id)
    
    def get_pattern(self, pattern_id: str) -> Optional[VisualPattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[VisualPattern]:
        """Search patterns by semantic tags"""
        scores: Dict[str, int] = {}
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in self.tag_index:
                for pattern_id in self.tag_index[tag_lower]:
                    scores[pattern_id] = scores.get(pattern_id, 0) + 1
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [self.patterns[pid] for pid in sorted_ids[:limit]]
    
    def search_by_name(self, name: str) -> Optional[VisualPattern]:
        """Search pattern by name"""
        name_lower = name.lower()
        for pattern in self.patterns.values():
            if pattern.name.lower() == name_lower:
                return pattern
        return None
    
    def get_by_abstraction_level(self, level: int) -> List[VisualPattern]:
        """Get patterns by abstraction level"""
        pattern_ids = self.level_index.get(level, set())
        return [self.patterns[pid] for pid in pattern_ids]
    
    def get_all_patterns(self) -> List[VisualPattern]:
        """Get all patterns"""
        return list(self.patterns.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_patterns': len(self.patterns),
            'by_level': {level: len(ids) for level, ids in self.level_index.items()},
            'total_tags': len(self.tag_index),
            'top_tags': sorted(
                [(tag, len(ids)) for tag, ids in self.tag_index.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
