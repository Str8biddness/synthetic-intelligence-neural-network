"""
Multi-Layer Scene Composition - Depth-based rendering with atmospheric effects
Separates scenes into layers with proper depth ordering and atmospheric effects
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DepthLayer(Enum):
    """Depth layers for scene composition"""
    FAR_BACKGROUND = 0    # Sky, distant mountains
    BACKGROUND = 1        # Far scenery
    MID_BACKGROUND = 2    # Medium distance objects
    MIDGROUND = 3         # Main scene elements
    MID_FOREGROUND = 4    # Closer objects
    FOREGROUND = 5        # Closest elements
    OVERLAY = 6           # UI elements, effects


@dataclass
class AtmosphericEffect:
    """Atmospheric effect configuration"""
    fog_density: float = 0.0        # 0-1, increases with distance
    fog_color: str = "#B0C4DE"      # Fog/haze color
    blur_amount: float = 0.0        # Gaussian blur for depth
    brightness_shift: float = 0.0   # Lighten distant objects
    saturation_shift: float = 0.0   # Desaturate distant objects
    scale_factor: float = 1.0       # Perspective scaling
    
    def to_dict(self) -> Dict:
        return {
            'fog_density': self.fog_density,
            'fog_color': self.fog_color,
            'blur_amount': self.blur_amount,
            'brightness_shift': self.brightness_shift,
            'saturation_shift': self.saturation_shift,
            'scale_factor': self.scale_factor
        }


@dataclass
class LayerElement:
    """An element in a depth layer"""
    id: str
    pattern_id: str
    pattern_name: str
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    rotation: float = 0.0
    opacity: float = 1.0
    z_index: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'rotation': self.rotation,
            'opacity': self.opacity,
            'z_index': self.z_index
        }


@dataclass
class SceneLayer:
    """A single depth layer in the scene"""
    depth: DepthLayer
    elements: List[LayerElement] = field(default_factory=list)
    atmospheric_effect: AtmosphericEffect = field(default_factory=AtmosphericEffect)
    parallax_factor: float = 1.0  # For animation parallax
    
    def to_dict(self) -> Dict:
        return {
            'depth': self.depth.value,
            'depth_name': self.depth.name,
            'elements': [e.to_dict() for e in self.elements],
            'atmospheric_effect': self.atmospheric_effect.to_dict(),
            'parallax_factor': self.parallax_factor
        }


@dataclass
class MultiLayerScene:
    """Complete multi-layer scene composition"""
    id: str = ""
    name: str = ""
    width: int = 800
    height: int = 600
    layers: Dict[DepthLayer, SceneLayer] = field(default_factory=dict)
    global_lighting: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'layers': {k.name: v.to_dict() for k, v in self.layers.items()},
            'global_lighting': self.global_lighting,
            'layer_count': len(self.layers),
            'total_elements': sum(len(layer.elements) for layer in self.layers.values())
        }


class MultiLayerComposer:
    """
    Composes scenes with multiple depth layers and atmospheric effects
    Handles depth-based rendering, fog, blur, and lighting
    """
    
    # Default atmospheric effects per layer
    DEFAULT_ATMOSPHERICS = {
        DepthLayer.FAR_BACKGROUND: AtmosphericEffect(
            fog_density=0.4, blur_amount=2.0, brightness_shift=0.3,
            saturation_shift=-0.3, scale_factor=0.3
        ),
        DepthLayer.BACKGROUND: AtmosphericEffect(
            fog_density=0.25, blur_amount=1.5, brightness_shift=0.2,
            saturation_shift=-0.2, scale_factor=0.5
        ),
        DepthLayer.MID_BACKGROUND: AtmosphericEffect(
            fog_density=0.15, blur_amount=1.0, brightness_shift=0.1,
            saturation_shift=-0.1, scale_factor=0.7
        ),
        DepthLayer.MIDGROUND: AtmosphericEffect(
            fog_density=0.05, blur_amount=0.0, brightness_shift=0.0,
            saturation_shift=0.0, scale_factor=1.0
        ),
        DepthLayer.MID_FOREGROUND: AtmosphericEffect(
            fog_density=0.0, blur_amount=0.0, brightness_shift=-0.05,
            saturation_shift=0.05, scale_factor=1.2
        ),
        DepthLayer.FOREGROUND: AtmosphericEffect(
            fog_density=0.0, blur_amount=0.5, brightness_shift=-0.1,
            saturation_shift=0.1, scale_factor=1.5
        ),
        DepthLayer.OVERLAY: AtmosphericEffect(
            fog_density=0.0, blur_amount=0.0, brightness_shift=0.0,
            saturation_shift=0.0, scale_factor=1.0
        )
    }
    
    # Parallax factors for animation
    PARALLAX_FACTORS = {
        DepthLayer.FAR_BACKGROUND: 0.1,
        DepthLayer.BACKGROUND: 0.3,
        DepthLayer.MID_BACKGROUND: 0.5,
        DepthLayer.MIDGROUND: 1.0,
        DepthLayer.MID_FOREGROUND: 1.3,
        DepthLayer.FOREGROUND: 1.6,
        DepthLayer.OVERLAY: 0.0
    }
    
    def __init__(self, pattern_db):
        self.pattern_db = pattern_db
    
    def create_scene(self, name: str, width: int = 800, height: int = 600) -> MultiLayerScene:
        """Create a new multi-layer scene"""
        import uuid
        scene = MultiLayerScene(
            id=str(uuid.uuid4())[:8],
            name=name,
            width=width,
            height=height,
            global_lighting={'ambient': 1.0, 'direction': 45, 'intensity': 0.8}
        )
        
        # Initialize all layers
        for depth in DepthLayer:
            scene.layers[depth] = SceneLayer(
                depth=depth,
                atmospheric_effect=self.DEFAULT_ATMOSPHERICS[depth],
                parallax_factor=self.PARALLAX_FACTORS[depth]
            )
        
        return scene
    
    def assign_to_layer(self, scene: MultiLayerScene, pattern_id: str, 
                        pattern_name: str, layer: DepthLayer,
                        x: float, y: float, width: float = 100, 
                        height: float = 100) -> LayerElement:
        """Assign a pattern to a specific depth layer"""
        import uuid
        
        # Apply atmospheric scaling
        atm = scene.layers[layer].atmospheric_effect
        scaled_width = width * atm.scale_factor
        scaled_height = height * atm.scale_factor
        
        # Calculate y position based on depth (higher = further back)
        depth_y_offset = (DepthLayer.FOREGROUND.value - layer.value) * 20
        
        element = LayerElement(
            id=str(uuid.uuid4())[:8],
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            x=x,
            y=y + depth_y_offset,
            width=scaled_width,
            height=scaled_height,
            opacity=1.0 - atm.fog_density * 0.5,
            z_index=layer.value * 100 + len(scene.layers[layer].elements)
        )
        
        scene.layers[layer].elements.append(element)
        return element
    
    def auto_assign_layer(self, pattern_name: str, semantic_tags: set) -> DepthLayer:
        """Automatically determine the best layer for a pattern based on semantics"""
        # Sky and celestial -> far background
        if semantic_tags & {'sky', 'sun', 'moon', 'stars', 'clouds', 'aurora', 'celestial'}:
            return DepthLayer.FAR_BACKGROUND
        
        # Distant landscape -> background
        if semantic_tags & {'mountain', 'mountains', 'range', 'horizon', 'distant', 'far'}:
            return DepthLayer.BACKGROUND
        
        # Trees, buildings in distance -> mid background
        if semantic_tags & {'forest', 'hills', 'skyline', 'city', 'background'}:
            return DepthLayer.MID_BACKGROUND
        
        # Main scene elements -> midground
        if semantic_tags & {'house', 'tree', 'person', 'animal', 'building', 'scene'}:
            return DepthLayer.MIDGROUND
        
        # Closer objects -> mid foreground
        if semantic_tags & {'flower', 'plant', 'rock', 'fence', 'close'}:
            return DepthLayer.MID_FOREGROUND
        
        # Very close elements -> foreground
        if semantic_tags & {'leaf', 'grass', 'foreground', 'frame', 'border'}:
            return DepthLayer.FOREGROUND
        
        # Effects and overlays
        if semantic_tags & {'rain', 'snow', 'fog', 'mist', 'overlay', 'effect'}:
            return DepthLayer.OVERLAY
        
        # Default to midground
        return DepthLayer.MIDGROUND
    
    def render_to_svg(self, scene: MultiLayerScene) -> str:
        """Render multi-layer scene to SVG with atmospheric effects"""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}">'
        ]
        
        # Add filter definitions for atmospheric effects
        svg_parts.append(self._generate_filter_defs(scene))
        
        # Render layers from back to front
        for depth in sorted(scene.layers.keys(), key=lambda d: d.value):
            layer = scene.layers[depth]
            if layer.elements:
                svg_parts.append(self._render_layer(layer, scene))
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _generate_filter_defs(self, scene: MultiLayerScene) -> str:
        """Generate SVG filter definitions for atmospheric effects"""
        defs = ['<defs>']
        
        for depth, layer in scene.layers.items():
            atm = layer.atmospheric_effect
            filter_id = f"atm_{depth.name.lower()}"
            
            filters = [f'<filter id="{filter_id}">']
            
            # Gaussian blur for depth
            if atm.blur_amount > 0:
                filters.append(f'<feGaussianBlur stdDeviation="{atm.blur_amount}" result="blur"/>')
            
            # Fog/haze overlay
            if atm.fog_density > 0:
                # Create fog color flood
                filters.append(f'<feFlood flood-color="{atm.fog_color}" flood-opacity="{atm.fog_density}" result="fog"/>')
                filters.append('<feBlend in="SourceGraphic" in2="fog" mode="screen" result="fogged"/>')
            
            # Brightness/saturation adjustments
            if atm.brightness_shift != 0 or atm.saturation_shift != 0:
                # Simplified - just adjust brightness
                brightness = 1 + atm.brightness_shift
                filters.append(f'''
                    <feComponentTransfer result="adjusted">
                        <feFuncR type="linear" slope="{brightness}"/>
                        <feFuncG type="linear" slope="{brightness}"/>
                        <feFuncB type="linear" slope="{brightness}"/>
                    </feComponentTransfer>
                ''')
            
            filters.append('</filter>')
            defs.append('\n'.join(filters))
        
        # Add gradient for sky
        defs.append('''
            <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#1a1a2e"/>
                <stop offset="50%" style="stop-color:#16213e"/>
                <stop offset="100%" style="stop-color:#0f3460"/>
            </linearGradient>
        ''')
        
        defs.append('</defs>')
        return '\n'.join(defs)
    
    def _render_layer(self, layer: SceneLayer, scene: MultiLayerScene) -> str:
        """Render a single layer with its elements"""
        filter_id = f"atm_{layer.depth.name.lower()}"
        
        # Group for layer
        parts = [f'<g id="layer_{layer.depth.name.lower()}" filter="url(#{filter_id})" opacity="{1.0 - layer.atmospheric_effect.fog_density * 0.3}">']
        
        # Sort elements by z-index
        sorted_elements = sorted(layer.elements, key=lambda e: e.z_index)
        
        for element in sorted_elements:
            # Get pattern and render
            pattern = self.pattern_db.get_pattern(element.pattern_id) if self.pattern_db else None
            
            if pattern:
                # Render pattern SVG within element bounds
                transform = f'translate({element.x},{element.y}) scale({element.width/100},{element.height/100})'
                if element.rotation:
                    transform += f' rotate({element.rotation},{element.width/2},{element.height/2})'
                
                parts.append(f'<g transform="{transform}" opacity="{element.opacity}">')
                # Add pattern shapes
                for shape in pattern.shape_descriptors:
                    parts.append(shape.to_svg_element())
                parts.append('</g>')
            else:
                # Placeholder rectangle
                parts.append(f'<rect x="{element.x}" y="{element.y}" width="{element.width}" height="{element.height}" fill="#333" opacity="{element.opacity}" stroke="#666" stroke-width="1"/>')
        
        parts.append('</g>')
        return '\n'.join(parts)
    
    def compose_from_patterns(self, patterns: List[Dict], 
                              width: int = 800, height: int = 600) -> MultiLayerScene:
        """Compose a multi-layer scene from a list of patterns"""
        scene = self.create_scene("composed_scene", width, height)
        
        for i, p in enumerate(patterns):
            pattern_id = p.get('id', '')
            pattern_name = p.get('name', f'pattern_{i}')
            tags = set(p.get('tags', []))
            
            # Auto-assign layer
            layer = self.auto_assign_layer(pattern_name, tags)
            
            # Calculate position based on layer and index
            x = (i % 5) * (width // 5) + 20
            y = self._calculate_y_position(layer, height)
            
            self.assign_to_layer(scene, pattern_id, pattern_name, layer, x, y)
        
        return scene
    
    def _calculate_y_position(self, layer: DepthLayer, height: int) -> float:
        """Calculate Y position based on layer depth"""
        layer_positions = {
            DepthLayer.FAR_BACKGROUND: height * 0.05,
            DepthLayer.BACKGROUND: height * 0.15,
            DepthLayer.MID_BACKGROUND: height * 0.30,
            DepthLayer.MIDGROUND: height * 0.45,
            DepthLayer.MID_FOREGROUND: height * 0.65,
            DepthLayer.FOREGROUND: height * 0.80,
            DepthLayer.OVERLAY: height * 0.0
        }
        return layer_positions.get(layer, height * 0.5)
    
    def get_layer_info(self, scene: MultiLayerScene) -> Dict[str, Any]:
        """Get information about scene layers"""
        return {
            'total_layers': len([l for l in scene.layers.values() if l.elements]),
            'total_elements': sum(len(l.elements) for l in scene.layers.values()),
            'layers': {
                depth.name: {
                    'element_count': len(layer.elements),
                    'parallax': layer.parallax_factor,
                    'fog_density': layer.atmospheric_effect.fog_density,
                    'blur': layer.atmospheric_effect.blur_amount
                }
                for depth, layer in scene.layers.items()
            }
        }
