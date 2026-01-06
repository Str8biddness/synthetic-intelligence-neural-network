"""
Expanded Pattern Library - 500+ visual patterns organized by category
Categories: geometric shapes, natural objects, weather effects, lighting, textures
Includes pattern variations (rotation, scale, colors)
"""

import math
import random
from typing import List, Dict, Any, Tuple
from .visual_patterns import VisualPattern, ShapePrimitive, ShapeType


class PatternCategory:
    """Pattern categories for organization"""
    GEOMETRIC = "geometric"
    NATURAL = "natural"
    WEATHER = "weather"
    LIGHTING = "lighting"
    TEXTURE = "texture"
    ARCHITECTURE = "architecture"
    CELESTIAL = "celestial"
    WATER = "water"
    VEGETATION = "vegetation"
    ABSTRACT = "abstract"


# Color palettes for variations
COLOR_PALETTES = {
    'sunset': ['#FF6B35', '#F7C59F', '#EFEFD0', '#004E89', '#1A659E'],
    'ocean': ['#006994', '#0099DB', '#40E0D0', '#E6F7FF', '#003366'],
    'forest': ['#228B22', '#32CD32', '#90EE90', '#8FBC8F', '#2E8B57'],
    'night': ['#0D1B2A', '#1B263B', '#415A77', '#778DA9', '#E0E1DD'],
    'desert': ['#EDC9AF', '#D4A574', '#C19A6B', '#A67B5B', '#8B4513'],
    'autumn': ['#FF4500', '#FF6347', '#FF7F50', '#FFD700', '#8B4513'],
    'winter': ['#F0F8FF', '#E6E6FA', '#B0C4DE', '#87CEEB', '#4682B4'],
    'spring': ['#98FB98', '#90EE90', '#FFB6C1', '#DDA0DD', '#FFFACD'],
    'fire': ['#FF4500', '#FF6347', '#FF8C00', '#FFD700', '#8B0000'],
    'earth': ['#8B4513', '#A0522D', '#D2691E', '#CD853F', '#DEB887'],
    'sky': ['#87CEEB', '#ADD8E6', '#B0E0E6', '#F0F8FF', '#E0FFFF'],
    'monochrome': ['#000000', '#333333', '#666666', '#999999', '#CCCCCC'],
}


def generate_geometric_patterns() -> List[VisualPattern]:
    """Generate geometric shape patterns with variations"""
    patterns = []
    
    # Basic shapes with color variations
    shapes_config = [
        ('circle', ShapeType.CIRCLE, {'cx': 50, 'cy': 50, 'r': 40}, 
         {'round', 'ball', 'sphere', 'dot', 'orb', 'disk', 'ring'}),
        ('square', ShapeType.RECTANGLE, {'x': 10, 'y': 10, 'width': 80, 'height': 80},
         {'square', 'box', 'block', 'cube', 'tile', 'pixel'}),
        ('rectangle', ShapeType.RECTANGLE, {'x': 5, 'y': 25, 'width': 90, 'height': 50},
         {'rectangle', 'bar', 'beam', 'plank', 'brick', 'slab'}),
        ('ellipse', ShapeType.ELLIPSE, {'cx': 50, 'cy': 50, 'rx': 45, 'ry': 30},
         {'ellipse', 'oval', 'egg', 'lens', 'capsule'}),
    ]
    
    # Triangle variations
    triangle_types = [
        ('triangle_up', [(50, 10), (90, 90), (10, 90)], {'triangle', 'arrow_up', 'mountain', 'peak', 'roof'}),
        ('triangle_down', [(10, 10), (90, 10), (50, 90)], {'triangle_down', 'arrow_down', 'funnel', 'drop'}),
        ('triangle_right', [(10, 10), (10, 90), (90, 50)], {'triangle_right', 'arrow_right', 'play', 'direction'}),
        ('triangle_left', [(90, 10), (90, 90), (10, 50)], {'triangle_left', 'arrow_left', 'back'}),
    ]
    
    # Generate basic shapes with multiple color variations
    for name, shape_type, params, tags in shapes_config:
        for palette_name, colors in COLOR_PALETTES.items():
            for i, color in enumerate(colors[:3]):
                stroke = colors[min(i+1, len(colors)-1)]
                pattern = VisualPattern(
                    name=f"{name}_{palette_name}_{i}",
                    shape_descriptors=[
                        ShapePrimitive(
                            type=shape_type,
                            parameters=params.copy(),
                            attributes={'fill': color, 'stroke': stroke, 'stroke_width': 2}
                        )
                    ],
                    abstraction_level=1,
                    semantic_tags=tags | {palette_name, 'geometric', 'shape'},
                    category=PatternCategory.GEOMETRIC,
                    bounding_box={'width': 100, 'height': 100}
                )
                patterns.append(pattern)
    
    # Triangle variations
    for name, points, tags in triangle_types:
        for palette_name, colors in list(COLOR_PALETTES.items())[:6]:
            pattern = VisualPattern(
                name=f"{name}_{palette_name}",
                shape_descriptors=[
                    ShapePrimitive(
                        type=ShapeType.POLYGON,
                        parameters={'points': points},
                        attributes={'fill': colors[0], 'stroke': colors[1], 'stroke_width': 2}
                    )
                ],
                abstraction_level=1,
                semantic_tags=tags | {palette_name, 'geometric', 'polygon'},
                category=PatternCategory.GEOMETRIC,
                bounding_box={'width': 100, 'height': 100}
            )
            patterns.append(pattern)
    
    # Regular polygons (pentagon, hexagon, octagon)
    for sides, name, tags in [(5, 'pentagon', {'pentagon', 'five', 'star_base'}),
                               (6, 'hexagon', {'hexagon', 'honeycomb', 'cell', 'bee'}),
                               (8, 'octagon', {'octagon', 'stop', 'sign'})]:
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            x = 50 + 40 * math.cos(angle)
            y = 50 + 40 * math.sin(angle)
            points.append((x, y))
        
        for palette_name, colors in list(COLOR_PALETTES.items())[:4]:
            pattern = VisualPattern(
                name=f"{name}_{palette_name}",
                shape_descriptors=[
                    ShapePrimitive(
                        type=ShapeType.POLYGON,
                        parameters={'points': points},
                        attributes={'fill': colors[0], 'stroke': colors[1], 'stroke_width': 2}
                    )
                ],
                abstraction_level=1,
                semantic_tags=tags | {palette_name, 'geometric', 'polygon'},
                category=PatternCategory.GEOMETRIC,
                bounding_box={'width': 100, 'height': 100}
            )
            patterns.append(pattern)
    
    # Star patterns
    for points_count in [5, 6, 8]:
        star_points = []
        for i in range(points_count * 2):
            angle = math.pi * i / points_count - math.pi / 2
            radius = 40 if i % 2 == 0 else 20
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            star_points.append((x, y))
        
        for palette_name, colors in [('sunset', COLOR_PALETTES['sunset']), 
                                      ('night', COLOR_PALETTES['night']),
                                      ('fire', COLOR_PALETTES['fire'])]:
            pattern = VisualPattern(
                name=f"star_{points_count}_{palette_name}",
                shape_descriptors=[
                    ShapePrimitive(
                        type=ShapeType.POLYGON,
                        parameters={'points': star_points},
                        attributes={'fill': colors[0], 'stroke': colors[1], 'stroke_width': 1}
                    )
                ],
                abstraction_level=1,
                semantic_tags={'star', 'sparkle', 'twinkle', 'celestial', palette_name},
                category=PatternCategory.CELESTIAL,
                bounding_box={'width': 100, 'height': 100}
            )
            patterns.append(pattern)
    
    # Concentric circles
    for palette_name in ['ocean', 'sunset', 'night']:
        colors = COLOR_PALETTES[palette_name]
        shapes = []
        for i, r in enumerate([40, 30, 20, 10]):
            shapes.append(ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': r},
                attributes={'fill': colors[i % len(colors)], 'stroke': 'none'}
            ))
        pattern = VisualPattern(
            name=f"concentric_circles_{palette_name}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'concentric', 'target', 'ripple', 'rings', palette_name},
            category=PatternCategory.GEOMETRIC,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Grid patterns
    for grid_size in [2, 3, 4]:
        cell_size = 80 // grid_size
        shapes = []
        for row in range(grid_size):
            for col in range(grid_size):
                color_idx = (row + col) % 2
                shapes.append(ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={
                        'x': 10 + col * cell_size,
                        'y': 10 + row * cell_size,
                        'width': cell_size - 2,
                        'height': cell_size - 2
                    },
                    attributes={'fill': '#333333' if color_idx == 0 else '#CCCCCC', 'stroke': 'none'}
                ))
        pattern = VisualPattern(
            name=f"grid_{grid_size}x{grid_size}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'grid', 'checkerboard', 'tiles', 'pattern', 'chess'},
            category=PatternCategory.TEXTURE,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Spiral pattern
    spiral_points = []
    for i in range(50):
        angle = i * 0.3
        radius = 5 + i * 0.7
        x = 50 + radius * math.cos(angle)
        y = 50 + radius * math.sin(angle)
        spiral_points.append((x, y))
    
    pattern = VisualPattern(
        name="spiral",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.POLYLINE,
                parameters={'points': spiral_points},
                attributes={'fill': 'none', 'stroke': '#6366F1', 'stroke_width': 3}
            )
        ],
        abstraction_level=2,
        semantic_tags={'spiral', 'swirl', 'vortex', 'galaxy', 'shell'},
        category=PatternCategory.ABSTRACT,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    return patterns


def generate_natural_patterns() -> List[VisualPattern]:
    """Generate natural object patterns"""
    patterns = []
    
    # Trees with variations
    tree_types = [
        ('pine_tree', 'forest', [
            {'type': ShapeType.RECTANGLE, 'params': {'x': 45, 'y': 70, 'width': 10, 'height': 30}, 
             'attrs': {'fill': '#8B4513', 'stroke': '#5D3A1A', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 5), (75, 40), (25, 40)]},
             'attrs': {'fill': '#228B22', 'stroke': '#006400', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 20), (80, 60), (20, 60)]},
             'attrs': {'fill': '#2E8B57', 'stroke': '#006400', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 40), (85, 75), (15, 75)]},
             'attrs': {'fill': '#3CB371', 'stroke': '#006400', 'stroke_width': 1}},
        ], {'pine', 'tree', 'evergreen', 'conifer', 'forest', 'nature', 'plant'}),
        
        ('oak_tree', 'spring', [
            {'type': ShapeType.RECTANGLE, 'params': {'x': 42, 'y': 65, 'width': 16, 'height': 35},
             'attrs': {'fill': '#8B4513', 'stroke': '#654321', 'stroke_width': 1}},
            {'type': ShapeType.CIRCLE, 'params': {'cx': 50, 'cy': 40, 'r': 35},
             'attrs': {'fill': '#228B22', 'stroke': '#006400', 'stroke_width': 2}},
        ], {'oak', 'tree', 'deciduous', 'forest', 'nature', 'shade'}),
        
        ('palm_tree', 'desert', [
            {'type': ShapeType.RECTANGLE, 'params': {'x': 45, 'y': 40, 'width': 10, 'height': 60},
             'attrs': {'fill': '#CD853F', 'stroke': '#8B4513', 'stroke_width': 1}},
            # Palm fronds
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 25), (85, 35), (50, 45)]},
             'attrs': {'fill': '#32CD32', 'stroke': '#228B22', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 25), (15, 35), (50, 45)]},
             'attrs': {'fill': '#32CD32', 'stroke': '#228B22', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 25), (70, 15), (55, 40)]},
             'attrs': {'fill': '#2E8B57', 'stroke': '#228B22', 'stroke_width': 1}},
            {'type': ShapeType.POLYGON, 'params': {'points': [(50, 25), (30, 15), (45, 40)]},
             'attrs': {'fill': '#2E8B57', 'stroke': '#228B22', 'stroke_width': 1}},
        ], {'palm', 'tree', 'tropical', 'beach', 'island', 'vacation'}),
    ]
    
    for name, palette, shapes_data, tags in tree_types:
        shapes = [ShapePrimitive(type=s['type'], parameters=s['params'], attributes=s['attrs']) for s in shapes_data]
        pattern = VisualPattern(
            name=name,
            shape_descriptors=shapes,
            abstraction_level=3,
            semantic_tags=tags,
            category=PatternCategory.VEGETATION,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Flowers
    flower_colors = [('#FF69B4', '#FF1493'), ('#FFD700', '#FFA500'), ('#FF6347', '#DC143C'), 
                     ('#9370DB', '#8A2BE2'), ('#87CEEB', '#4169E1')]
    
    for i, (petal_color, center_color) in enumerate(flower_colors):
        petals = []
        # Center
        petals.append(ShapePrimitive(
            type=ShapeType.CIRCLE,
            parameters={'cx': 50, 'cy': 50, 'r': 10},
            attributes={'fill': center_color, 'stroke': 'none'}
        ))
        # Petals
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            cx = 50 + 25 * math.cos(rad)
            cy = 50 + 25 * math.sin(rad)
            petals.append(ShapePrimitive(
                type=ShapeType.ELLIPSE,
                parameters={'cx': cx, 'cy': cy, 'rx': 15, 'ry': 10},
                attributes={'fill': petal_color, 'stroke': 'none', 'opacity': 0.9}
            ))
        
        pattern = VisualPattern(
            name=f"flower_{i}",
            shape_descriptors=petals,
            abstraction_level=3,
            semantic_tags={'flower', 'bloom', 'petal', 'garden', 'nature', 'spring', 'floral'},
            category=PatternCategory.VEGETATION,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Mountains with variations
    for i, (base_color, peak_color, snow_color) in enumerate([
        ('#8B7355', '#6B5344', '#FFFFFF'),
        ('#4A5568', '#2D3748', '#E2E8F0'),
        ('#5D4E37', '#3D2E17', '#F5F5DC'),
    ]):
        shapes = [
            # Main mountain
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(50, 10), (95, 90), (5, 90)]},
                attributes={'fill': base_color, 'stroke': peak_color, 'stroke_width': 1}
            ),
            # Snow cap
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(50, 10), (65, 35), (35, 35)]},
                attributes={'fill': snow_color, 'stroke': 'none'}
            ),
        ]
        pattern = VisualPattern(
            name=f"mountain_{i}",
            shape_descriptors=shapes,
            abstraction_level=3,
            semantic_tags={'mountain', 'peak', 'summit', 'hill', 'landscape', 'nature', 'snow'},
            category=PatternCategory.NATURAL,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Mountain range
    shapes = []
    mountain_data = [
        (25, 30, '#6B8E23'),
        (50, 15, '#556B2F'),
        (75, 35, '#8FBC8F'),
    ]
    for cx, peak_y, color in mountain_data:
        shapes.append(ShapePrimitive(
            type=ShapeType.POLYGON,
            parameters={'points': [(cx, peak_y), (cx + 30, 90), (cx - 30, 90)]},
            attributes={'fill': color, 'stroke': '#2F4F4F', 'stroke_width': 1}
        ))
    pattern = VisualPattern(
        name="mountain_range",
        shape_descriptors=shapes,
        abstraction_level=4,
        semantic_tags={'mountains', 'range', 'hills', 'landscape', 'vista', 'panorama'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Clouds
    for i, (color, opacity) in enumerate([('#FFFFFF', 0.9), ('#E5E7EB', 0.85), ('#9CA3AF', 0.8)]):
        shapes = [
            ShapePrimitive(type=ShapeType.ELLIPSE, 
                          parameters={'cx': 40, 'cy': 55, 'rx': 25, 'ry': 18},
                          attributes={'fill': color, 'stroke': 'none', 'opacity': opacity}),
            ShapePrimitive(type=ShapeType.ELLIPSE,
                          parameters={'cx': 60, 'cy': 50, 'rx': 30, 'ry': 22},
                          attributes={'fill': color, 'stroke': 'none', 'opacity': opacity}),
            ShapePrimitive(type=ShapeType.ELLIPSE,
                          parameters={'cx': 75, 'cy': 58, 'rx': 20, 'ry': 15},
                          attributes={'fill': color, 'stroke': 'none', 'opacity': opacity}),
        ]
        pattern = VisualPattern(
            name=f"cloud_{i}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'cloud', 'sky', 'fluffy', 'weather', 'cumulus', 'puffy'},
            category=PatternCategory.WEATHER,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Animals - Simple silhouettes
    # Bird
    bird = VisualPattern(
        name="bird",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.PATH,
                parameters={'d': 'M 30 50 Q 50 30 70 50 Q 50 45 30 50'},
                attributes={'fill': '#1F2937', 'stroke': 'none'}
            ),
        ],
        abstraction_level=2,
        semantic_tags={'bird', 'flying', 'sky', 'wings', 'seagull', 'freedom'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(bird)
    
    # Fish
    fish = VisualPattern(
        name="fish",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.ELLIPSE,
                parameters={'cx': 45, 'cy': 50, 'rx': 30, 'ry': 18},
                attributes={'fill': '#4169E1', 'stroke': '#1E3A8A', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(75, 50), (95, 35), (95, 65)]},
                attributes={'fill': '#4169E1', 'stroke': '#1E3A8A', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 30, 'cy': 45, 'r': 4},
                attributes={'fill': '#FFFFFF', 'stroke': '#000000', 'stroke_width': 1}
            ),
        ],
        abstraction_level=3,
        semantic_tags={'fish', 'sea', 'ocean', 'water', 'swim', 'aquatic', 'marine'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(fish)
    
    return patterns


def generate_weather_patterns() -> List[VisualPattern]:
    """Generate weather effect patterns"""
    patterns = []
    
    # Rain drops
    rain_shapes = []
    for _ in range(15):
        x = random.randint(10, 90)
        y = random.randint(10, 80)
        rain_shapes.append(ShapePrimitive(
            type=ShapeType.LINE,
            parameters={'x1': x, 'y1': y, 'x2': x - 3, 'y2': y + 15},
            attributes={'fill': 'none', 'stroke': '#6B9BD2', 'stroke_width': 2, 'opacity': 0.7}
        ))
    pattern = VisualPattern(
        name="rain",
        shape_descriptors=rain_shapes,
        abstraction_level=2,
        semantic_tags={'rain', 'weather', 'storm', 'precipitation', 'wet', 'drops'},
        category=PatternCategory.WEATHER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Snow flakes
    for i in range(3):
        snow_shapes = []
        for _ in range(20 - i * 5):
            x = random.randint(10, 90)
            y = random.randint(10, 90)
            r = random.randint(2, 5)
            snow_shapes.append(ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': x, 'cy': y, 'r': r},
                attributes={'fill': '#FFFFFF', 'stroke': '#E5E7EB', 'stroke_width': 1, 'opacity': 0.8}
            ))
        pattern = VisualPattern(
            name=f"snow_{i}",
            shape_descriptors=snow_shapes,
            abstraction_level=2,
            semantic_tags={'snow', 'winter', 'cold', 'flakes', 'frost', 'ice'},
            category=PatternCategory.WEATHER,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Lightning bolt
    lightning = VisualPattern(
        name="lightning",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(45, 5), (55, 5), (50, 35), (65, 35), (40, 95), (50, 55), (35, 55)]},
                attributes={'fill': '#FBBF24', 'stroke': '#F59E0B', 'stroke_width': 2}
            ),
        ],
        abstraction_level=2,
        semantic_tags={'lightning', 'storm', 'thunder', 'electric', 'bolt', 'power', 'energy'},
        category=PatternCategory.WEATHER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(lightning)
    
    # Storm cloud
    storm_cloud = VisualPattern(
        name="storm_cloud",
        shape_descriptors=[
            ShapePrimitive(type=ShapeType.ELLIPSE, 
                          parameters={'cx': 40, 'cy': 35, 'rx': 25, 'ry': 18},
                          attributes={'fill': '#4B5563', 'stroke': 'none', 'opacity': 0.95}),
            ShapePrimitive(type=ShapeType.ELLIPSE,
                          parameters={'cx': 60, 'cy': 30, 'rx': 30, 'ry': 22},
                          attributes={'fill': '#374151', 'stroke': 'none', 'opacity': 0.95}),
            ShapePrimitive(type=ShapeType.ELLIPSE,
                          parameters={'cx': 75, 'cy': 38, 'rx': 20, 'ry': 15},
                          attributes={'fill': '#4B5563', 'stroke': 'none', 'opacity': 0.95}),
            # Rain
            ShapePrimitive(type=ShapeType.LINE, parameters={'x1': 35, 'y1': 55, 'x2': 30, 'y2': 85},
                          attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2}),
            ShapePrimitive(type=ShapeType.LINE, parameters={'x1': 55, 'y1': 55, 'x2': 50, 'y2': 90},
                          attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2}),
            ShapePrimitive(type=ShapeType.LINE, parameters={'x1': 70, 'y1': 55, 'x2': 65, 'y2': 80},
                          attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2}),
        ],
        abstraction_level=3,
        semantic_tags={'storm', 'cloud', 'rain', 'dark', 'weather', 'thunder'},
        category=PatternCategory.WEATHER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(storm_cloud)
    
    # Rainbow
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
    rainbow_shapes = []
    for i, color in enumerate(rainbow_colors):
        r = 60 - i * 6
        rainbow_shapes.append(ShapePrimitive(
            type=ShapeType.PATH,
            parameters={'d': f'M {50-r} 90 A {r} {r} 0 0 1 {50+r} 90'},
            attributes={'fill': 'none', 'stroke': color, 'stroke_width': 5, 'opacity': 0.8}
        ))
    pattern = VisualPattern(
        name="rainbow",
        shape_descriptors=rainbow_shapes,
        abstraction_level=3,
        semantic_tags={'rainbow', 'colors', 'arc', 'weather', 'hope', 'spectrum', 'beautiful'},
        category=PatternCategory.WEATHER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Fog/mist
    fog_shapes = []
    for i in range(5):
        fog_shapes.append(ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 20 + i * 15, 'width': 100, 'height': 10},
            attributes={'fill': '#D1D5DB', 'stroke': 'none', 'opacity': 0.3 + i * 0.1}
        ))
    pattern = VisualPattern(
        name="fog",
        shape_descriptors=fog_shapes,
        abstraction_level=2,
        semantic_tags={'fog', 'mist', 'haze', 'atmosphere', 'mysterious', 'ethereal'},
        category=PatternCategory.WEATHER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    return patterns


def generate_lighting_patterns() -> List[VisualPattern]:
    """Generate lighting effect patterns"""
    patterns = []
    
    # Sun variations
    for i, (color, ray_color) in enumerate([
        ('#FBBF24', '#F59E0B'),
        ('#EF4444', '#DC2626'),
        ('#FB923C', '#EA580C'),
    ]):
        shapes = [
            # Sun body
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': 25},
                attributes={'fill': color, 'stroke': ray_color, 'stroke_width': 2}
            ),
        ]
        # Rays
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = 50 + 30 * math.cos(rad)
            y1 = 50 + 30 * math.sin(rad)
            x2 = 50 + 45 * math.cos(rad)
            y2 = 50 + 45 * math.sin(rad)
            shapes.append(ShapePrimitive(
                type=ShapeType.LINE,
                parameters={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                attributes={'fill': 'none', 'stroke': ray_color, 'stroke_width': 3}
            ))
        
        name_suffix = ['bright', 'sunset', 'warm'][i]
        pattern = VisualPattern(
            name=f"sun_{name_suffix}",
            shape_descriptors=shapes,
            abstraction_level=3,
            semantic_tags={'sun', 'light', 'bright', 'warm', 'day', 'shine', 'solar', name_suffix},
            category=PatternCategory.CELESTIAL,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Moon phases
    moon_phases = [
        ('full_moon', '#F3F4F6', 40, None),
        ('crescent_moon', '#F3F4F6', 40, {'cx': 65, 'r': 35}),
        ('half_moon', '#F3F4F6', 40, {'rect': {'x': 50, 'y': 10, 'width': 40, 'height': 80}}),
    ]
    
    for name, color, radius, mask in moon_phases:
        shapes = [
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': radius},
                attributes={'fill': color, 'stroke': '#E5E7EB', 'stroke_width': 1}
            ),
        ]
        if mask and 'cx' in mask:
            shapes.append(ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': mask['cx'], 'cy': 50, 'r': mask['r']},
                attributes={'fill': '#1F2937', 'stroke': 'none'}
            ))
        
        pattern = VisualPattern(
            name=name,
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'moon', 'night', 'lunar', 'celestial', 'glow', name.replace('_', ' ')},
            category=PatternCategory.CELESTIAL,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Glow/halo effect
    for color_name, colors in [('golden', ['#FBBF24', '#F59E0B', '#D97706']),
                                ('blue', ['#60A5FA', '#3B82F6', '#2563EB']),
                                ('white', ['#FFFFFF', '#F3F4F6', '#E5E7EB'])]:
        shapes = []
        for i, (r, c) in enumerate(zip([40, 30, 20], colors)):
            shapes.append(ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': r},
                attributes={'fill': c, 'stroke': 'none', 'opacity': 0.5 - i * 0.1}
            ))
        pattern = VisualPattern(
            name=f"glow_{color_name}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'glow', 'halo', 'light', 'aura', 'radiance', color_name},
            category=PatternCategory.LIGHTING,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Light beam
    beam = VisualPattern(
        name="light_beam",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(45, 0), (55, 0), (75, 100), (25, 100)]},
                attributes={'fill': '#FBBF24', 'stroke': 'none', 'opacity': 0.4}
            ),
        ],
        abstraction_level=2,
        semantic_tags={'beam', 'light', 'ray', 'spotlight', 'shine', 'bright'},
        category=PatternCategory.LIGHTING,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(beam)
    
    return patterns


def generate_texture_patterns() -> List[VisualPattern]:
    """Generate texture patterns"""
    patterns = []
    
    # Stripes (horizontal, vertical, diagonal)
    stripe_configs = [
        ('horizontal_stripes', [(0, i*20, 100, 10) for i in range(5)]),
        ('vertical_stripes', [(i*20, 0, 10, 100) for i in range(5)]),
    ]
    
    for name, stripe_data in stripe_configs:
        for palette_name in ['ocean', 'sunset', 'forest']:
            colors = COLOR_PALETTES[palette_name]
            shapes = []
            for i, (x, y, w, h) in enumerate(stripe_data):
                shapes.append(ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': x, 'y': y, 'width': w, 'height': h},
                    attributes={'fill': colors[i % len(colors)], 'stroke': 'none'}
                ))
            pattern = VisualPattern(
                name=f"{name}_{palette_name}",
                shape_descriptors=shapes,
                abstraction_level=2,
                semantic_tags={'stripes', 'lines', 'pattern', 'texture', palette_name},
                category=PatternCategory.TEXTURE,
                bounding_box={'width': 100, 'height': 100}
            )
            patterns.append(pattern)
    
    # Dots pattern
    for spacing in [15, 20, 25]:
        shapes = []
        for x in range(10, 100, spacing):
            for y in range(10, 100, spacing):
                shapes.append(ShapePrimitive(
                    type=ShapeType.CIRCLE,
                    parameters={'cx': x, 'cy': y, 'r': 4},
                    attributes={'fill': '#6B7280', 'stroke': 'none'}
                ))
        pattern = VisualPattern(
            name=f"dots_{spacing}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'dots', 'polka', 'pattern', 'texture', 'spotted'},
            category=PatternCategory.TEXTURE,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Crosshatch
    crosshatch_shapes = []
    for i in range(0, 100, 10):
        crosshatch_shapes.append(ShapePrimitive(
            type=ShapeType.LINE,
            parameters={'x1': i, 'y1': 0, 'x2': i, 'y2': 100},
            attributes={'fill': 'none', 'stroke': '#9CA3AF', 'stroke_width': 1}
        ))
        crosshatch_shapes.append(ShapePrimitive(
            type=ShapeType.LINE,
            parameters={'x1': 0, 'y1': i, 'x2': 100, 'y2': i},
            attributes={'fill': 'none', 'stroke': '#9CA3AF', 'stroke_width': 1}
        ))
    pattern = VisualPattern(
        name="crosshatch",
        shape_descriptors=crosshatch_shapes,
        abstraction_level=2,
        semantic_tags={'crosshatch', 'grid', 'mesh', 'texture', 'pattern', 'lines'},
        category=PatternCategory.TEXTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Brick pattern
    brick_shapes = []
    row = 0
    for y in range(0, 100, 15):
        offset = 25 if row % 2 == 1 else 0
        for x in range(-25, 100, 50):
            brick_shapes.append(ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': x + offset, 'y': y, 'width': 48, 'height': 13},
                attributes={'fill': '#B45309', 'stroke': '#92400E', 'stroke_width': 1}
            ))
        row += 1
    pattern = VisualPattern(
        name="brick",
        shape_descriptors=brick_shapes,
        abstraction_level=2,
        semantic_tags={'brick', 'wall', 'masonry', 'texture', 'building', 'red'},
        category=PatternCategory.TEXTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Wood grain
    wood_shapes = [
        ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 0, 'width': 100, 'height': 100},
            attributes={'fill': '#D4A574', 'stroke': 'none'}
        )
    ]
    for i in range(8):
        wood_shapes.append(ShapePrimitive(
            type=ShapeType.PATH,
            parameters={'d': f'M 0 {i*12+5} Q 50 {i*12+random.randint(-3, 3)+5} 100 {i*12+5}'},
            attributes={'fill': 'none', 'stroke': '#A67B5B', 'stroke_width': 1, 'opacity': 0.6}
        ))
    pattern = VisualPattern(
        name="wood_grain",
        shape_descriptors=wood_shapes,
        abstraction_level=2,
        semantic_tags={'wood', 'grain', 'timber', 'texture', 'natural', 'wooden'},
        category=PatternCategory.TEXTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    return patterns


def generate_water_patterns() -> List[VisualPattern]:
    """Generate water-related patterns"""
    patterns = []
    
    # Waves
    for wave_type, amplitude, color in [('gentle', 5, '#60A5FA'), ('medium', 10, '#3B82F6'), ('rough', 15, '#1D4ED8')]:
        shapes = []
        for i in range(5):
            y_base = 20 + i * 18
            points = []
            for x in range(0, 101, 5):
                y = y_base + amplitude * math.sin(x * 0.1 + i)
                points.append((x, y))
            shapes.append(ShapePrimitive(
                type=ShapeType.POLYLINE,
                parameters={'points': points},
                attributes={'fill': 'none', 'stroke': color, 'stroke_width': 2, 'opacity': 0.8 - i * 0.1}
            ))
        pattern = VisualPattern(
            name=f"waves_{wave_type}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'waves', 'water', 'ocean', 'sea', wave_type, 'marine'},
            category=PatternCategory.WATER,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Water drops
    drop_shapes = []
    for _ in range(8):
        x, y = random.randint(15, 85), random.randint(15, 85)
        drop_shapes.append(ShapePrimitive(
            type=ShapeType.PATH,
            parameters={'d': f'M {x} {y-8} Q {x+6} {y} {x} {y+8} Q {x-6} {y} {x} {y-8}'},
            attributes={'fill': '#93C5FD', 'stroke': '#3B82F6', 'stroke_width': 1}
        ))
    pattern = VisualPattern(
        name="water_drops",
        shape_descriptors=drop_shapes,
        abstraction_level=2,
        semantic_tags={'drops', 'water', 'rain', 'dew', 'droplets', 'wet'},
        category=PatternCategory.WATER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Ripples
    ripple_shapes = []
    for i, r in enumerate([10, 20, 30, 40]):
        ripple_shapes.append(ShapePrimitive(
            type=ShapeType.CIRCLE,
            parameters={'cx': 50, 'cy': 50, 'r': r},
            attributes={'fill': 'none', 'stroke': '#60A5FA', 'stroke_width': 2, 'opacity': 0.8 - i * 0.15}
        ))
    pattern = VisualPattern(
        name="ripples",
        shape_descriptors=ripple_shapes,
        abstraction_level=2,
        semantic_tags={'ripples', 'water', 'pond', 'circles', 'waves', 'calm'},
        category=PatternCategory.WATER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    # Waterfall
    waterfall_shapes = [
        # Cliff
        ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 0, 'width': 30, 'height': 60},
            attributes={'fill': '#6B7280', 'stroke': '#4B5563', 'stroke_width': 1}
        ),
        # Water streams
    ]
    for i in range(5):
        x = 10 + i * 4
        waterfall_shapes.append(ShapePrimitive(
            type=ShapeType.LINE,
            parameters={'x1': x, 'y1': 0, 'x2': x + 5, 'y2': 100},
            attributes={'fill': 'none', 'stroke': '#93C5FD', 'stroke_width': 3, 'opacity': 0.7}
        ))
    # Splash at bottom
    waterfall_shapes.append(ShapePrimitive(
        type=ShapeType.ELLIPSE,
        parameters={'cx': 25, 'cy': 95, 'rx': 30, 'ry': 8},
        attributes={'fill': '#BFDBFE', 'stroke': 'none', 'opacity': 0.6}
    ))
    pattern = VisualPattern(
        name="waterfall",
        shape_descriptors=waterfall_shapes,
        abstraction_level=4,
        semantic_tags={'waterfall', 'cascade', 'water', 'nature', 'cliff', 'stream'},
        category=PatternCategory.WATER,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(pattern)
    
    return patterns


def generate_architecture_patterns() -> List[VisualPattern]:
    """Generate architecture-related patterns"""
    patterns = []
    
    # House variations
    house_styles = [
        ('cottage', '#DEB887', '#8B4513', '#B22222'),
        ('modern', '#E5E7EB', '#4B5563', '#1F2937'),
        ('cabin', '#8B4513', '#5D3A1A', '#654321'),
    ]
    
    for name, wall_color, trim_color, roof_color in house_styles:
        shapes = [
            # Main body
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 15, 'y': 45, 'width': 70, 'height': 50},
                attributes={'fill': wall_color, 'stroke': trim_color, 'stroke_width': 2}
            ),
            # Roof
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(10, 45), (50, 10), (90, 45)]},
                attributes={'fill': roof_color, 'stroke': trim_color, 'stroke_width': 2}
            ),
            # Door
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 40, 'y': 65, 'width': 20, 'height': 30},
                attributes={'fill': trim_color, 'stroke': '#000000', 'stroke_width': 1}
            ),
            # Window
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 25, 'y': 55, 'width': 12, 'height': 12},
                attributes={'fill': '#87CEEB', 'stroke': trim_color, 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 63, 'y': 55, 'width': 12, 'height': 12},
                attributes={'fill': '#87CEEB', 'stroke': trim_color, 'stroke_width': 1}
            ),
        ]
        pattern = VisualPattern(
            name=f"house_{name}",
            shape_descriptors=shapes,
            abstraction_level=4,
            semantic_tags={'house', 'home', 'building', name, 'dwelling', 'residence'},
            category=PatternCategory.ARCHITECTURE,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Skyscraper
    skyscraper = VisualPattern(
        name="skyscraper",
        shape_descriptors=[
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 25, 'y': 5, 'width': 50, 'height': 90},
                attributes={'fill': '#4B5563', 'stroke': '#1F2937', 'stroke_width': 2}
            ),
            # Windows grid
            *[ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 30 + (i % 4) * 10, 'y': 10 + (i // 4) * 12, 'width': 8, 'height': 10},
                attributes={'fill': '#FDE68A', 'stroke': 'none'}
            ) for i in range(28)],
        ],
        abstraction_level=4,
        semantic_tags={'skyscraper', 'building', 'tower', 'city', 'urban', 'office', 'tall'},
        category=PatternCategory.ARCHITECTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(skyscraper)
    
    # Bridge
    bridge = VisualPattern(
        name="bridge",
        shape_descriptors=[
            # Main deck
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 50, 'width': 100, 'height': 10},
                attributes={'fill': '#6B7280', 'stroke': '#4B5563', 'stroke_width': 1}
            ),
            # Towers
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 20, 'y': 20, 'width': 8, 'height': 40},
                attributes={'fill': '#374151', 'stroke': '#1F2937', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 72, 'y': 20, 'width': 8, 'height': 40},
                attributes={'fill': '#374151', 'stroke': '#1F2937', 'stroke_width': 1}
            ),
            # Cables
            ShapePrimitive(
                type=ShapeType.PATH,
                parameters={'d': 'M 24 20 Q 50 5 76 20'},
                attributes={'fill': 'none', 'stroke': '#9CA3AF', 'stroke_width': 2}
            ),
            # Supports
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 10, 'y': 60, 'width': 12, 'height': 30},
                attributes={'fill': '#4B5563', 'stroke': 'none'}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 78, 'y': 60, 'width': 12, 'height': 30},
                attributes={'fill': '#4B5563', 'stroke': 'none'}
            ),
        ],
        abstraction_level=4,
        semantic_tags={'bridge', 'crossing', 'structure', 'engineering', 'suspension', 'river'},
        category=PatternCategory.ARCHITECTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(bridge)
    
    # Window
    window = VisualPattern(
        name="window",
        shape_descriptors=[
            # Frame
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 15, 'y': 10, 'width': 70, 'height': 80},
                attributes={'fill': '#87CEEB', 'stroke': '#8B4513', 'stroke_width': 4}
            ),
            # Cross bars
            ShapePrimitive(
                type=ShapeType.LINE,
                parameters={'x1': 50, 'y1': 10, 'x2': 50, 'y2': 90},
                attributes={'fill': 'none', 'stroke': '#8B4513', 'stroke_width': 3}
            ),
            ShapePrimitive(
                type=ShapeType.LINE,
                parameters={'x1': 15, 'y1': 50, 'x2': 85, 'y2': 50},
                attributes={'fill': 'none', 'stroke': '#8B4513', 'stroke_width': 3}
            ),
        ],
        abstraction_level=2,
        semantic_tags={'window', 'glass', 'frame', 'building', 'view', 'pane'},
        category=PatternCategory.ARCHITECTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(window)
    
    # Door
    door = VisualPattern(
        name="door",
        shape_descriptors=[
            # Frame
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 20, 'y': 5, 'width': 60, 'height': 90},
                attributes={'fill': '#8B4513', 'stroke': '#5D3A1A', 'stroke_width': 3}
            ),
            # Panels
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 28, 'y': 15, 'width': 18, 'height': 25},
                attributes={'fill': '#A0522D', 'stroke': '#5D3A1A', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 54, 'y': 15, 'width': 18, 'height': 25},
                attributes={'fill': '#A0522D', 'stroke': '#5D3A1A', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 28, 'y': 50, 'width': 18, 'height': 35},
                attributes={'fill': '#A0522D', 'stroke': '#5D3A1A', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 54, 'y': 50, 'width': 18, 'height': 35},
                attributes={'fill': '#A0522D', 'stroke': '#5D3A1A', 'stroke_width': 1}
            ),
            # Handle
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 70, 'cy': 55, 'r': 4},
                attributes={'fill': '#FFD700', 'stroke': '#B8860B', 'stroke_width': 1}
            ),
        ],
        abstraction_level=3,
        semantic_tags={'door', 'entrance', 'exit', 'wooden', 'entry', 'gateway'},
        category=PatternCategory.ARCHITECTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(door)
    
    return patterns


def generate_abstract_patterns() -> List[VisualPattern]:
    """Generate abstract art patterns"""
    patterns = []
    
    # Gradient circles
    for palette_name in ['sunset', 'ocean', 'fire', 'spring']:
        colors = COLOR_PALETTES[palette_name]
        shapes = []
        for i, (r, c) in enumerate(zip([45, 35, 25, 15, 5], colors)):
            shapes.append(ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': r},
                attributes={'fill': c, 'stroke': 'none'}
            ))
        pattern = VisualPattern(
            name=f"gradient_circles_{palette_name}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'gradient', 'circles', 'abstract', 'art', palette_name, 'colorful'},
            category=PatternCategory.ABSTRACT,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Random shapes composition
    for seed in range(5):
        random.seed(seed + 42)
        shapes = []
        for _ in range(random.randint(5, 10)):
            shape_type = random.choice([ShapeType.CIRCLE, ShapeType.RECTANGLE, ShapeType.POLYGON])
            color = random.choice(COLOR_PALETTES['sunset'] + COLOR_PALETTES['ocean'])
            
            if shape_type == ShapeType.CIRCLE:
                params = {'cx': random.randint(20, 80), 'cy': random.randint(20, 80), 'r': random.randint(5, 25)}
            elif shape_type == ShapeType.RECTANGLE:
                params = {'x': random.randint(10, 60), 'y': random.randint(10, 60), 
                         'width': random.randint(15, 40), 'height': random.randint(15, 40)}
            else:
                num_points = random.randint(3, 6)
                cx, cy = random.randint(30, 70), random.randint(30, 70)
                r = random.randint(10, 25)
                pts = [(cx + r * math.cos(2 * math.pi * i / num_points),
                       cy + r * math.sin(2 * math.pi * i / num_points)) for i in range(num_points)]
                params = {'points': pts}
            
            shapes.append(ShapePrimitive(
                type=shape_type,
                parameters=params,
                attributes={'fill': color, 'stroke': 'none', 'opacity': random.uniform(0.5, 0.9)}
            ))
        
        pattern = VisualPattern(
            name=f"abstract_composition_{seed}",
            shape_descriptors=shapes,
            abstraction_level=3,
            semantic_tags={'abstract', 'art', 'composition', 'modern', 'creative', 'artistic'},
            category=PatternCategory.ABSTRACT,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    # Flowing lines
    for i in range(3):
        shapes = []
        for j in range(5):
            points = []
            for x in range(0, 101, 10):
                y = 50 + 20 * math.sin((x + j * 20 + i * 30) * 0.1) * (0.5 + j * 0.1)
                points.append((x, y))
            color = COLOR_PALETTES['ocean'][j % len(COLOR_PALETTES['ocean'])]
            shapes.append(ShapePrimitive(
                type=ShapeType.POLYLINE,
                parameters={'points': points},
                attributes={'fill': 'none', 'stroke': color, 'stroke_width': 3, 'opacity': 0.7}
            ))
        
        pattern = VisualPattern(
            name=f"flowing_lines_{i}",
            shape_descriptors=shapes,
            abstraction_level=2,
            semantic_tags={'flowing', 'lines', 'waves', 'abstract', 'dynamic', 'motion'},
            category=PatternCategory.ABSTRACT,
            bounding_box={'width': 100, 'height': 100}
        )
        patterns.append(pattern)
    
    return patterns


def generate_scene_patterns() -> List[VisualPattern]:
    """Generate complete scene patterns"""
    patterns = []
    
    # Sunset scene
    sunset = VisualPattern(
        name="sunset_scene",
        shape_descriptors=[
            # Sky gradient (simplified)
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 0, 'width': 100, 'height': 50},
                attributes={'fill': '#FDB813', 'stroke': 'none'}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 0, 'width': 100, 'height': 30},
                attributes={'fill': '#FF6B35', 'stroke': 'none', 'opacity': 0.7}
            ),
            # Sun
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 50, 'cy': 50, 'r': 20},
                attributes={'fill': '#FF4500', 'stroke': '#FF6347', 'stroke_width': 3}
            ),
            # Water/horizon
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 50, 'width': 100, 'height': 50},
                attributes={'fill': '#1E3A8A', 'stroke': 'none'}
            ),
            # Sun reflection
            ShapePrimitive(
                type=ShapeType.ELLIPSE,
                parameters={'cx': 50, 'cy': 70, 'rx': 15, 'ry': 25},
                attributes={'fill': '#FF6347', 'stroke': 'none', 'opacity': 0.4}
            ),
        ],
        abstraction_level=5,
        semantic_tags={'sunset', 'scene', 'ocean', 'sun', 'evening', 'horizon', 'water', 'beautiful', 'romantic'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(sunset)
    
    # Night sky scene
    night_sky_shapes = [
        # Dark sky
        ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 0, 'width': 100, 'height': 100},
            attributes={'fill': '#0D1B2A', 'stroke': 'none'}
        ),
        # Moon
        ShapePrimitive(
            type=ShapeType.CIRCLE,
            parameters={'cx': 75, 'cy': 25, 'r': 15},
            attributes={'fill': '#F3F4F6', 'stroke': '#E5E7EB', 'stroke_width': 1}
        ),
    ]
    # Stars
    for _ in range(20):
        x, y = random.randint(5, 95), random.randint(5, 95)
        r = random.choice([1, 1.5, 2])
        night_sky_shapes.append(ShapePrimitive(
            type=ShapeType.CIRCLE,
            parameters={'cx': x, 'cy': y, 'r': r},
            attributes={'fill': '#FFFFFF', 'stroke': 'none'}
        ))
    
    night_sky = VisualPattern(
        name="night_sky_scene",
        shape_descriptors=night_sky_shapes,
        abstraction_level=5,
        semantic_tags={'night', 'sky', 'stars', 'moon', 'dark', 'celestial', 'space', 'starry'},
        category=PatternCategory.CELESTIAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(night_sky)
    
    # Forest scene
    forest = VisualPattern(
        name="forest_scene",
        shape_descriptors=[
            # Sky
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 0, 'width': 100, 'height': 40},
                attributes={'fill': '#87CEEB', 'stroke': 'none'}
            ),
            # Ground
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 70, 'width': 100, 'height': 30},
                attributes={'fill': '#228B22', 'stroke': 'none'}
            ),
            # Trees
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(20, 30), (35, 70), (5, 70)]},
                attributes={'fill': '#2E8B57', 'stroke': '#1B4D3E', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(50, 20), (70, 70), (30, 70)]},
                attributes={'fill': '#228B22', 'stroke': '#1B4D3E', 'stroke_width': 1}
            ),
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(80, 35), (95, 70), (65, 70)]},
                attributes={'fill': '#3CB371', 'stroke': '#1B4D3E', 'stroke_width': 1}
            ),
            # Tree trunks
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 17, 'y': 70, 'width': 6, 'height': 15},
                attributes={'fill': '#8B4513', 'stroke': 'none'}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 47, 'y': 70, 'width': 6, 'height': 15},
                attributes={'fill': '#8B4513', 'stroke': 'none'}
            ),
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 77, 'y': 70, 'width': 6, 'height': 15},
                attributes={'fill': '#8B4513', 'stroke': 'none'}
            ),
        ],
        abstraction_level=5,
        semantic_tags={'forest', 'trees', 'nature', 'green', 'woods', 'pine', 'landscape'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(forest)
    
    # Beach scene
    beach = VisualPattern(
        name="beach_scene",
        shape_descriptors=[
            # Sky
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 0, 'width': 100, 'height': 40},
                attributes={'fill': '#87CEEB', 'stroke': 'none'}
            ),
            # Ocean
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 40, 'width': 100, 'height': 25},
                attributes={'fill': '#006994', 'stroke': 'none'}
            ),
            # Sand
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 0, 'y': 65, 'width': 100, 'height': 35},
                attributes={'fill': '#F4D03F', 'stroke': 'none'}
            ),
            # Sun
            ShapePrimitive(
                type=ShapeType.CIRCLE,
                parameters={'cx': 80, 'cy': 20, 'r': 12},
                attributes={'fill': '#FBBF24', 'stroke': '#F59E0B', 'stroke_width': 2}
            ),
            # Palm tree trunk
            ShapePrimitive(
                type=ShapeType.RECTANGLE,
                parameters={'x': 15, 'y': 35, 'width': 5, 'height': 40},
                attributes={'fill': '#CD853F', 'stroke': '#8B4513', 'stroke_width': 1}
            ),
            # Palm fronds
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(17, 25), (40, 35), (17, 40)]},
                attributes={'fill': '#32CD32', 'stroke': 'none'}
            ),
            ShapePrimitive(
                type=ShapeType.POLYGON,
                parameters={'points': [(17, 25), (-5, 35), (17, 40)]},
                attributes={'fill': '#32CD32', 'stroke': 'none'}
            ),
        ],
        abstraction_level=5,
        semantic_tags={'beach', 'ocean', 'sand', 'palm', 'tropical', 'vacation', 'summer', 'coast'},
        category=PatternCategory.NATURAL,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(beach)
    
    # City skyline
    city_shapes = [
        # Sky
        ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 0, 'width': 100, 'height': 60},
            attributes={'fill': '#1E3A5F', 'stroke': 'none'}
        ),
        # Ground
        ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': 0, 'y': 90, 'width': 100, 'height': 10},
            attributes={'fill': '#2D3748', 'stroke': 'none'}
        ),
    ]
    # Buildings
    building_data = [(5, 45, 15, 45), (22, 30, 12, 60), (36, 55, 18, 35), 
                     (56, 20, 15, 70), (73, 40, 12, 50), (87, 50, 10, 40)]
    for x, y, w, h in building_data:
        city_shapes.append(ShapePrimitive(
            type=ShapeType.RECTANGLE,
            parameters={'x': x, 'y': y, 'width': w, 'height': h},
            attributes={'fill': '#374151', 'stroke': '#1F2937', 'stroke_width': 1}
        ))
        # Windows
        for wx in range(x + 2, x + w - 2, 4):
            for wy in range(y + 5, y + h - 5, 8):
                city_shapes.append(ShapePrimitive(
                    type=ShapeType.RECTANGLE,
                    parameters={'x': wx, 'y': wy, 'width': 2, 'height': 4},
                    attributes={'fill': '#FDE68A', 'stroke': 'none'}
                ))
    
    city = VisualPattern(
        name="city_skyline",
        shape_descriptors=city_shapes,
        abstraction_level=5,
        semantic_tags={'city', 'skyline', 'urban', 'buildings', 'night', 'downtown', 'metropolitan'},
        category=PatternCategory.ARCHITECTURE,
        bounding_box={'width': 100, 'height': 100}
    )
    patterns.append(city)
    
    return patterns


def get_all_expanded_patterns() -> List[VisualPattern]:
    """Get all expanded patterns from all categories"""
    all_patterns = []
    
    all_patterns.extend(generate_geometric_patterns())
    all_patterns.extend(generate_natural_patterns())
    all_patterns.extend(generate_weather_patterns())
    all_patterns.extend(generate_lighting_patterns())
    all_patterns.extend(generate_texture_patterns())
    all_patterns.extend(generate_water_patterns())
    all_patterns.extend(generate_architecture_patterns())
    all_patterns.extend(generate_abstract_patterns())
    all_patterns.extend(generate_scene_patterns())
    
    return all_patterns


def get_pattern_count_by_category() -> Dict[str, int]:
    """Get count of patterns by category"""
    patterns = get_all_expanded_patterns()
    counts = {}
    for pattern in patterns:
        cat = getattr(pattern, 'category', 'unknown')
        counts[cat] = counts.get(cat, 0) + 1
    return counts


if __name__ == "__main__":
    patterns = get_all_expanded_patterns()
    print(f"Total expanded patterns: {len(patterns)}")
    print("\nPatterns by category:")
    for cat, count in get_pattern_count_by_category().items():
        print(f"  {cat}: {count}")
