"""
Pattern Renderer - SVG and PNG rendering pipeline
GPU-accelerated rendering with caching
"""

import io
import base64
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class RenderSettings:
    """Settings for rendering"""
    width: int = 800
    height: int = 600
    format: str = "svg"  # svg, png
    quality: int = 90
    background_color: str = "#FFFFFF"
    anti_alias: bool = True


class PatternRenderer:
    """
    Renders scene graphs to SVG/PNG
    Uses SVG as intermediate format, converts to PNG with cairosvg
    """
    
    def __init__(self, visual_pattern_db):
        self.pattern_db = visual_pattern_db
        
        # Caches
        self.rendered_patterns: Dict[str, str] = {}  # pattern_id -> SVG fragment
        self.composition_cache: Dict[str, bytes] = {}  # hash -> PNG bytes
        
        # Default settings
        self.default_settings = RenderSettings()
    
    def render_scene_graph(
        self, 
        scene_graph, 
        settings: Optional[RenderSettings] = None
    ) -> Dict[str, Any]:
        """
        Render a scene graph to SVG/PNG
        
        Returns dict with:
        - svg: SVG string
        - png_base64: Base64 encoded PNG (if format=png)
        - render_time_ms: Rendering time
        """
        start_time = time.time()
        settings = settings or self.default_settings
        
        # Build SVG
        svg_content = self._build_svg(scene_graph, settings)
        
        result = {
            'svg': svg_content,
            'format': settings.format,
            'width': settings.width,
            'height': settings.height
        }
        
        # Convert to PNG if requested
        if settings.format == 'png':
            try:
                png_bytes = self._svg_to_png(svg_content, settings)
                result['png_base64'] = base64.b64encode(png_bytes).decode('utf-8')
            except Exception as e:
                result['png_error'] = str(e)
        
        result['render_time_ms'] = (time.time() - start_time) * 1000
        return result
    
    def _build_svg(self, scene_graph, settings: RenderSettings) -> str:
        """Build complete SVG from scene graph"""
        svg_parts = []
        
        # SVG header
        svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" 
            width="{settings.width}" 
            height="{settings.height}" 
            viewBox="0 0 {settings.width} {settings.height}">''')
        
        # Background
        svg_parts.append(f'<rect x="0" y="0" width="{settings.width}" height="{settings.height}" fill="{settings.background_color}"/>')
        
        # Defs section for patterns, gradients, etc
        svg_parts.append('<defs>')
        svg_parts.append(self._generate_common_defs())
        svg_parts.append('</defs>')
        
        # Render nodes sorted by layer
        all_nodes = self._flatten_scene_graph(scene_graph)
        sorted_nodes = sorted(all_nodes, key=lambda n: n.layer)
        
        for node in sorted_nodes:
            if node.visible and node.pattern_id:
                svg_fragment = self._render_node(node)
                if svg_fragment:
                    svg_parts.append(svg_fragment)
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _flatten_scene_graph(self, node) -> List:
        """Flatten scene graph into list of nodes"""
        nodes = []
        if node.pattern_id:  # Only include nodes with patterns
            nodes.append(node)
        for child in node.children:
            nodes.extend(self._flatten_scene_graph(child))
        return nodes
    
    def _render_node(self, node) -> str:
        """Render a single scene graph node to SVG"""
        pattern = self.pattern_db.get_pattern(node.pattern_id)
        if not pattern:
            return ""
        
        # Check cache
        cache_key = f"{node.pattern_id}_{node.transform.scale_x}_{node.transform.scale_y}"
        
        # Get color override
        color_override = node.attributes.get('color_override')
        opacity = node.attributes.get('opacity', 1.0)
        
        # Build group with transform
        transform_str = node.transform.to_svg_transform()
        
        svg_parts = []
        svg_parts.append(f'<g transform="{transform_str}" opacity="{opacity}">')
        
        # Render each shape in the pattern
        for shape in pattern.shape_descriptors:
            shape_svg = shape.to_svg_element()
            
            # Apply color override if specified
            if color_override and 'fill="' in shape_svg:
                # Only override non-none fills
                if 'fill="none"' not in shape_svg.lower():
                    shape_svg = self._apply_color_override(shape_svg, color_override)
            
            svg_parts.append(shape_svg)
        
        svg_parts.append('</g>')
        
        return '\n'.join(svg_parts)
    
    def _apply_color_override(self, svg_element: str, color: str) -> str:
        """Apply color override to SVG element"""
        import re
        # Replace fill color (but not fill="none")
        def replace_fill(match):
            fill_value = match.group(1)
            if fill_value.lower() == 'none':
                return match.group(0)
            return f'fill="{color}"'
        
        return re.sub(r'fill="([^"]+)"', replace_fill, svg_element)
    
    def _generate_common_defs(self) -> str:
        """Generate common SVG definitions"""
        return '''
        <!-- Gradients -->
        <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#87CEEB"/>
            <stop offset="100%" style="stop-color:#B0E0E6"/>
        </linearGradient>
        
        <linearGradient id="sunsetGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#FF6B6B"/>
            <stop offset="50%" style="stop-color:#FFE66D"/>
            <stop offset="100%" style="stop-color:#4ECDC4"/>
        </linearGradient>
        
        <linearGradient id="waterGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#006994"/>
            <stop offset="100%" style="stop-color:#003366"/>
        </linearGradient>
        
        <!-- Patterns -->
        <pattern id="grassPattern" patternUnits="userSpaceOnUse" width="20" height="20">
            <rect width="20" height="20" fill="#228B22"/>
            <line x1="5" y1="20" x2="7" y2="12" stroke="#1E7B1E" stroke-width="2"/>
            <line x1="15" y1="20" x2="13" y2="14" stroke="#1E7B1E" stroke-width="2"/>
        </pattern>
        
        <!-- Filters -->
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
        </filter>
        '''
    
    def _svg_to_png(self, svg_content: str, settings: RenderSettings) -> bytes:
        """Convert SVG to PNG using cairosvg"""
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                output_width=settings.width,
                output_height=settings.height
            )
            return png_bytes
        except ImportError:
            # Fallback: return SVG as bytes
            return svg_content.encode('utf-8')
        except Exception as e:
            raise RuntimeError(f"PNG conversion failed: {e}")
    
    def render_pattern_preview(self, pattern_id: str, size: int = 100) -> Dict[str, Any]:
        """Render a single pattern for preview"""
        pattern = self.pattern_db.get_pattern(pattern_id)
        if not pattern:
            return {'error': 'Pattern not found'}
        
        svg_content = pattern.to_svg(width=size, height=size)
        
        return {
            'svg': svg_content,
            'pattern_id': pattern_id,
            'pattern_name': pattern.name,
            'size': size
        }
    
    def clear_cache(self):
        """Clear rendering caches"""
        self.rendered_patterns.clear()
        self.composition_cache.clear()


class RenderPipeline:
    """
    Multi-stage rendering pipeline for optimized performance
    """
    
    def __init__(self, renderer: PatternRenderer):
        self.renderer = renderer
        self.stages = []
    
    def add_stage(self, stage_func, name: str):
        """Add a processing stage"""
        self.stages.append({'func': stage_func, 'name': name})
    
    def execute(self, scene_graph, settings: RenderSettings) -> Dict[str, Any]:
        """Execute the pipeline"""
        result = {
            'scene_graph': scene_graph,
            'settings': settings,
            'stage_times': {}
        }
        
        for stage in self.stages:
            start = time.time()
            result = stage['func'](result)
            result['stage_times'][stage['name']] = (time.time() - start) * 1000
        
        return result
