"""
Animation System - Real-time animations leveraging fast pattern generation
Supports frame-by-frame generation, transitions, and animated SVG export
"""

import math
import time
import uuid
import base64
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class EasingType(Enum):
    """Animation easing functions"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"
    SPRING = "spring"


class AnimationType(Enum):
    """Types of animations"""
    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"
    OPACITY = "opacity"
    COLOR = "color"
    MORPH = "morph"
    PULSE = "pulse"
    WAVE = "wave"
    FLOAT = "float"
    SPIN = "spin"
    SHAKE = "shake"
    GLOW = "glow"


@dataclass
class Keyframe:
    """A single keyframe in an animation"""
    time: float  # 0-1 normalized time
    value: Any  # Value at this keyframe
    easing: EasingType = EasingType.EASE_IN_OUT
    
    def to_dict(self) -> Dict:
        return {
            'time': self.time,
            'value': self.value,
            'easing': self.easing.value
        }


@dataclass
class AnimationTrack:
    """Animation track for a single property"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_id: str = ""  # Element to animate
    property_name: str = ""  # Property to animate (x, y, rotation, opacity, etc.)
    animation_type: AnimationType = AnimationType.TRANSLATE
    keyframes: List[Keyframe] = field(default_factory=list)
    loop: bool = False
    loop_count: int = -1  # -1 = infinite
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'target_id': self.target_id,
            'property_name': self.property_name,
            'animation_type': self.animation_type.value,
            'keyframes': [k.to_dict() for k in self.keyframes],
            'loop': self.loop,
            'loop_count': self.loop_count
        }


@dataclass
class AnimationPreset:
    """Pre-built animation preset"""
    name: str
    description: str
    duration_ms: int
    tracks: List[AnimationTrack]
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'duration_ms': self.duration_ms,
            'tracks': [t.to_dict() for t in self.tracks]
        }


@dataclass
class AnimatedScene:
    """A complete animated scene"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    width: int = 800
    height: int = 600
    duration_ms: int = 3000
    fps: int = 30
    tracks: List[AnimationTrack] = field(default_factory=list)
    elements: Dict[str, Dict] = field(default_factory=dict)  # element_id -> element data
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'duration_ms': self.duration_ms,
            'fps': self.fps,
            'frame_count': self.get_frame_count(),
            'tracks': [t.to_dict() for t in self.tracks],
            'elements': self.elements
        }
    
    def get_frame_count(self) -> int:
        return int(self.duration_ms * self.fps / 1000)


class AnimationSystem:
    """
    Real-time animation system leveraging fast pattern generation
    Supports SVG animations, frame generation, and transitions
    """
    
    # Pre-built animation presets
    PRESETS = {
        'pulse': AnimationPreset(
            name='pulse',
            description='Pulsing scale animation',
            duration_ms=1000,
            tracks=[]
        ),
        'float': AnimationPreset(
            name='float',
            description='Gentle floating motion',
            duration_ms=2000,
            tracks=[]
        ),
        'spin': AnimationPreset(
            name='spin',
            description='Continuous rotation',
            duration_ms=2000,
            tracks=[]
        ),
        'fade_in': AnimationPreset(
            name='fade_in',
            description='Fade in from transparent',
            duration_ms=500,
            tracks=[]
        ),
        'shake': AnimationPreset(
            name='shake',
            description='Shake/vibrate effect',
            duration_ms=500,
            tracks=[]
        ),
        'bounce': AnimationPreset(
            name='bounce',
            description='Bouncing motion',
            duration_ms=1000,
            tracks=[]
        ),
        'wave': AnimationPreset(
            name='wave',
            description='Wave-like motion',
            duration_ms=2000,
            tracks=[]
        ),
        'glow': AnimationPreset(
            name='glow',
            description='Pulsing glow effect',
            duration_ms=1500,
            tracks=[]
        )
    }
    
    def __init__(self, pattern_db=None):
        self.pattern_db = pattern_db
        self._initialize_presets()
    
    def _initialize_presets(self):
        """Initialize animation presets with actual tracks"""
        # Pulse preset
        self.PRESETS['pulse'].tracks = [
            AnimationTrack(
                property_name='scale',
                animation_type=AnimationType.SCALE,
                keyframes=[
                    Keyframe(0.0, 1.0, EasingType.EASE_IN_OUT),
                    Keyframe(0.5, 1.15, EasingType.EASE_IN_OUT),
                    Keyframe(1.0, 1.0, EasingType.EASE_IN_OUT)
                ],
                loop=True
            )
        ]
        
        # Float preset
        self.PRESETS['float'].tracks = [
            AnimationTrack(
                property_name='y',
                animation_type=AnimationType.TRANSLATE,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.EASE_IN_OUT),
                    Keyframe(0.5, -15, EasingType.EASE_IN_OUT),
                    Keyframe(1.0, 0, EasingType.EASE_IN_OUT)
                ],
                loop=True
            )
        ]
        
        # Spin preset
        self.PRESETS['spin'].tracks = [
            AnimationTrack(
                property_name='rotation',
                animation_type=AnimationType.ROTATE,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.LINEAR),
                    Keyframe(1.0, 360, EasingType.LINEAR)
                ],
                loop=True
            )
        ]
        
        # Fade in preset
        self.PRESETS['fade_in'].tracks = [
            AnimationTrack(
                property_name='opacity',
                animation_type=AnimationType.OPACITY,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.EASE_OUT),
                    Keyframe(1.0, 1.0, EasingType.EASE_OUT)
                ],
                loop=False
            )
        ]
        
        # Shake preset
        self.PRESETS['shake'].tracks = [
            AnimationTrack(
                property_name='x',
                animation_type=AnimationType.SHAKE,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.LINEAR),
                    Keyframe(0.1, -5, EasingType.LINEAR),
                    Keyframe(0.2, 5, EasingType.LINEAR),
                    Keyframe(0.3, -5, EasingType.LINEAR),
                    Keyframe(0.4, 5, EasingType.LINEAR),
                    Keyframe(0.5, -3, EasingType.LINEAR),
                    Keyframe(0.6, 3, EasingType.LINEAR),
                    Keyframe(0.7, -2, EasingType.LINEAR),
                    Keyframe(0.8, 2, EasingType.LINEAR),
                    Keyframe(1.0, 0, EasingType.EASE_OUT)
                ],
                loop=False
            )
        ]
        
        # Bounce preset
        self.PRESETS['bounce'].tracks = [
            AnimationTrack(
                property_name='y',
                animation_type=AnimationType.TRANSLATE,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.EASE_IN),
                    Keyframe(0.4, -30, EasingType.EASE_OUT),
                    Keyframe(0.6, 0, EasingType.BOUNCE),
                    Keyframe(0.75, -10, EasingType.EASE_OUT),
                    Keyframe(0.85, 0, EasingType.EASE_IN),
                    Keyframe(0.92, -3, EasingType.EASE_OUT),
                    Keyframe(1.0, 0, EasingType.EASE_IN)
                ],
                loop=True
            )
        ]
        
        # Wave preset
        self.PRESETS['wave'].tracks = [
            AnimationTrack(
                property_name='y',
                animation_type=AnimationType.WAVE,
                keyframes=[
                    Keyframe(0.0, 0, EasingType.EASE_IN_OUT),
                    Keyframe(0.25, -10, EasingType.EASE_IN_OUT),
                    Keyframe(0.5, 0, EasingType.EASE_IN_OUT),
                    Keyframe(0.75, 10, EasingType.EASE_IN_OUT),
                    Keyframe(1.0, 0, EasingType.EASE_IN_OUT)
                ],
                loop=True
            )
        ]
        
        # Glow preset
        self.PRESETS['glow'].tracks = [
            AnimationTrack(
                property_name='glow_intensity',
                animation_type=AnimationType.GLOW,
                keyframes=[
                    Keyframe(0.0, 0.5, EasingType.EASE_IN_OUT),
                    Keyframe(0.5, 1.0, EasingType.EASE_IN_OUT),
                    Keyframe(1.0, 0.5, EasingType.EASE_IN_OUT)
                ],
                loop=True
            )
        ]
    
    def create_animated_scene(self, name: str, width: int = 800, 
                              height: int = 600, duration_ms: int = 3000,
                              fps: int = 30) -> AnimatedScene:
        """Create a new animated scene"""
        return AnimatedScene(
            name=name,
            width=width,
            height=height,
            duration_ms=duration_ms,
            fps=fps
        )
    
    def add_element(self, scene: AnimatedScene, element_id: str,
                    pattern_id: str, x: float, y: float,
                    width: float = 100, height: float = 100) -> None:
        """Add an element to the animated scene"""
        scene.elements[element_id] = {
            'pattern_id': pattern_id,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'rotation': 0,
            'scale': 1.0,
            'opacity': 1.0
        }
    
    def apply_preset(self, scene: AnimatedScene, element_id: str, 
                     preset_name: str) -> AnimationTrack:
        """Apply an animation preset to an element"""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = self.PRESETS[preset_name]
        
        # Clone tracks for this element
        for track_template in preset.tracks:
            track = AnimationTrack(
                target_id=element_id,
                property_name=track_template.property_name,
                animation_type=track_template.animation_type,
                keyframes=track_template.keyframes.copy(),
                loop=track_template.loop,
                loop_count=track_template.loop_count
            )
            scene.tracks.append(track)
        
        return scene.tracks[-1] if scene.tracks else None
    
    def add_custom_animation(self, scene: AnimatedScene, element_id: str,
                             property_name: str, keyframes: List[Tuple[float, Any]],
                             loop: bool = False) -> AnimationTrack:
        """Add a custom animation to an element"""
        track = AnimationTrack(
            target_id=element_id,
            property_name=property_name,
            keyframes=[Keyframe(time=t, value=v) for t, v in keyframes],
            loop=loop
        )
        scene.tracks.append(track)
        return track
    
    def interpolate_value(self, keyframes: List[Keyframe], time: float, 
                          easing: EasingType = EasingType.LINEAR) -> Any:
        """Interpolate value at given time using keyframes"""
        if not keyframes:
            return 0
        
        # Clamp time to 0-1
        time = max(0, min(1, time))
        
        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]
        
        for i, kf in enumerate(keyframes):
            if kf.time <= time:
                prev_kf = kf
            if kf.time >= time:
                next_kf = kf
                break
        
        # Calculate local time between keyframes
        if prev_kf.time == next_kf.time:
            return prev_kf.value
        
        local_t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
        
        # Apply easing
        eased_t = self._apply_easing(local_t, prev_kf.easing)
        
        # Interpolate values
        if isinstance(prev_kf.value, (int, float)):
            return prev_kf.value + (next_kf.value - prev_kf.value) * eased_t
        elif isinstance(prev_kf.value, str):  # Color interpolation
            return self._interpolate_color(prev_kf.value, next_kf.value, eased_t)
        else:
            return prev_kf.value
    
    def _apply_easing(self, t: float, easing: EasingType) -> float:
        """Apply easing function to normalized time"""
        if easing == EasingType.LINEAR:
            return t
        elif easing == EasingType.EASE_IN:
            return t * t
        elif easing == EasingType.EASE_OUT:
            return 1 - (1 - t) * (1 - t)
        elif easing == EasingType.EASE_IN_OUT:
            return t * t * (3 - 2 * t)
        elif easing == EasingType.BOUNCE:
            if t < 0.5:
                return 8 * t * t * t * t
            else:
                return 1 - 8 * (1 - t) ** 4
        elif easing == EasingType.ELASTIC:
            if t == 0 or t == 1:
                return t
            return math.pow(2, -10 * t) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1
        elif easing == EasingType.SPRING:
            return 1 - math.cos(t * 4.5 * math.pi) * math.exp(-t * 6)
        return t
    
    def _interpolate_color(self, color1: str, color2: str, t: float) -> str:
        """Interpolate between two hex colors"""
        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        
        r = int(rgb1[0] + (rgb2[0] - rgb1[0]) * t)
        g = int(rgb1[1] + (rgb2[1] - rgb1[1]) * t)
        b = int(rgb1[2] + (rgb2[2] - rgb1[2]) * t)
        
        return rgb_to_hex((r, g, b))
    
    def render_frame(self, scene: AnimatedScene, frame_number: int) -> Dict[str, Any]:
        """Render a single frame of the animation"""
        total_frames = scene.get_frame_count()
        normalized_time = frame_number / total_frames if total_frames > 0 else 0
        
        # Calculate element states at this time
        frame_state = {}
        
        for element_id, element in scene.elements.items():
            state = element.copy()
            
            # Apply animations
            for track in scene.tracks:
                if track.target_id == element_id:
                    # Handle looping
                    track_time = normalized_time
                    if track.loop:
                        track_time = normalized_time % 1.0
                    
                    value = self.interpolate_value(track.keyframes, track_time)
                    
                    # Apply value based on property
                    if track.property_name == 'x':
                        state['x'] = element['x'] + value
                    elif track.property_name == 'y':
                        state['y'] = element['y'] + value
                    elif track.property_name == 'rotation':
                        state['rotation'] = value
                    elif track.property_name == 'scale':
                        state['scale'] = value
                    elif track.property_name == 'opacity':
                        state['opacity'] = value
            
            frame_state[element_id] = state
        
        return {
            'frame_number': frame_number,
            'normalized_time': normalized_time,
            'elements': frame_state
        }
    
    def export_animated_svg(self, scene: AnimatedScene) -> str:
        """Export scene as animated SVG with CSS animations"""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}">'
        ]
        
        # Add styles for animations
        svg_parts.append('<style>')
        
        # Generate CSS keyframes for each track
        for track in scene.tracks:
            animation_name = f"anim_{track.target_id}_{track.property_name}"
            duration_s = scene.duration_ms / 1000
            
            css_keyframes = [f"@keyframes {animation_name} {{"]
            for kf in track.keyframes:
                percentage = int(kf.time * 100)
                
                if track.property_name == 'x':
                    css_keyframes.append(f"  {percentage}% {{ transform: translateX({kf.value}px); }}")
                elif track.property_name == 'y':
                    css_keyframes.append(f"  {percentage}% {{ transform: translateY({kf.value}px); }}")
                elif track.property_name == 'rotation':
                    css_keyframes.append(f"  {percentage}% {{ transform: rotate({kf.value}deg); }}")
                elif track.property_name == 'scale':
                    css_keyframes.append(f"  {percentage}% {{ transform: scale({kf.value}); }}")
                elif track.property_name == 'opacity':
                    css_keyframes.append(f"  {percentage}% {{ opacity: {kf.value}; }}")
            
            css_keyframes.append("}")
            svg_parts.append('\n'.join(css_keyframes))
            
            # Apply animation to element
            iteration = "infinite" if track.loop else "1"
            svg_parts.append(f"#{track.target_id} {{ animation: {animation_name} {duration_s}s ease-in-out {iteration}; }}")
        
        svg_parts.append('</style>')
        
        # Add elements
        for element_id, element in scene.elements.items():
            x, y = element['x'], element['y']
            w, h = element['width'], element['height']
            
            # Get pattern if available
            if self.pattern_db:
                pattern = self.pattern_db.get_pattern(element['pattern_id'])
                if pattern:
                    svg_parts.append(f'<g id="{element_id}" transform="translate({x},{y})">')
                    for shape in pattern.shape_descriptors:
                        svg_parts.append(shape.to_svg_element())
                    svg_parts.append('</g>')
                    continue
            
            # Fallback placeholder
            svg_parts.append(f'<rect id="{element_id}" x="{x}" y="{y}" width="{w}" height="{h}" fill="#3b82f6" rx="5"/>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def generate_frames(self, scene: AnimatedScene, 
                        frame_callback: Optional[Callable] = None) -> List[Dict]:
        """Generate all frames for an animation"""
        frames = []
        total_frames = scene.get_frame_count()
        
        start_time = time.time()
        
        for frame_num in range(total_frames):
            frame = self.render_frame(scene, frame_num)
            frames.append(frame)
            
            if frame_callback:
                frame_callback(frame_num, total_frames, frame)
        
        generation_time = (time.time() - start_time) * 1000
        avg_frame_time = generation_time / total_frames if total_frames > 0 else 0
        
        return {
            'frames': frames,
            'total_frames': total_frames,
            'duration_ms': scene.duration_ms,
            'fps': scene.fps,
            'generation_time_ms': generation_time,
            'avg_frame_time_ms': avg_frame_time
        }
    
    def get_available_presets(self) -> Dict[str, Dict]:
        """Get list of available animation presets"""
        return {
            name: preset.to_dict() 
            for name, preset in self.PRESETS.items()
        }
