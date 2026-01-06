"""
Pattern-Based Image Generation System
Pure pattern matching/composition - NO neural networks
"""

from .visual_patterns import (
    VisualPattern,
    ShapePrimitive,
    ColorDistribution,
    TextureSignature,
    RelationGraph,
    VisualPatternDatabase
)
from .text_to_visual import TextToVisualDecomposer
from .scene_composer import SceneGraphNode, SceneComposer, Constraint
from .renderer import PatternRenderer
from .consciousness_controller import ImageGenerationController
from .optimization import RealTimeOptimizer

__all__ = [
    'VisualPattern',
    'ShapePrimitive',
    'ColorDistribution',
    'TextureSignature',
    'RelationGraph',
    'VisualPatternDatabase',
    'TextToVisualDecomposer',
    'SceneGraphNode',
    'SceneComposer',
    'Constraint',
    'PatternRenderer',
    'ImageGenerationController',
    'RealTimeOptimizer'
]
