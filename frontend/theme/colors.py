"""
DeepGuard Enterprise AI Platform - Color System

Complete color palette with light and dark modes.
Inspired by premium SaaS platforms like OpenAI, Stripe, and Vercel.
"""

from dataclasses import dataclass
from typing import Dict, TypedDict


class ColorScale(TypedDict):
    """Color scale with 9 levels (50-900)"""
    _50: str
    _100: str
    _200: str
    _300: str
    _400: str
    _500: str
    _600: str
    _700: str
    _800: str
    _900: str


class SemanticColors(TypedDict):
    """Semantic color tokens"""
    text_primary: str
    text_secondary: str
    text_tertiary: str
    background: str
    surface: str
    surface_elevated: str
    border: str
    border_hover: str
    accent: str
    overlay: str


@dataclass
class ColorPalette:
    """Complete color palette with light and dark modes"""
    
    # Primary Colors - DeepGuard Security Blue (Trust, Professionalism)
    PRIMARY: ColorScale = {
        "_50": "#eff6ff",
        "_100": "#dbeafe",
        "_200": "#bfdbfe",
        "_300": "#93c5fd",
        "_400": "#60a5fa",
        "_500": "#3b82f6",  # Primary
        "_600": "#2563eb",
        "_700": "#1d4ed8",
        "_800": "#1e40af",
        "_900": "#1e3a8a",
    }
    
    # Secondary Colors - AI Purple (Innovation, Intelligence)
    SECONDARY: ColorScale = {
        "_50": "#faf5ff",
        "_100": "#f3e8ff",
        "_200": "#e9d5ff",
        "_300": "#d8b4fe",
        "_400": "#c084fc",
        "_500": "#a855f7",  # Secondary
        "_600": "#9333ea",
        "_700": "#7e22ce",
        "_800": "#6b21a8",
        "_900": "#581c87",
    }
    
    # Success Colors - Security Green (Trust, Safety)
    SUCCESS: ColorScale = {
        "_50": "#f0fdf4",
        "_100": "#dcfce7",
        "_200": "#bbf7d0",
        "_300": "#86efac",
        "_400": "#4ade80",
        "_500": "#22c55e",  # Success
        "_600": "#16a34a",
        "_700": "#15803d",
        "_800": "#166534",
        "_900": "#14532d",
    }
    
    # Warning Colors - Alert Yellow (Caution, Attention)
    WARNING: ColorScale = {
        "_50": "#fefce8",
        "_100": "#fef9c3",
        "_200": "#fef08a",
        "_300": "#fde047",
        "_400": "#facc15",
        "_500": "#eab308",  # Warning
        "_600": "#ca8a04",
        "_700": "#a16207",
        "_800": "#854d0e",
        "_900": "#713f12",
    }
    
    # Error Colors - Security Red (Danger, Threat)
    ERROR: ColorScale = {
        "_50": "#fef2f2",
        "_100": "#fee2e2",
        "_200": "#fecaca",
        "_300": "#fca5a5",
        "_400": "#f87171",
        "_500": "#ef4444",  # Error
        "_600": "#dc2626",
        "_700": "#b91c1c",
        "_800": "#991b1b",
        "_900": "#7f1d1d",
    }
    
    # Info Colors - System Blue (Information, Status)
    INFO: ColorScale = {
        "_50": "#eff6ff",
        "_100": "#dbeafe",
        "_200": "#bfdbfe",
        "_300": "#93c5fd",
        "_400": "#60a5fa",
        "_500": "#3b82f6",  # Info
        "_600": "#2563eb",
        "_700": "#1d4ed8",
        "_800": "#1e40af",
        "_900": "#1e3a8a",
    }
    
    # Neutrals - Professional Grays
    NEUTRAL: ColorScale = {
        "_50": "#fafafa",
        "_100": "#f5f5f5",
        "_200": "#e5e5e5",
        "_300": "#d4d4d4",
        "_400": "#a3a3a3",
        "_500": "#737373",
        "_600": "#525252",
        "_700": "#404040",
        "_800": "#262626",
        "_900": "#171717",
    }
    
    # Special Gradients for premium UI
    GRADIENTS = {
        "primary": "linear-gradient(135deg, #3b82f6 0%, #a855f7 100%)",
        "success": "linear-gradient(135deg, #22c55e 0%, #3b82f6 50%)",
        "warning": "linear-gradient(135deg, #eab308 0%, #f97316 100%)",
        "error": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
        "premium": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)",
        "ai": "linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%)",
    }


class ColorTokens:
    """Semantic color tokens for light and dark modes"""
    
    LIGHT: SemanticColors = {
        "text_primary": "#171717",  # neutral-900
        "text_secondary": "#525252",  # neutral-600
        "text_tertiary": "#a3a3a3",  # neutral-400
        "background": "#ffffff",  # white
        "surface": "#fafafa",  # neutral-50
        "surface_elevated": "#ffffff",  # white
        "border": "#e5e5e5",  # neutral-200
        "border_hover": "#d4d4d4",  # neutral-300
        "accent": "#3b82f6",  # primary-500
        "overlay": "rgba(0, 0, 0, 0.4)",  # overlay
    }
    
    DARK: SemanticColors = {
        "text_primary": "#f5f5f5",  # neutral-100
        "text_secondary": "#a3a3a3",  # neutral-400
        "text_tertiary": "#737373",  # neutral-500
        "background": "#0a0a0a",  # neutral-950
        "surface": "#171717",  # neutral-900
        "surface_elevated": "#262626",  # neutral-800
        "border": "#404040",  # neutral-700
        "border_hover": "#525252",  # neutral-600
        "accent": "#60a5fa",  # primary-400
        "overlay": "rgba(0, 0, 0, 0.6)",  # overlay
    }
    
    # Glassmorphism effects
    GLASS = {
        "light": "rgba(255, 255, 255, 0.8)",
        "dark": "rgba(30, 30, 30, 0.8)",
        "blur": "blur(20px)",
    }
    
    @classmethod
    def get_mode(cls, mode: str = "dark") -> SemanticColors:
        """Get color tokens for specified mode"""
        return cls.DARK if mode == "dark" else cls.LIGHT
    
    @classmethod
    def get_css_variables(cls, mode: str = "dark") -> str:
        """Generate CSS variables for the color system"""
        colors = cls.get_mode(mode)
        palette = ColorPalette()
        
        css_vars = []
        
        # Add mode variable
        css_vars.append(f"--color-mode: {'dark' if mode == 'dark' else 'light'};")
        
        # Add semantic tokens
        for key, value in colors.items():
            css_vars.append(f"--color-{key.replace('_', '-')}: {value};")
        
        # Add primary scale
        for shade, value in palette.PRIMARY.items():
            css_vars.append(f"--color-primary{shade}: {value};")
        
        # Add secondary scale
        for shade, value in palette.SECONDARY.items():
            css_vars.append(f"--color-secondary{shade}: {value};")
        
        # Add semantic scales
        for semantic_name in ["success", "warning", "error", "info"]:
            scale = getattr(palette, semantic_name.upper())
            for shade, value in scale.items():
                css_vars.append(f"--color-{semantic_name}{shade}: {value};")
        
        # Add neutral scale
        for shade, value in palette.NEUTRAL.items():
            css_vars.append(f"--color-neutral{shade}: {value};")
        
        # Add gradients
        for name, gradient in palette.GRADIENTS.items():
            css_vars.append(f"--gradient-{name}: {gradient};")
        
        # Add glass effects
        for name, value in cls.GLASS.items():
            css_vars.append(f"--glass-{name}: {value};")
        
        return "\n".join(css_vars)


# Color utility functions
def get_color(scale: str, shade: str = "_500", mode: str = "dark") -> str:
    """Get a specific color from a scale"""
    palette = ColorPalette()
    
    scale_map = {
        "primary": palette.PRIMARY,
        "secondary": palette.SECONDARY,
        "success": palette.SUCCESS,
        "warning": palette.WARNING,
        "error": palette.ERROR,
        "info": palette.INFO,
        "neutral": palette.NEUTRAL,
    }
    
    return scale_map.get(scale, palette.NEUTRAL).get(shade, "#000000")


def get_gradient(name: str = "primary") -> str:
    """Get a gradient by name"""
    return ColorPalette.GRADIENTS.get(name, ColorPalette.GRADIENTS["primary"])


def get_text_color(bg_color: str, mode: str = "dark") -> str:
    """Get appropriate text color for a background color"""
    # Simple luminance calculation
    import re
    
    # Extract RGB values
    hex_color = bg_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Calculate relative luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return "#ffffff" if luminance < 0.5 else "#000000"