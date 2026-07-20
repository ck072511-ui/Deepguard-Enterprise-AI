"""
DeepGuard Enterprise AI Platform - Border Radius System

Consistent border radius system for modern, rounded UI elements.
Inspired by Apple, Linear, and Stripe design systems.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BorderRadius:
    """Border radius system for soft, modern UI"""
    
    # Border radius scale
    SCALE: Dict[str, str] = {
        "none": "0",
        "sm": "0.125rem",     # 2px
        "base": "0.25rem",    # 4px
        "md": "0.375rem",     # 6px
        "lg": "0.5rem",       # 8px
        "xl": "0.75rem",      # 12px
        "2xl": "1rem",        # 16px
        "3xl": "1.5rem",      # 24px
        "4xl": "2rem",        # 32px
        "full": "9999px",     # Fully rounded
    }
    
    # Component-specific radii
    COMPONENTS: Dict[str, str] = {
        "button": "lg",           # 8px
        "card": "xl",             # 12px
        "input": "lg",            # 8px
        "modal": "2xl",           # 16px
        "badge": "full",          # Fully rounded
        "avatar": "full",         # Fully rounded
        "tooltip": "md",          # 6px
        "alert": "lg",            # 8px
        "toast": "lg",            # 8px
        "dropdown": "lg",         # 8px
    }
    
    @classmethod
    def get_radius(cls, size: str) -> str:
        """Get border radius value by size name"""
        return cls.SCALE.get(size, cls.SCALE["lg"])
    
    @classmethod
    def get_component_radius(cls, component: str) -> str:
        """Get border radius for specific component"""
        size_name = cls.COMPONENTS.get(component, "lg")
        return cls.get_radius(size_name)
    
    @classmethod
    def get_css_variables(cls) -> str:
        """Generate CSS variables for border radius"""
        css_vars = []
        
        # Add radius scale
        for name, value in cls.SCALE.items():
            css_vars.append(f"--radius-{name}: {value};")
        
        # Add component radii
        for component, size in cls.COMPONENTS.items():
            css_vars.append(f"--radius-{component}: var(--radius-{size});")
        
        return "\n".join(css_vars)
    
    @classmethod
    def get_radius_css(cls) -> str:
        """Generate border radius utility classes"""
        css = """
        /* Border Radius Utilities */
        """
        
        # Radius utilities
        for name, value in cls.SCALE.items():
            css += f"""
            .rounded-{name} {{ border-radius: {value}; }}
            .rounded-t-{name} {{ border-top-left-radius: {value}; border-top-right-radius: {value}; }}
            .rounded-r-{name} {{ border-top-right-radius: {value}; border-bottom-right-radius: {value}; }}
            .rounded-b-{name} {{ border-bottom-right-radius: {value}; border-bottom-left-radius: {value}; }}
            .rounded-l-{name} {{ border-top-left-radius: {value}; border-bottom-left-radius: {value}; }}
            .rounded-tl-{name} {{ border-top-left-radius: {value}; }}
            .rounded-tr-{name} {{ border-top-right-radius: {value}; }}
            .rounded-br-{name} {{ border-bottom-right-radius: {value}; }}
            .rounded-bl-{name} {{ border-bottom-left-radius: {value}; }}
            """
        
        # Component classes
        css += """
        /* Component Radius Classes */
        .rounded-button { border-radius: var(--radius-button); }
        .rounded-card { border-radius: var(--radius-card); }
        .rounded-input { border-radius: var(--radius-input); }
        .rounded-modal { border-radius: var(--radius-modal); }
        .rounded-badge { border-radius: var(--radius-badge); }
        .rounded-avatar { border-radius: var(--radius-avatar); }
        .rounded-tooltip { border-radius: var(--radius-tooltip); }
        .rounded-alert { border-radius: var(--radius-alert); }
        .rounded-toast { border-radius: var(--radius-toast); }
        .rounded-dropdown { border-radius: var(--radius-dropdown); }
        """
        
        # Special rounded corners
        css += """
        /* Special Rounded Corners */
        .rounded-pill {
            border-radius: var(--radius-full);
        }
        
        .rounded-glass {
            border-radius: var(--radius-2xl);
            overflow: hidden;
        }
        
        .rounded-frost {
            border-radius: var(--radius-xl);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        /* Border Styles */
        .border-solid { border-style: solid; }
        .border-dashed { border-style: dashed; }
        .border-dotted { border-style: dotted; }
        .border-none { border-style: none; }
        
        .border-0 { border-width: 0; }
        .border { border-width: 1px; }
        .border-2 { border-width: 2px; }
        .border-4 { border-width: 4px; }
        .border-8 { border-width: 8px; }
        
        /* Outline Styles */
        .outline-none { outline: 2px solid transparent; outline-offset: 2px; }
        .outline { outline-style: solid; }
        .outline-dashed { outline-style: dashed; }
        .outline-dotted { outline-style: dotted; }
        
        .outline-0 { outline-width: 0; }
        .outline-1 { outline-width: 1px; }
        .outline-2 { outline-width: 2px; }
        .outline-4 { outline-width: 4px; }
        
        /* Ring Styles (for focus states) */
        .ring-0 { --ring-width: 0px; }
        .ring-1 { --ring-width: 1px; }
        .ring-2 { --ring-width: 2px; }
        .ring-4 { --ring-width: 4px; }
        .ring-8 { --ring-width: 8px; }
        
        .ring-primary { --ring-color: var(--color-primary-500); }
        .ring-secondary { --ring-color: var(--color-secondary-500); }
        .ring-success { --ring-color: var(--color-success-500); }
        .ring-error { --ring-color: var(--color-error-500); }
        
        .focus-ring:focus {
            outline: none;
            box-shadow: 0 0 0 var(--ring-width) var(--ring-color);
        }
        """
        
        return css


# Border radius utility functions
def apply_rounded_corners(element: str, corners: Dict[str, str] = None) -> str:
    """Apply rounded corners to an element"""
    if corners is None:
        corners = {"all": "lg"}
    
    radius = BorderRadius()
    css = ""
    
    if "all" in corners:
        value = radius.get_radius(corners["all"])
        css += f"border-radius: {value};"
    else:
        for corner, size in corners.items():
            value = radius.get_radius(size)
            if corner == "top-left":
                css += f"border-top-left-radius: {value};"
            elif corner == "top-right":
                css += f"border-top-right-radius: {value};"
            elif corner == "bottom-right":
                css += f"border-bottom-right-radius: {value};"
            elif corner == "bottom-left":
                css += f"border-bottom-left-radius: {value};"
    
    return css


def get_border_radius_px(radius_name: str) -> int:
    """Get border radius in pixels"""
    radius = BorderRadius()
    value = radius.get_radius(radius_name)
    
    # Convert rem to px (assuming 1rem = 16px)
    try:
        rem = float(value.replace("rem", "").strip())
        return int(rem * 16)
    except:
        return 8  # Default 8px