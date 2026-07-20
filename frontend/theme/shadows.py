"""
DeepGuard Enterprise AI Platform - Shadow System

Elevation and shadow system for depth and hierarchy.
Inspired by Material Design, Apple, and Linear design systems.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Shadows:
    """Shadow system for depth and elevation"""
    
    # Shadow scale
    SCALE: Dict[str, str] = {
        # Small shadows
        "xs": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        "sm": "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
        
        # Medium shadows
        "base": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        "md": "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
        
        # Large shadows
        "lg": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
        "xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
        
        # Extra large shadows
        "2xl": "0 50px 100px -20px rgba(0, 0, 0, 0.25)",
        "3xl": "0 80px 150px -30px rgba(0, 0, 0, 0.3)",
        
        # Special effects
        "inner": "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)",
        "glass": "0 8px 32px 0 rgba(31, 38, 135, 0.37)",
        "frost": "inset 0 0 0 1px rgba(255, 255, 255, 0.1), 0 8px 32px 0 rgba(31, 38, 135, 0.37)",
        "glow": "0 0 20px rgba(59, 130, 246, 0.5)",
        "glow-success": "0 0 20px rgba(34, 197, 94, 0.5)",
        "glow-error": "0 0 20px rgba(239, 68, 68, 0.5)",
        "glow-warning": "0 0 20px rgba(234, 179, 8, 0.5)",
    }
    
    # Elevation levels (for consistent depth)
    ELEVATION: Dict[str, Dict[str, str]] = {
        "0": {
            "shadow": "none",
            "z_index": "0",
        },
        "1": {
            "shadow": "xs",
            "z_index": "10",
        },
        "2": {
            "shadow": "sm",
            "z_index": "20",
        },
        "3": {
            "shadow": "base",
            "z_index": "30",
        },
        "4": {
            "shadow": "md",
            "z_index": "40",
        },
        "5": {
            "shadow": "lg",
            "z_index": "50",
        },
        "6": {
            "shadow": "xl",
            "z_index": "60",
        },
        "7": {
            "shadow": "2xl",
            "z_index": "70",
        },
        "8": {
            "shadow": "3xl",
            "z_index": "80",
        },
    }
    
    # Component-specific shadows
    COMPONENTS: Dict[str, Dict[str, str]] = {
        "card": {
            "default": "base",
            "hover": "md",
            "active": "lg",
        },
        "button": {
            "default": "sm",
            "hover": "base",
            "active": "inner",
        },
        "modal": {
            "default": "2xl",
        },
        "dropdown": {
            "default": "xl",
        },
        "tooltip": {
            "default": "lg",
        },
        "toast": {
            "default": "xl",
        },
        "input": {
            "default": "none",
            "focus": "glow",
        },
        "navigation": {
            "default": "sm",
        },
    }
    
    @classmethod
    def get_shadow(cls, size: str) -> str:
        """Get shadow value by size name"""
        return cls.SCALE.get(size, cls.SCALE["none"])
    
    @classmethod
    def get_elevation(cls, level: str) -> Dict[str, str]:
        """Get elevation configuration by level"""
        return cls.ELEVATION.get(level, cls.ELEVATION["0"])
    
    @classmethod
    def get_component_shadow(cls, component: str, state: str = "default") -> str:
        """Get shadow for specific component and state"""
        component_config = cls.COMPONENTS.get(component, {})
        shadow_name = component_config.get(state, "none")
        return cls.get_shadow(shadow_name)
    
    @classmethod
    def get_css_variables(cls) -> str:
        """Generate CSS variables for shadows"""
        css_vars = []
        
        # Add shadow scale
        for name, value in cls.SCALE.items():
            css_vars.append(f"--shadow-{name}: {value};")
        
        # Add elevation variables
        for level, config in cls.ELEVATION.items():
            css_vars.append(f"--elevation-{level}-shadow: var(--shadow-{config['shadow']});")
            css_vars.append(f"--elevation-{level}-z-index: {config['z_index']};")
        
        # Add component shadows
        for component, states in cls.COMPONENTS.items():
            for state, shadow_name in states.items():
                css_vars.append(f"--shadow-{component}-{state}: var(--shadow-{shadow_name});")
        
        return "\n".join(css_vars)
    
    @classmethod
    def get_shadow_css(cls) -> str:
        """Generate shadow utility classes"""
        css = """
        /* Shadow Utilities */
        """
        
        # Shadow utilities
        for name, value in cls.SCALE.items():
            css += f"""
            .shadow-{name} {{ box-shadow: {value}; }}
            """
        
        # Elevation utilities
        for level, config in cls.ELEVATION.items():
            shadow = cls.get_shadow(config["shadow"])
            css += f"""
            .elevation-{level} {{
                box-shadow: {shadow};
                z-index: {config["z_index"]};
            }}
            """
        
        # Component shadow classes
        css += """
        /* Component Shadow Classes */
        .shadow-card { box-shadow: var(--shadow-card-default); }
        .shadow-card-hover { box-shadow: var(--shadow-card-hover); }
        .shadow-card-active { box-shadow: var(--shadow-card-active); }
        
        .shadow-button { box-shadow: var(--shadow-button-default); }
        .shadow-button-hover { box-shadow: var(--shadow-button-hover); }
        .shadow-button-active { box-shadow: var(--shadow-button-active); }
        
        .shadow-modal { box-shadow: var(--shadow-modal-default); }
        .shadow-dropdown { box-shadow: var(--shadow-dropdown-default); }
        .shadow-tooltip { box-shadow: var(--shadow-tooltip-default); }
        .shadow-toast { box-shadow: var(--shadow-toast-default); }
        
        .shadow-input { box-shadow: var(--shadow-input-default); }
        .shadow-input-focus { box-shadow: var(--shadow-input-focus); }
        
        .shadow-navigation { box-shadow: var(--shadow-navigation-default); }
        """
        
        # Special shadow effects
        css += """
        /* Special Shadow Effects */
        .shadow-glass {
            box-shadow: var(--shadow-glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }
        
        .shadow-frost {
            box-shadow: var(--shadow-frost);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        .shadow-glow {
            box-shadow: var(--shadow-glow);
        }
        
        .shadow-glow-success {
            box-shadow: var(--shadow-glow-success);
        }
        
        .shadow-glow-error {
            box-shadow: var(--shadow-glow-error);
        }
        
        .shadow-glow-warning {
            box-shadow: var(--shadow-glow-warning);
        }
        
        /* Hover Effects */
        .hover-shadow-md:hover {
            box-shadow: var(--shadow-md);
        }
        
        .hover-shadow-lg:hover {
            box-shadow: var(--shadow-lg);
        }
        
        .hover-shadow-xl:hover {
            box-shadow: var(--shadow-xl);
        }
        
        .hover-glow:hover {
            box-shadow: var(--shadow-glow);
        }
        
        /* Focus Effects */
        .focus-shadow:focus {
            box-shadow: var(--shadow-glow);
            outline: none;
        }
        
        .focus-shadow-success:focus {
            box-shadow: var(--shadow-glow-success);
            outline: none;
        }
        
        .focus-shadow-error:focus {
            box-shadow: var(--shadow-glow-error);
            outline: none;
        }
        
        /* Active States */
        .active-shadow:active {
            box-shadow: var(--shadow-inner);
        }
        
        /* Transition Effects */
        .shadow-transition {
            transition: box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .shadow-transition-slow {
            transition: box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .shadow-transition-fast {
            transition: box-shadow 0.1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        """
        
        return css


# Shadow utility functions
def apply_shadow(element: str, shadow_type: str = "base") -> str:
    """Apply shadow to an element"""
    shadows = Shadows()
    shadow_value = shadows.get_shadow(shadow_type)
    
    return f"""
    .{element} {{
        box-shadow: {shadow_value};
    }}
    """


def get_shadow_for_elevation(level: int) -> str:
    """Get shadow for specific elevation level"""
    shadows = Shadows()
    elevation = shadows.get_elevation(str(level))
    return shadows.get_shadow(elevation["shadow"])


def create_glow_effect(color: str, intensity: float = 0.5) -> str:
    """Create a glow effect with specified color"""
    return f"0 0 20px rgba({color}, {intensity})"


def apply_component_shadow(component: str, state: str = "default") -> str:
    """Apply component-specific shadow"""
    shadows = Shadows()
    return shadows.get_component_shadow(component, state)