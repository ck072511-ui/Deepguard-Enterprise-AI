"""
DeepGuard Enterprise AI Platform - Animation System

Smooth animations and transitions for premium user experience.
Inspired by Apple, Linear, and Framer motion principles.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Animations:
    """Animation system for micro-interactions and transitions"""
    
    # Animation durations
    DURATIONS: Dict[str, str] = {
        "75": "75ms",
        "100": "100ms",
        "150": "150ms",
        "200": "200ms",
        "300": "300ms",
        "500": "500ms",
        "700": "700ms",
        "1000": "1000ms",
    }
    
    # Timing functions
    EASING: Dict[str, str] = {
        "linear": "linear",
        "ease": "ease",
        "ease-in": "ease-in",
        "ease-out": "ease-out",
        "ease-in-out": "ease-in-out",
        "spring": "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
        "bounce": "cubic-bezier(0.68, -0.55, 0.265, 1.55)",
        "material": "cubic-bezier(0.4, 0, 0.2, 1)",
        "sharp": "cubic-bezier(0.4, 0, 0.6, 1)",
    }
    
    # Animation delays
    DELAYS: Dict[str, str] = {
        "75": "75ms",
        "100": "100ms",
        "150": "150ms",
        "200": "200ms",
        "300": "300ms",
        "500": "500ms",
        "700": "700ms",
        "1000": "1000ms",
    }
    
    # Keyframe animations
    KEYFRAMES: Dict[str, str] = {
        "fade-in": """
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        """,
        "fade-out": """
            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }
        """,
        "slide-up": """
            @keyframes slideUp {
                from { 
                    opacity: 0;
                    transform: translateY(10px);
                }
                to { 
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        """,
        "slide-down": """
            @keyframes slideDown {
                from { 
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to { 
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        """,
        "slide-left": """
            @keyframes slideLeft {
                from { 
                    opacity: 0;
                    transform: translateX(10px);
                }
                to { 
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        """,
        "slide-right": """
            @keyframes slideRight {
                from { 
                    opacity: 0;
                    transform: translateX(-10px);
                }
                to { 
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        """,
        "scale-up": """
            @keyframes scaleUp {
                from { 
                    opacity: 0;
                    transform: scale(0.95);
                }
                to { 
                    opacity: 1;
                    transform: scale(1);
                }
            }
        """,
        "scale-down": """
            @keyframes scaleDown {
                from { 
                    opacity: 1;
                    transform: scale(1);
                }
                to { 
                    opacity: 0;
                    transform: scale(0.95);
                }
            }
        """,
        "pulse": """
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        """,
        "pulse-strong": """
            @keyframes pulseStrong {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
        """,
        "spin": """
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        """,
        "bounce": """
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
        """,
        "float": """
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
        """,
        "shimmer": """
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
        """,
        "loading": """
            @keyframes loading {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
        """,
        "progress": """
            @keyframes progress {
                0% { width: 0%; }
                100% { width: 100%; }
            }
        """,
        "ripple": """
            @keyframes ripple {
                0% { transform: scale(0); opacity: 1; }
                100% { transform: scale(4); opacity: 0; }
            }
        """,
    }
    
    # Predefined animation combinations
    PRESETS: Dict[str, Dict[str, str]] = {
        "fade-in-up": {
            "animation": "fadeIn slideUp",
            "duration": "300ms",
            "easing": "ease-out",
        },
        "fade-in-down": {
            "animation": "fadeIn slideDown",
            "duration": "300ms",
            "easing": "ease-out",
        },
        "fade-in-left": {
            "animation": "fadeIn slideLeft",
            "duration": "300ms",
            "easing": "ease-out",
        },
        "fade-in-right": {
            "animation": "fadeIn slideRight",
            "duration": "300ms",
            "easing": "ease-out",
        },
        "scale-in": {
            "animation": "fadeIn scaleUp",
            "duration": "300ms",
            "easing": "spring",
        },
        "modal-enter": {
            "animation": "fadeIn scaleUp",
            "duration": "200ms",
            "easing": "material",
        },
        "modal-exit": {
            "animation": "fadeOut scaleDown",
            "duration": "150ms",
            "easing": "sharp",
        },
        "button-press": {
            "animation": "scaleDown",
            "duration": "100ms",
            "easing": "ease-in",
        },
        "button-release": {
            "animation": "scaleUp",
            "duration": "200ms",
            "easing": "spring",
        },
        "notification": {
            "animation": "fadeIn slideLeft",
            "duration": "300ms",
            "easing": "spring",
        },
        "skeleton": {
            "animation": "shimmer",
            "duration": "2s",
            "easing": "linear",
            "iteration": "infinite",
        },
        "loading-spinner": {
            "animation": "spin",
            "duration": "1s",
            "easing": "linear",
            "iteration": "infinite",
        },
        "progress-bar": {
            "animation": "progress",
            "duration": "2s",
            "easing": "ease-in-out",
        },
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, str]:
        """Get animation preset configuration"""
        return cls.PRESETS.get(preset_name, cls.PRESETS["fade-in-up"])
    
    @classmethod
    def get_css_variables(cls) -> str:
        """Generate CSS variables for animations"""
        css_vars = []
        
        # Add durations
        for name, value in cls.DURATIONS.items():
            css_vars.append(f"--duration-{name}: {value};")
        
        # Add easing functions
        for name, value in cls.EASING.items():
            css_vars.append(f"--easing-{name}: {value};")
        
        # Add delays
        for name, value in cls.DELAYS.items():
            css_vars.append(f"--delay-{name}: {value};")
        
        # Add preset variables
        for preset_name, preset in cls.PRESETS.items():
            css_vars.append(f"--animation-{preset_name}: {preset['animation']};")
            css_vars.append(f"--animation-{preset_name}-duration: {preset['duration']};")
            css_vars.append(f"--animation-{preset_name}-easing: var(--easing-{preset['easing']});")
            if "iteration" in preset:
                css_vars.append(f"--animation-{preset_name}-iteration: {preset['iteration']};")
        
        return "\n".join(css_vars)
    
    @classmethod
    def get_animation_css(cls) -> str:
        """Generate complete animation CSS"""
        css = """
        /* Animation System */
        """
        
        # Add keyframes
        for name, keyframe in cls.KEYFRAMES.items():
            css += keyframe
            css += "\n"
        
        # Animation utility classes
        css += """
        /* Animation Utilities */
        .animate-none { animation: none; }
        .animate-fill-forwards { animation-fill-mode: forwards; }
        .animate-fill-backwards { animation-fill-mode: backwards; }
        .animate-fill-both { animation-fill-mode: both; }
        
        /* Preset Animation Classes */
        """
        
        for preset_name in cls.PRESETS.keys():
            preset = cls.get_preset(preset_name)
            animation_css = f"""
            .animate-{preset_name} {{
                animation-name: {preset['animation'].replace(' ', ', ')};
                animation-duration: {preset['duration']};
                animation-timing-function: {preset['easing']};
            """
            if "iteration" in preset:
                animation_css += f"animation-iteration-count: {preset['iteration']};\n"
            animation_css += "}\n"
            css += animation_css
        
        # Duration utilities
        for name, value in cls.DURATIONS.items():
            css += f"""
            .duration-{name} {{ animation-duration: {value}; }}
            """
        
        # Easing utilities
        for name, value in cls.EASING.items():
            css += f"""
            .easing-{name} {{ animation-timing-function: {value}; }}
            """
        
        # Delay utilities
        for name, value in cls.DELAYS.items():
            css += f"""
            .delay-{name} {{ animation-delay: {value}; }}
            """
        
        # Common animation classes
        css += """
        /* Common Animation Classes */
        .animate-fade-in {
            animation: fadeIn var(--duration-300) var(--easing-ease-out);
        }
        
        .animate-fade-out {
            animation: fadeOut var(--duration-300) var(--easing-ease-out);
        }
        
        .animate-slide-up {
            animation: slideUp var(--duration-300) var(--easing-ease-out);
        }
        
        .animate-slide-down {
            animation: slideDown var(--duration-300) var(--easing-ease-out);
        }
        
        .animate-pulse {
            animation: pulse 2s var(--easing-ease-in-out) infinite;
        }
        
        .animate-pulse-strong {
            animation: pulseStrong 2s var(--easing-ease-in-out) infinite;
        }
        
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        
        .animate-bounce {
            animation: bounce 1s var(--easing-bounce) infinite;
        }
        
        .animate-float {
            animation: float 3s var(--easing-ease-in-out) infinite;
        }
        
        .animate-shimmer {
            animation: shimmer 2s linear infinite;
        }
        
        .animate-loading {
            animation: loading 1.5s linear infinite;
        }
        
        .animate-progress {
            animation: progress 2s var(--easing-ease-in-out);
        }
        
        .animate-ripple {
            animation: ripple 0.6s linear;
        }
        
        /* Transition Utilities */
        .transition-none { transition: none; }
        .transition-all { transition: all var(--duration-300) var(--easing-ease-in-out); }
        .transition-colors { transition: color var(--duration-300) var(--easing-ease-in-out), 
                                     background-color var(--duration-300) var(--easing-ease-in-out), 
                                     border-color var(--duration-300) var(--easing-ease-in-out); }
        .transition-transform { transition: transform var(--duration-300) var(--easing-ease-in-out); }
        .transition-opacity { transition: opacity var(--duration-300) var(--easing-ease-in-out); }
        .transition-shadow { transition: box-shadow var(--duration-300) var(--easing-ease-in-out); }
        
        .transition-fast { transition-duration: var(--duration-150); }
        .transition-normal { transition-duration: var(--duration-300); }
        .transition-slow { transition-duration: var(--duration-500); }
        
        /* Hover Effects */
        .hover-lift:hover {
            transform: translateY(-2px);
            transition: transform var(--duration-200) var(--easing-spring);
        }
        
        .hover-scale:hover {
            transform: scale(1.05);
            transition: transform var(--duration-200) var(--easing-spring);
        }
        
        .hover-glow:hover {
            box-shadow: var(--shadow-glow);
            transition: box-shadow var(--duration-300) var(--easing-ease-in-out);
        }
        
        /* Loading States */
        .loading {
            position: relative;
            overflow: hidden;
        }
        
        .loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.1),
                transparent
            );
            animation: shimmer 2s linear infinite;
        }
        
        /* Skeleton Loading */
        .skeleton {
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0.05) 25%,
                rgba(255, 255, 255, 0.1) 50%,
                rgba(255, 255, 255, 0.05) 75%
            );
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }
        
        /* Progress Indicator */
        .progress-indicator {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .progress-indicator-fill {
            height: 100%;
            background: var(--gradient-primary);
            animation: progress 2s var(--easing-ease-in-out) infinite;
        }
        
        /* Spinner */
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: var(--color-primary-500);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .spinner-lg {
            width: 48px;
            height: 48px;
            border-width: 4px;
        }
        
        .spinner-sm {
            width: 16px;
            height: 16px;
            border-width: 2px;
        }
        """
        
        return css


# Animation utility functions
def apply_animation(element: str, animation_preset: str = "fade-in-up") -> str:
    """Apply animation to an element"""
    animations = Animations()
    preset = animations.get_preset(animation_preset)
    
    css = f"""
    .{element} {{
        animation-name: {preset['animation'].replace(' ', ', ')};
        animation-duration: {preset['duration']};
        animation-timing-function: {preset['easing']};
    """
    
    if "iteration" in preset:
        css += f"animation-iteration-count: {preset['iteration']};\n"
    
    css += "}"
    
    return css


def create_stagger_animation(base_class: str, count: int = 5, delay_step: int = 100) -> str:
    """Create staggered animation for multiple elements"""
    css = ""
    
    for i in range(count):
        delay = i * delay_step
        css += f"""
        .{base_class}:nth-child({i + 1}) {{
            animation-delay: {delay}ms;
        }}
        """
    
    return css


def get_animation_delay(index: int, step: int = 100) -> str:
    """Get animation delay for staggered animations"""
    return f"{index * step}ms"