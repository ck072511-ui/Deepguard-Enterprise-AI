"""
DeepGuard Enterprise AI Platform - Theme Manager

Manages theme switching between light and dark modes.
Provides CSS generation and theme utilities.
"""

import streamlit as st
from typing import Dict, Any
from .colors import ColorPalette, ColorTokens
from .typography import Typography
from .spacing import Spacing
from .border_radius import BorderRadius
from .shadows import Shadows
from .animations import Animations


class ThemeManager:
    """Manages application theme and provides CSS generation"""
    
    def __init__(self, mode: str = "dark"):
        self.mode = mode
        self.colors = ColorTokens.get_mode(mode)
        self.palette = ColorPalette()
        
    @staticmethod
    def get_theme_from_session() -> str:
        """Get theme mode from Streamlit session state"""
        if "theme_mode" not in st.session_state:
            st.session_state.theme_mode = "dark"
        return st.session_state.theme_mode
    
    @staticmethod
    def set_theme_in_session(mode: str):
        """Set theme mode in Streamlit session state"""
        st.session_state.theme_mode = mode
    
    def switch_mode(self, new_mode: str) -> 'ThemeManager':
        """Switch between light and dark modes"""
        self.mode = new_mode
        self.colors = ColorTokens.get_mode(new_mode)
        return self
    
    def get_css_variables(self) -> str:
        """Generate CSS variables for the current theme"""
        css_vars = []
        
        # Add mode variable
        css_vars.append(f"--color-mode: {self.mode};")
        
        # Add color tokens
        for key, value in self.colors.items():
            css_vars.append(f"--color-{key.replace('_', '-')}: {value};")
        
        # Add color scales
        for shade, value in self.palette.PRIMARY.items():
            css_vars.append(f"--color-primary{shade}: {value};")
        
        for shade, value in self.palette.SECONDARY.items():
            css_vars.append(f"--color-secondary{shade}: {value};")
        
        for semantic_name in ["SUCCESS", "WARNING", "ERROR", "INFO"]:
            scale = getattr(self.palette, semantic_name)
            for shade, value in scale.items():
                css_vars.append(f"--color-{semantic_name.lower()}{shade}: {value};")
        
        for shade, value in self.palette.NEUTRAL.items():
            css_vars.append(f"--color-neutral{shade}: {value};")
        
        # Add gradients
        for name, gradient in self.palette.GRADIENTS.items():
            css_vars.append(f"--gradient-{name}: {gradient};")
        
        # Add typography variables
        typography = Typography()
        css_vars.append(typography.get_css_variables())
        
        # Add spacing variables
        spacing = Spacing()
        css_vars.append(spacing.get_css_variables())
        
        # Add border radius variables
        border_radius = BorderRadius()
        css_vars.append(border_radius.get_css_variables())
        
        # Add shadow variables
        shadows = Shadows()
        css_vars.append(shadows.get_css_variables())
        
        # Add animation variables
        animations = Animations()
        css_vars.append(animations.get_css_variables())
        
        return "\n".join(css_vars)
    
    def generate_css(self) -> str:
        """Generate complete CSS for the current theme"""
        css = f"""
        <style>
        /* DeepGuard Enterprise AI Platform - {self.mode.title()} Theme */
        
        /* CSS Variables */
        :root {{
        {self.get_css_variables()}
        }}
        
        /* Base Styles */
        .stApp {{
            background: var(--color-background);
            color: var(--color-text-primary);
            font-family: var(--font-sans);
            line-height: var(--line-height-normal);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            transition: background-color var(--duration-300) var(--easing-ease-in-out),
                        color var(--duration-300) var(--easing-ease-in-out);
        }}
        
        /* Theme-specific overrides */
        {'/* Dark mode specific styles */' if self.mode == 'dark' else '/* Light mode specific styles */'}
        
        /* Glassmorphism effects */
        .glass-effect {{
            background: {ColorTokens.GLASS['dark' if self.mode == 'dark' else 'light']};
            backdrop-filter: {ColorTokens.GLASS['blur']};
            -webkit-backdrop-filter: {ColorTokens.GLASS['blur']};
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* Gradient text */
        .text-gradient-primary {{
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .text-gradient-ai {{
            background: var(--gradient-ai);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        /* Utility classes */
        .bg-primary-gradient {{
            background: var(--gradient-primary);
        }}
        
        .bg-success-gradient {{
            background: var(--gradient-success);
        }}
        
        .bg-error-gradient {{
            background: var(--gradient-error);
        }}
        
        /* Add keyframe animations */
        {Animations().KEYFRAMES['fade-in']}
        {Animations().KEYFRAMES['slide-up']}
        {Animations().KEYFRAMES['slide-down']}
        {Animations().KEYFRAMES['pulse']}
        {Animations().KEYFRAMES['spin']}
        {Animations().KEYFRAMES['shimmer']}
        {Animations().KEYFRAMES['loading']}
        
        /* Streamlit overrides */
        .stButton > button {{
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: var(--radius-button);
            padding: var(--spacing-3) var(--spacing-6);
            font-weight: var(--font-weight-medium);
            transition: all var(--duration-200) var(--easing-ease-in-out);
            box-shadow: var(--shadow-button-default);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-button-hover);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: var(--shadow-button-active);
        }}
        
        /* Form elements */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {{
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-input);
            color: var(--color-text-primary);
            transition: all var(--duration-200) var(--easing-ease-in-out);
        }}
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {{
            border-color: var(--color-accent);
            box-shadow: var(--shadow-input-focus);
        }}
        
        /* Tables */
        .stTable {{
            background: var(--color-surface);
            border-radius: var(--radius-card);
            border: 1px solid var(--color-border);
        }}
        
        /* Metrics */
        .metric-card {{
            background: var(--color-surface-elevated);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-card);
            padding: var(--spacing-5);
            transition: all var(--duration-300) var(--easing-ease-in-out);
        }}
        
        .metric-card:hover {{
            border-color: var(--color-accent);
            transform: translateY(-2px);
            box-shadow: var(--shadow-card-hover);
        }}
        
        .metric-value {{
            font-size: var(--font-size-4xl);
            font-weight: var(--font-weight-bold);
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
            margin: var(--spacing-2) 0;
        }}
        
        .metric-label {{
            font-size: var(--font-size-sm);
            color: var(--color-text-secondary);
            text-transform: uppercase;
            letter-spacing: var(--letter-spacing-wide);
            font-weight: var(--font-weight-semibold);
        }}
        
        /* Progress bars */
        .stProgress > div > div > div > div {{
            background: var(--gradient-primary);
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--color-surface);
            border-radius: var(--radius-full);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--color-border);
            border-radius: var(--radius-full);
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--color-border-hover);
        }}
        
        </style>
        """
        
        return css
    
    @staticmethod
    def get_google_fonts() -> str:
        """Return Google Fonts imports"""
        return Typography.get_google_fonts()
    
    @staticmethod
    def apply_theme_to_streamlit():
        """Apply theme to Streamlit application"""
        # Get current theme mode
        theme_mode = ThemeManager.get_theme_from_session()
        theme = ThemeManager(theme_mode)
        
        # Apply Google Fonts
        st.markdown(theme.get_google_fonts(), unsafe_allow_html=True)
        
        # Apply CSS
        st.markdown(theme.generate_css(), unsafe_allow_html=True)
        
        # Add theme toggle to sidebar
        with st.sidebar:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🌙" if theme_mode == "light" else "☀️"):
                    new_mode = "light" if theme_mode == "dark" else "dark"
                    ThemeManager.set_theme_in_session(new_mode)
                    st.rerun()
        
        return theme


# Theme context manager
class ThemeContext:
    """Context manager for theme application"""
    
    def __enter__(self):
        self.theme = ThemeManager.apply_theme_to_streamlit()
        return self.theme
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Quick theme application function
def apply_theme(mode: str = None):
    """Quick function to apply theme to Streamlit"""
    if mode:
        ThemeManager.set_theme_in_session(mode)
    return ThemeManager.apply_theme_to_streamlit()