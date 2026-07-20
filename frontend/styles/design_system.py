"""
DeepGuard Enterprise AI Platform - Design System

Enterprise-grade design system inspired by Apple, OpenAI, Stripe, and Vercel.
Pixel-perfect implementation for a premium AI cybersecurity platform.
"""

# ============================================================================
# COLOR SYSTEM
# Enterprise-grade color palettes with light/dark modes
# ============================================================================

class ColorPalette:
    """Primary color system for enterprise AI platform"""
    
    # Light Theme (Apple-inspired minimalism)
    LIGHT = {
        # Primary - DeepGuard Blue (Professional, Trustworthy)
        "primary": {
            "50": "#f0f9ff",
            "100": "#e0f2fe",
            "200": "#bae6fd",
            "300": "#7dd3fc",
            "400": "#38bdf8",
            "500": "#0ea5e9",  # Primary
            "600": "#0284c7",
            "700": "#0369a1",
            "800": "#075985",
            "900": "#0c4a6e",
        },
        
        # Secondary - Purple (AI/ML accent)
        "secondary": {
            "50": "#faf5ff",
            "100": "#f3e8ff",
            "200": "#e9d5ff",
            "300": "#d8b4fe",
            "400": "#c084fc",
            "500": "#a855f7",  # Secondary
            "600": "#9333ea",
            "700": "#7e22ce",
            "800": "#6b21a8",
            "900": "#581c87",
        },
        
        # Neutrals (Apple-style grays)
        "neutral": {
            "50": "#fafafa",
            "100": "#f5f5f5",
            "200": "#e5e5e5",
            "300": "#d4d4d4",
            "400": "#a3a3a3",
            "500": "#737373",
            "600": "#525252",
            "700": "#404040",
            "800": "#262626",
            "900": "#171717",
        },
        
        # Semantic Colors
        "success": {
            "50": "#f0fdf4",
            "500": "#22c55e",  # Green
            "600": "#16a34a",
        },
        "warning": {
            "50": "#fefce8",
            "500": "#eab308",  # Yellow
            "600": "#ca8a04",
        },
        "error": {
            "50": "#fef2f2",
            "500": "#ef4444",  # Red
            "600": "#dc2626",
        },
        "info": {
            "50": "#eff6ff",
            "500": "#3b82f6",  # Blue
            "600": "#2563eb",
        },
        
        # Surface Colors
        "background": "#ffffff",
        "surface": "#f8fafc",
        "card": "#ffffff",
        "border": "#e2e8f0",
        "accent": "#0ea5e9",
    }
    
    # Dark Theme (Linear/Stripe-inspired)
    DARK = {
        # Primary - Cyber Blue (Security-focused)
        "primary": {
            "50": "#0a2540",
            "100": "#0d2d4c",
            "200": "#113558",
            "300": "#153d64",
            "400": "#194570",
            "500": "#1d4d7c",  # Primary
            "600": "#215588",
            "700": "#255d94",
            "800": "#2965a0",
            "900": "#2d6dac",
        },
        
        # Secondary - AI Purple
        "secondary": {
            "50": "#1e1b4b",
            "100": "#2e266d",
            "200": "#3e318f",
            "300": "#4e3cb1",
            "400": "#5e47d3",
            "500": "#6e52f5",  # Secondary
            "600": "#7e5df7",
            "700": "#8e68f9",
            "800": "#9e73fb",
            "900": "#ae7efd",
        },
        
        # Neutrals (Dark mode grays)
        "neutral": {
            "50": "#0a0a0a",
            "100": "#171717",
            "200": "#262626",
            "300": "#404040",
            "400": "#525252",
            "500": "#737373",
            "600": "#a3a3a3",
            "700": "#d4d4d4",
            "800": "#e5e5e5",
            "900": "#f5f5f5",
        },
        
        # Semantic Colors
        "success": {
            "50": "#052e16",
            "500": "#22c55e",
            "600": "#16a34a",
        },
        "warning": {
            "50": "#422006",
            "500": "#eab308",
            "600": "#ca8a04",
        },
        "error": {
            "50": "#450a0a",
            "500": "#ef4444",
            "600": "#dc2626",
        },
        "info": {
            "50": "#172554",
            "500": "#3b82f6",
            "600": "#2563eb",
        },
        
        # Surface Colors
        "background": "#0a0a0a",
        "surface": "#171717",
        "card": "#1e1e1e",
        "border": "#2d2d2d",
        "accent": "#3b82f6",
    }


# ============================================================================
# TYPOGRAPHY SYSTEM
# Professional font hierarchy inspired by Apple and Stripe
# ============================================================================

class Typography:
    """Professional typography system for enterprise UI"""
    
    # Font Sizes (px)
    SCALE = {
        "xs": "0.75rem",      # 12px
        "sm": "0.875rem",     # 14px
        "base": "1rem",       # 16px
        "lg": "1.125rem",     # 18px
        "xl": "1.25rem",      # 20px
        "2xl": "1.5rem",      # 24px
        "3xl": "1.875rem",    # 30px
        "4xl": "2.25rem",     # 36px
        "5xl": "3rem",        # 48px
        "6xl": "3.75rem",     # 60px
        "7xl": "4.5rem",      # 72px
    }
    
    # Font Weights
    WEIGHTS = {
        "light": "300",
        "normal": "400",
        "medium": "500",
        "semibold": "600",
        "bold": "700",
        "extrabold": "800",
    }
    
    # Line Heights
    LINE_HEIGHTS = {
        "none": "1",
        "tight": "1.25",
        "snug": "1.375",
        "normal": "1.5",
        "relaxed": "1.625",
        "loose": "2",
    }
    
    # Letter Spacing
    LETTER_SPACING = {
        "tighter": "-0.05em",
        "tight": "-0.025em",
        "normal": "0",
        "wide": "0.025em",
        "wider": "0.05em",
        "widest": "0.1em",
    }


# ============================================================================
# SPACING SYSTEM
# 8-point grid system for pixel-perfect layouts
# ============================================================================

class Spacing:
    """8-point grid spacing system"""
    
    SCALE = {
        "0": "0",
        "px": "1px",
        "0.5": "0.125rem",    # 2px
        "1": "0.25rem",       # 4px
        "1.5": "0.375rem",    # 6px
        "2": "0.5rem",        # 8px
        "2.5": "0.625rem",    # 10px
        "3": "0.75rem",       # 12px
        "3.5": "0.875rem",    # 14px
        "4": "1rem",          # 16px
        "5": "1.25rem",       # 20px
        "6": "1.5rem",        # 24px
        "7": "1.75rem",       # 28px
        "8": "2rem",          # 32px
        "9": "2.25rem",       # 36px
        "10": "2.5rem",       # 40px
        "11": "2.75rem",      # 44px
        "12": "3rem",         # 48px
        "14": "3.5rem",       # 56px
        "16": "4rem",         # 64px
        "20": "5rem",         # 80px
        "24": "6rem",         # 96px
        "28": "7rem",         # 112px
        "32": "8rem",         # 128px
        "36": "9rem",         # 144px
        "40": "10rem",        # 160px
    }


# ============================================================================
# BORDER RADIUS
# Consistent rounded corners inspired by modern SaaS platforms
# ============================================================================

class BorderRadius:
    """Border radius system for soft, modern UI"""
    
    SCALE = {
        "none": "0",
        "sm": "0.125rem",     # 2px
        "base": "0.25rem",    # 4px
        "md": "0.375rem",     # 6px
        "lg": "0.5rem",       # 8px
        "xl": "0.75rem",      # 12px
        "2xl": "1rem",        # 16px
        "3xl": "1.5rem",      # 24px
        "full": "9999px",
    }


# ============================================================================
# SHADOWS & ELEVATION
# Soft shadows for depth and hierarchy
# ============================================================================

class Shadows:
    """Elevation system with soft shadows"""
    
    SCALE = {
        "xs": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        "sm": "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
        "base": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        "md": "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
        "lg": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
        "xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
        "2xl": "0 50px 100px -20px rgba(0, 0, 0, 0.25)",
        
        # Glassmorphism shadows
        "glass": "0 8px 32px 0 rgba(31, 38, 135, 0.37)",
        "frost": "inset 0 0 0 1px rgba(255, 255, 255, 0.1), 0 8px 32px 0 rgba(31, 38, 135, 0.37)",
        
        # Inner shadows
        "inner": "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)",
    }


# ============================================================================
# ANIMATIONS & TRANSITIONS
# Smooth animations for premium feel
# ============================================================================

class Animations:
    """Animation system for micro-interactions"""
    
    DURATIONS = {
        "75": "75ms",
        "100": "100ms",
        "150": "150ms",
        "200": "200ms",
        "300": "300ms",
        "500": "500ms",
        "700": "700ms",
        "1000": "1000ms",
    }
    
    EASING = {
        "linear": "linear",
        "in": "cubic-bezier(0.4, 0, 1, 1)",
        "out": "cubic-bezier(0, 0, 0.2, 1)",
        "in-out": "cubic-bezier(0.4, 0, 0.2, 1)",
        "spring": "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
    }
    
    # Keyframe animations
    KEYFRAMES = {
        "fade-in": """
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
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
        "pulse": """
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        """,
        "spin": """
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        """,
    }


# ============================================================================
# THEME MANAGER
# Handles light/dark mode and theme switching
# ============================================================================

class ThemeManager:
    """Manages application theme and color modes"""
    
    def __init__(self, mode="dark"):
        self.mode = mode
        self.colors = ColorPalette.DARK if mode == "dark" else ColorPalette.LIGHT
        
    def switch_mode(self, new_mode):
        """Switch between light and dark modes"""
        self.mode = new_mode
        self.colors = ColorPalette.DARK if new_mode == "dark" else ColorPalette.LIGHT
        return self
    
    def get_css_variables(self):
        """Generate CSS variables for the current theme"""
        css_vars = []
        
        # Primary colors
        for shade, value in self.colors["primary"].items():
            css_vars.append(f"--color-primary-{shade}: {value};")
        
        # Secondary colors
        for shade, value in self.colors["secondary"].items():
            css_vars.append(f"--color-secondary-{shade}: {value};")
        
        # Neutral colors
        for shade, value in self.colors["neutral"].items():
            css_vars.append(f"--color-neutral-{shade}: {value};")
        
        # Semantic colors
        for semantic in ["success", "warning", "error", "info"]:
            for shade, value in self.colors[semantic].items():
                css_vars.append(f"--color-{semantic}-{shade}: {value};")
        
        # Surface colors
        css_vars.append(f"--color-background: {self.colors['background']};")
        css_vars.append(f"--color-surface: {self.colors['surface']};")
        css_vars.append(f"--color-card: {self.colors['card']};")
        css_vars.append(f"--color-border: {self.colors['border']};")
        css_vars.append(f"--color-accent: {self.colors['accent']};")
        
        return "\n".join(css_vars)


# ============================================================================
# STREAMLIT CSS GENERATOR
# Generates complete CSS for Streamlit with our design system
# ============================================================================

class StreamlitCSSGenerator:
    """Generates comprehensive CSS for Streamlit with our design system"""
    
    @staticmethod
    def generate_css(theme_mode="dark"):
        """Generate complete CSS with design system"""
        theme = ThemeManager(theme_mode)
        
        css = f"""
        <style>
        /* DeepGuard Enterprise AI Platform - Design System CSS */
        /* Theme: {theme_mode.title()} Mode */
        
        /* CSS Variables */
        :root {{
        {theme.get_css_variables()}
        
        /* Typography */
        --font-size-xs: {Typography.SCALE['xs']};
        --font-size-sm: {Typography.SCALE['sm']};
        --font-size-base: {Typography.SCALE['base']};
        --font-size-lg: {Typography.SCALE['lg']};
        --font-size-xl: {Typography.SCALE['xl']};
        --font-size-2xl: {Typography.SCALE['2xl']};
        --font-size-3xl: {Typography.SCALE['3xl']};
        --font-size-4xl: {Typography.SCALE['4xl']};
        --font-size-5xl: {Typography.SCALE['5xl']};
        --font-size-6xl: {Typography.SCALE['6xl']};
        --font-size-7xl: {Typography.SCALE['7xl']};
        
        /* Spacing */
        --spacing-0: {Spacing.SCALE['0']};
        --spacing-1: {Spacing.SCALE['1']};
        --spacing-2: {Spacing.SCALE['2']};
        --spacing-3: {Spacing.SCALE['3']};
        --spacing-4: {Spacing.SCALE['4']};
        --spacing-5: {Spacing.SCALE['5']};
        --spacing-6: {Spacing.SCALE['6']};
        --spacing-8: {Spacing.SCALE['8']};
        --spacing-10: {Spacing.SCALE['10']};
        --spacing-12: {Spacing.SCALE['12']};
        --spacing-16: {Spacing.SCALE['16']};
        --spacing-20: {Spacing.SCALE['20']};
        --spacing-24: {Spacing.SCALE['24']};
        --spacing-32: {Spacing.SCALE['32']};
        --spacing-40: {Spacing.SCALE['40']};
        
        /* Border Radius */
        --radius-none: {BorderRadius.SCALE['none']};
        --radius-sm: {BorderRadius.SCALE['sm']};
        --radius-base: {BorderRadius.SCALE['base']};
        --radius-md: {BorderRadius.SCALE['md']};
        --radius-lg: {BorderRadius.SCALE['lg']};
        --radius-xl: {BorderRadius.SCALE['xl']};
        --radius-2xl: {BorderRadius.SCALE['2xl']};
        --radius-3xl: {BorderRadius.SCALE['3xl']};
        --radius-full: {BorderRadius.SCALE['full']};
        
        /* Shadows */
        --shadow-xs: {Shadows.SCALE['xs']};
        --shadow-sm: {Shadows.SCALE['sm']};
        --shadow-base: {Shadows.SCALE['base']};
        --shadow-md: {Shadows.SCALE['md']};
        --shadow-lg: {Shadows.SCALE['lg']};
        --shadow-xl: {Shadows.SCALE['xl']};
        --shadow-2xl: {Shadows.SCALE['2xl']};
        --shadow-glass: {Shadows.SCALE['glass']};
        --shadow-frost: {Shadows.SCALE['frost']};
        --shadow-inner: {Shadows.SCALE['inner']};
        
        /* Animations */
        --animation-duration-100: {Animations.DURATIONS['100']};
        --animation-duration-300: {Animations.DURATIONS['300']};
        --animation-duration-500: {Animations.DURATIONS['500']};
        --animation-easing-in-out: {Animations.EASING['in-out']};
        --animation-easing-spring: {Animations.EASING['spring']};
        }}
        
        /* Base Styles */
        .stApp {{
            background: var(--color-background);
            color: var(--color-neutral-700);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--color-neutral-900);
            font-weight: 600;
            line-height: 1.25;
            margin-top: 0;
            margin-bottom: var(--spacing-4);
        }}
        
        h1 {{ font-size: var(--font-size-5xl); }}
        h2 {{ font-size: var(--font-size-4xl); }}
        h3 {{ font-size: var(--font-size-3xl); }}
        h4 {{ font-size: var(--font-size-2xl); }}
        h5 {{ font-size: var(--font-size-xl); }}
        h6 {{ font-size: var(--font-size-lg); }}
        
        .text-xs {{ font-size: var(--font-size-xs); }}
        .text-sm {{ font-size: var(--font-size-sm); }}
        .text-base {{ font-size: var(--font-size-base); }}
        .text-lg {{ font-size: var(--font-size-lg); }}
        .text-xl {{ font-size: var(--font-size-xl); }}
        .text-2xl {{ font-size: var(--font-size-2xl); }}
        .text-3xl {{ font-size: var(--font-size-3xl); }}
        .text-4xl {{ font-size: var(--font-size-4xl); }}
        .text-5xl {{ font-size: var(--font-size-5xl); }}
        .text-6xl {{ font-size: var(--font-size-6xl); }}
        
        /* Cards */
        .glass-card {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: var(--radius-2xl);
            padding: var(--spacing-6);
            transition: all 0.3s var(--animation-easing-in-out);
            box-shadow: var(--shadow-frost);
        }}
        
        .glass-card:hover {{
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: var(--shadow-xl);
            transform: translateY(-2px);
        }}
        
        .frost-card {{
            background: linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.05) 0%,
                rgba(255, 255, 255, 0.02) 100%
            );
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-xl);
            padding: var(--spacing-5);
        }}
        
        /* Buttons */
        .btn-primary {{
            background: linear-gradient(135deg, var(--color-primary-500), var(--color-primary-600));
            color: white;
            border: none;
            border-radius: var(--radius-lg);
            padding: var(--spacing-3) var(--spacing-6);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s var(--animation-easing-in-out);
            box-shadow: var(--shadow-sm);
        }}
        
        .btn-primary:hover {{
            background: linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700));
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }}
        
        .btn-secondary {{
            background: rgba(255, 255, 255, 0.05);
            color: var(--color-neutral-300);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: var(--radius-lg);
            padding: var(--spacing-3) var(--spacing-6);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s var(--animation-easing-in-out);
        }}
        
        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }}
        
        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: var(--spacing-1) var(--spacing-2);
            border-radius: var(--radius-full);
            font-size: var(--font-size-xs);
            font-weight: 500;
            line-height: 1;
        }}
        
        .badge-success {{
            background: rgba(34, 197, 94, 0.15);
            color: var(--color-success-500);
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        
        .badge-warning {{
            background: rgba(234, 179, 8, 0.15);
            color: var(--color-warning-500);
            border: 1px solid rgba(234, 179, 8, 0.3);
        }}
        
        .badge-error {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--color-error-500);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        
        .badge-info {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--color-info-500);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        /* Inputs */
        .input-field {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--spacing-3) var(--spacing-4);
            color: var(--color-neutral-300);
            font-size: var(--font-size-base);
            transition: all 0.2s var(--animation-easing-in-out);
        }}
        
        .input-field:focus {{
            outline: none;
            border-color: var(--color-primary-500);
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }}
        
        /* Progress Bars */
        .progress-bar {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-full);
            overflow: hidden;
            height: 8px;
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, var(--color-primary-500), var(--color-secondary-500));
            height: 100%;
            border-radius: var(--radius-full);
            transition: width 0.3s var(--animation-easing-in-out);
        }}
        
        /* Animations */
        {Animations.KEYFRAMES['fade-in']}
        {Animations.KEYFRAMES['slide-up']}
        {Animations.KEYFRAMES['slide-down']}
        {Animations.KEYFRAMES['pulse']}
        {Animations.KEYFRAMES['spin']}
        
        .animate-fade-in {{
            animation: fadeIn var(--animation-duration-300) var(--animation-easing-in-out);
        }}
        
        .animate-slide-up {{
            animation: slideUp var(--animation-duration-300) var(--animation-easing-in-out);
        }}
        
        .animate-pulse {{
            animation: pulse 2s var(--animation-easing-in-out) infinite;
        }}
        
        .animate-spin {{
            animation: spin 1s linear infinite;
        }}
        
        /* Skeleton Loading */
        .skeleton {{
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0.05) 25%,
                rgba(255, 255, 255, 0.1) 50%,
                rgba(255, 255, 255, 0.05) 75%
            );
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }}
        
        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-full);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: var(--radius-full);
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        
        /* Layout Utilities */
        .flex {{ display: flex; }}
        .flex-col {{ flex-direction: column; }}
        .items-center {{ align-items: center; }}
        .justify-center {{ justify-content: center; }}
        .justify-between {{ justify-content: space-between; }}
        .gap-2 {{ gap: var(--spacing-2); }}
        .gap-4 {{ gap: var(--spacing-4); }}
        .gap-6 {{ gap: var(--spacing-6); }}
        .w-full {{ width: 100%; }}
        .h-full {{ height: 100%; }}
        .p-4 {{ padding: var(--spacing-4); }}
        .p-6 {{ padding: var(--spacing-6); }}
        .m-4 {{ margin: var(--spacing-4); }}
        .m-6 {{ margin: var(--spacing-6); }}
        
        /* Override Streamlit Defaults */
        .stButton > button {{
            background: linear-gradient(135deg, var(--color-primary-500), var(--color-primary-600));
            color: white;
            border: none;
            border-radius: var(--radius-lg);
            padding: var(--spacing-3) var(--spacing-6);
            font-weight: 500;
            transition: all 0.2s var(--animation-easing-in-out);
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700));
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }}
        
        .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-lg);
        }}
        
        /* Custom Metrics Display */
        .metric-card {{
            background: linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.05) 0%,
                rgba(255, 255, 255, 0.02) 100%
            );
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-xl);
            padding: var(--spacing-5);
            transition: all 0.3s var(--animation-easing-in-out);
        }}
        
        .metric-card:hover {{
            border-color: var(--color-primary-500);
            box-shadow: var(--shadow-lg);
        }}
        
        .metric-value {{
            font-size: var(--font-size-4xl);
            font-weight: 700;
            background: linear-gradient(135deg, var(--color-primary-400), var(--color-secondary-400));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
            margin: var(--spacing-2) 0;
        }}
        
        .metric-label {{
            font-size: var(--font-size-sm);
            color: var(--color-neutral-400);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        </style>
        """
        
        return css
    
    @staticmethod
    def get_google_fonts():
        """Return Google Fonts link for Inter font"""
        return """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        """


# ============================================================================
# COMPONENT FACTORY
# Pre-built component generators
# ============================================================================

class ComponentFactory:
    """Factory for generating common UI components"""
    
    @staticmethod
    def metric_card(label, value, change=None, icon=None):
        """Generate a metric card component"""
        icon_html = f"<div style='margin-bottom: 8px; font-size: 24px;'>{icon}</div>" if icon else ""
        change_html = f"<div class='text-sm' style='color: var(--color-success-500);'>{change}</div>" if change else ""
        
        return f"""
        <div class='metric-card animate-fade-in'>
            {icon_html}
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            {change_html}
        </div>
        """
    
    @staticmethod
    def status_badge(status, text):
        """Generate a status badge"""
        status_class = {
            "success": "badge-success",
            "warning": "badge-warning",
            "error": "badge-error",
            "info": "badge-info",
        }.get(status, "badge-info")
        
        return f"""
        <span class='badge {status_class}'>{text}</span>
        """
    
    @staticmethod
    def skeleton_card(width="100%", height="120px"):
        """Generate a skeleton loading card"""
        return f"""
        <div class='skeleton' style='width: {width}; height: {height}; border-radius: var(--radius-xl);'></div>
        """


# Export the design system
__all__ = [
    "ColorPalette",
    "Typography", 
    "Spacing",
    "BorderRadius",
    "Shadows",
    "Animations",
    "ThemeManager",
    "StreamlitCSSGenerator",
    "ComponentFactory",
]