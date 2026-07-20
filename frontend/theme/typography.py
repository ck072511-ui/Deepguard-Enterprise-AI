"""
DeepGuard Enterprise AI Platform - Typography System

Professional typography system inspired by Apple, Stripe, and Linear.
Complete font hierarchy for enterprise SaaS applications.
"""

from dataclasses import dataclass
from typing import Dict, TypedDict


class FontScale(TypedDict):
    """Typography scale with size, weight, line-height, and letter-spacing"""
    font_size: str
    font_weight: str
    line_height: str
    letter_spacing: str


@dataclass
class Typography:
    """Complete typography system for enterprise UI"""
    
    # Font Families
    FONT_FAMILIES = {
        "sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "mono": "'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace",
        "system": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    }
    
    # Font Sizes (rem units for scalability)
    FONT_SIZES: Dict[str, str] = {
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
        "8xl": "6rem",        # 96px
        "9xl": "8rem",        # 128px
    }
    
    # Font Weights
    FONT_WEIGHTS: Dict[str, str] = {
        "thin": "100",
        "extralight": "200",
        "light": "300",
        "normal": "400",
        "medium": "500",
        "semibold": "600",
        "bold": "700",
        "extrabold": "800",
        "black": "900",
    }
    
    # Line Heights
    LINE_HEIGHTS: Dict[str, str] = {
        "none": "1",
        "tight": "1.25",
        "snug": "1.375",
        "normal": "1.5",
        "relaxed": "1.625",
        "loose": "2",
    }
    
    # Letter Spacing
    LETTER_SPACING: Dict[str, str] = {
        "tighter": "-0.05em",
        "tight": "-0.025em",
        "normal": "0em",
        "wide": "0.025em",
        "wider": "0.05em",
        "widest": "0.1em",
    }
    
    # Typography Scale - Predefined text styles
    TEXT_STYLES: Dict[str, FontScale] = {
        # Display Text
        "display_2xl": {
            "font_size": FONT_SIZES["7xl"],
            "font_weight": FONT_WEIGHTS["bold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        "display_xl": {
            "font_size": FONT_SIZES["6xl"],
            "font_weight": FONT_WEIGHTS["bold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        "display_lg": {
            "font_size": FONT_SIZES["5xl"],
            "font_weight": FONT_WEIGHTS["bold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        "display_md": {
            "font_size": FONT_SIZES["4xl"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["snug"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        "display_sm": {
            "font_size": FONT_SIZES["3xl"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["snug"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        "display_xs": {
            "font_size": FONT_SIZES["2xl"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["snug"],
            "letter_spacing": LETTER_SPACING["tight"],
        },
        
        # Text Styles
        "text_xl": {
            "font_size": FONT_SIZES["xl"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["relaxed"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "text_lg": {
            "font_size": FONT_SIZES["lg"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["relaxed"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "text_md": {
            "font_size": FONT_SIZES["base"],
            "font_weight": FONT_WEIGHTS["medium"],
            "line_height": LINE_HEIGHTS["relaxed"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "text_sm": {
            "font_size": FONT_SIZES["sm"],
            "font_weight": FONT_WEIGHTS["normal"],
            "line_height": LINE_HEIGHTS["relaxed"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "text_xs": {
            "font_size": FONT_SIZES["xs"],
            "font_weight": FONT_WEIGHTS["normal"],
            "line_height": LINE_HEIGHTS["relaxed"],
            "letter_spacing": LETTER_SPACING["wide"],
        },
        
        # UI Components
        "label_lg": {
            "font_size": FONT_SIZES["sm"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["wide"],
        },
        "label_md": {
            "font_size": FONT_SIZES["xs"],
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["wide"],
        },
        "label_sm": {
            "font_size": "0.625rem",  # 10px
            "font_weight": FONT_WEIGHTS["semibold"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["widest"],
        },
        
        # Monospace (for code, metrics)
        "mono_lg": {
            "font_size": FONT_SIZES["lg"],
            "font_weight": FONT_WEIGHTS["medium"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "mono_md": {
            "font_size": FONT_SIZES["base"],
            "font_weight": FONT_WEIGHTS["medium"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
        "mono_sm": {
            "font_size": FONT_SIZES["sm"],
            "font_weight": FONT_WEIGHTS["medium"],
            "line_height": LINE_HEIGHTS["tight"],
            "letter_spacing": LETTER_SPACING["normal"],
        },
    }
    
    @classmethod
    def get_style(cls, style_name: str) -> FontScale:
        """Get a predefined text style"""
        return cls.TEXT_STYLES.get(style_name, cls.TEXT_STYLES["text_md"])
    
    @classmethod
    def get_css_variables(cls) -> str:
        """Generate CSS variables for typography"""
        css_vars = []
        
        # Font families
        for name, value in cls.FONT_FAMILIES.items():
            css_vars.append(f"--font-{name}: {value};")
        
        # Font sizes
        for name, value in cls.FONT_SIZES.items():
            css_vars.append(f"--font-size-{name}: {value};")
        
        # Font weights
        for name, value in cls.FONT_WEIGHTS.items():
            css_vars.append(f"--font-weight-{name}: {value};")
        
        # Line heights
        for name, value in cls.LINE_HEIGHTS.items():
            css_vars.append(f"--line-height-{name}: {value};")
        
        # Letter spacing
        for name, value in cls.LETTER_SPACING.items():
            css_vars.append(f"--letter-spacing-{name}: {value};")
        
        # Text styles
        for style_name, style in cls.TEXT_STYLES.items():
            css_vars.append(f"--text-{style_name.replace('_', '-')}-font-size: {style['font_size']};")
            css_vars.append(f"--text-{style_name.replace('_', '-')}-font-weight: {style['font_weight']};")
            css_vars.append(f"--text-{style_name.replace('_', '-')}-line-height: {style['line_height']};")
            css_vars.append(f"--text-{style_name.replace('_', '-')}-letter-spacing: {style['letter_spacing']};")
        
        return "\n".join(css_vars)
    
    @classmethod
    def get_google_fonts(cls) -> str:
        """Return Google Fonts imports"""
        return """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        """


# Typography utility functions
def apply_text_style(style_name: str, element_class: str = "") -> str:
    """Generate CSS for a text style"""
    typography = Typography()
    style = typography.get_style(style_name)
    
    css = f"""
    .text-{style_name.replace('_', '-')} {{
        font-size: {style['font_size']};
        font-weight: {style['font_weight']};
        line-height: {style['line_height']};
        letter-spacing: {style['letter_spacing']};
    }}
    """
    
    if element_class:
        return css.replace(f".text-{style_name.replace('_', '-')}", f".{element_class}")
    
    return css


def get_typography_css() -> str:
    """Generate complete typography CSS"""
    typography = Typography()
    
    css = """
    /* Typography System */
    """
    
    # Add Google Fonts import
    css += typography.get_google_fonts()
    
    # Add CSS variables
    css += ":root {\n"
    css += typography.get_css_variables()
    css += "\n}\n\n"
    
    # Add text style classes
    for style_name in typography.TEXT_STYLES.keys():
        css += apply_text_style(style_name)
        css += "\n"
    
    # Add utility classes
    css += """
    /* Utility Classes */
    .font-sans { font-family: var(--font-sans); }
    .font-mono { font-family: var(--font-mono); }
    .font-system { font-family: var(--font-system); }
    
    .text-center { text-align: center; }
    .text-left { text-align: left; }
    .text-right { text-align: right; }
    .text-justify { text-align: justify; }
    
    .truncate {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .text-gradient {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .text-gradient-success {
        background: var(--gradient-success);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .text-gradient-error {
        background: var(--gradient-error);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    """
    
    return css