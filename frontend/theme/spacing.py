"""
DeepGuard Enterprise AI Platform - Spacing System

8-point grid spacing system for pixel-perfect layouts.
Inspired by design systems from Apple, Stripe, and Linear.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Spacing:
    """8-point grid spacing system"""
    
    # Base spacing unit (8px)
    BASE_UNIT = "0.5rem"  # 8px
    
    # Spacing scale (multiples of 8px)
    SCALE: Dict[str, str] = {
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
        "44": "11rem",        # 176px
        "48": "12rem",        # 192px
        "52": "13rem",        # 208px
        "56": "14rem",        # 224px
        "60": "15rem",        # 240px
        "64": "16rem",        # 256px
        "72": "18rem",        # 288px
        "80": "20rem",        # 320px
        "96": "24rem",        # 384px
    }
    
    # Container widths
    CONTAINERS: Dict[str, str] = {
        "xs": "20rem",    # 320px
        "sm": "24rem",    # 384px
        "md": "28rem",    # 448px
        "lg": "32rem",    # 512px
        "xl": "36rem",    # 576px
        "2xl": "42rem",   # 672px
        "3xl": "48rem",   # 768px
        "4xl": "56rem",   # 896px
        "5xl": "64rem",   # 1024px
        "6xl": "72rem",   # 1152px
        "7xl": "80rem",   # 1280px
        "full": "100%",
    }
    
    # Aspect ratios
    ASPECT_RATIOS: Dict[str, str] = {
        "square": "1 / 1",
        "video": "16 / 9",
        "cinema": "21 / 9",
        "portrait": "9 / 16",
        "widescreen": "4 / 3",
        "ultrawide": "32 / 9",
    }
    
    @classmethod
    def get_spacing(cls, size: str) -> str:
        """Get spacing value by size name"""
        return cls.SCALE.get(size, cls.SCALE["4"])
    
    @classmethod
    def get_container(cls, size: str) -> str:
        """Get container width by size name"""
        return cls.CONTAINERS.get(size, cls.CONTAINERS["full"])
    
    @classmethod
    def get_css_variables(cls) -> str:
        """Generate CSS variables for spacing"""
        css_vars = []
        
        # Add base unit
        css_vars.append(f"--spacing-base: {cls.BASE_UNIT};")
        
        # Add spacing scale
        for name, value in cls.SCALE.items():
            css_vars.append(f"--spacing-{name}: {value};")
        
        # Add container widths
        for name, value in cls.CONTAINERS.items():
            css_vars.append(f"--container-{name}: {value};")
        
        # Add aspect ratios
        for name, value in cls.ASPECT_RATIOS.items():
            css_vars.append(f"--aspect-{name}: {value};")
        
        return "\n".join(css_vars)
    
    @classmethod
    def get_spacing_css(cls) -> str:
        """Generate spacing utility classes"""
        css = """
        /* Spacing Utilities */
        """
        
        # Margin utilities
        for name, value in cls.SCALE.items():
            css += f"""
            .m-{name} {{ margin: {value}; }}
            .mx-{name} {{ margin-left: {value}; margin-right: {value}; }}
            .my-{name} {{ margin-top: {value}; margin-bottom: {value}; }}
            .mt-{name} {{ margin-top: {value}; }}
            .mr-{name} {{ margin-right: {value}; }}
            .mb-{name} {{ margin-bottom: {value}; }}
            .ml-{name} {{ margin-left: {value}; }}
            """
        
        # Padding utilities
        for name, value in cls.SCALE.items():
            css += f"""
            .p-{name} {{ padding: {value}; }}
            .px-{name} {{ padding-left: {value}; padding-right: {value}; }}
            .py-{name} {{ padding-top: {value}; padding-bottom: {value}; }}
            .pt-{name} {{ padding-top: {value}; }}
            .pr-{name} {{ padding-right: {value}; }}
            .pb-{name} {{ padding-bottom: {value}; }}
            .pl-{name} {{ padding-left: {value}; }}
            """
        
        # Gap utilities
        for name, value in cls.SCALE.items():
            css += f"""
            .gap-{name} {{ gap: {value}; }}
            .gap-x-{name} {{ column-gap: {value}; }}
            .gap-y-{name} {{ row-gap: {value}; }}
            """
        
        # Container utilities
        for name, value in cls.CONTAINERS.items():
            css += f"""
            .container-{name} {{ max-width: {value}; }}
            .w-container-{name} {{ width: {value}; }}
            """
        
        # Aspect ratio utilities
        for name, value in cls.ASPECT_RATIOS.items():
            css += f"""
            .aspect-{name} {{ aspect-ratio: {value}; }}
            """
        
        # Layout utilities
        css += """
        /* Layout Utilities */
        .w-full { width: 100%; }
        .w-screen { width: 100vw; }
        .h-full { height: 100%; }
        .h-screen { height: 100vh; }
        .min-h-screen { min-height: 100vh; }
        .max-w-full { max-width: 100%; }
        .max-h-full { max-height: 100%; }
        
        .overflow-hidden { overflow: hidden; }
        .overflow-auto { overflow: auto; }
        .overflow-x-auto { overflow-x: auto; }
        .overflow-y-auto { overflow-y: auto; }
        
        .position-relative { position: relative; }
        .position-absolute { position: absolute; }
        .position-fixed { position: fixed; }
        .position-sticky { position: sticky; }
        
        .top-0 { top: 0; }
        .right-0 { right: 0; }
        .bottom-0 { bottom: 0; }
        .left-0 { left: 0; }
        
        .z-0 { z-index: 0; }
        .z-10 { z-index: 10; }
        .z-20 { z-index: 20; }
        .z-30 { z-index: 30; }
        .z-40 { z-index: 40; }
        .z-50 { z-index: 50; }
        .z-auto { z-index: auto; }
        
        /* Flex Utilities */
        .flex { display: flex; }
        .inline-flex { display: inline-flex; }
        .flex-row { flex-direction: row; }
        .flex-col { flex-direction: column; }
        .flex-wrap { flex-wrap: wrap; }
        .flex-nowrap { flex-wrap: nowrap; }
        .flex-1 { flex: 1 1 0%; }
        .flex-auto { flex: 1 1 auto; }
        .flex-none { flex: none; }
        
        .justify-start { justify-content: flex-start; }
        .justify-end { justify-content: flex-end; }
        .justify-center { justify-content: center; }
        .justify-between { justify-content: space-between; }
        .justify-around { justify-content: space-around; }
        .justify-evenly { justify-content: space-evenly; }
        
        .items-start { align-items: flex-start; }
        .items-end { align-items: flex-end; }
        .items-center { align-items: center; }
        .items-baseline { align-items: baseline; }
        .items-stretch { align-items: stretch; }
        
        /* Grid Utilities */
        .grid { display: grid; }
        .grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
        .grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .grid-cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
        .grid-cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
        .grid-cols-12 { grid-template-columns: repeat(12, minmax(0, 1fr)); }
        
        .col-span-1 { grid-column: span 1 / span 1; }
        .col-span-2 { grid-column: span 2 / span 2; }
        .col-span-3 { grid-column: span 3 / span 3; }
        .col-span-4 { grid-column: span 4 / span 4; }
        .col-span-5 { grid-column: span 5 / span 5; }
        .col-span-6 { grid-column: span 6 / span 6; }
        .col-span-12 { grid-column: span 12 / span 12; }
        
        .grid-rows-1 { grid-template-rows: repeat(1, minmax(0, 1fr)); }
        .grid-rows-2 { grid-template-rows: repeat(2, minmax(0, 1fr)); }
        .grid-rows-3 { grid-template-rows: repeat(3, minmax(0, 1fr)); }
        
        .row-span-1 { grid-row: span 1 / span 1; }
        .row-span-2 { grid-row: span 2 / span 2; }
        .row-span-3 { grid-row: span 3 / span 3; }
        """
        
        return css


# Spacing utility functions
def rem_to_px(rem_value: str) -> int:
    """Convert rem value to pixels (assuming 1rem = 16px)"""
    try:
        rem = float(rem_value.replace("rem", "").strip())
        return int(rem * 16)
    except:
        return 0


def px_to_rem(px_value: int) -> str:
    """Convert pixels to rem (assuming 1rem = 16px)"""
    return f"{px_value / 16}rem"


def get_spacing_map() -> Dict[str, int]:
    """Get spacing map with pixel values"""
    spacing = Spacing()
    spacing_map = {}
    
    for name, value in spacing.SCALE.items():
        spacing_map[name] = rem_to_px(value)
    
    return spacing_map