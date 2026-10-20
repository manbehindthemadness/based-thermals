"""
Palette menu snapshots shared by App colors and portable visualization controls.
"""

from ..palettes import PREFIX, load_palettes, palette_token, read_palette


def palette_options(
    builtins: tuple[tuple[str, str], ...], current: str,
) -> tuple[tuple[str, str], ...]:
    """
    Add saved palettes and retain embedded selections after library deletion or import.
    """
    choices = list(builtins)
    choices.extend((palette_token(value), f"Custom · {name}")
                   for name, value in sorted(load_palettes().items(), key=lambda item: item[0].casefold()))
    if current.startswith(PREFIX) and current not in dict(choices):
        choices.append((current, f"Embedded · {read_palette(current)['name']}"))
    return tuple(choices)
