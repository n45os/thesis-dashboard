import matplotlib.colors as mcolors

strategy_colors = {
    "fed_avg": "#ff33da",
    "multikrum": "#ffaa33",
    "krum": "#bbff33",
    "fed_median": "#33ff70",
    "bulyan": "#33fff8",
    "trimmed_mean": "#5832ff",
    "dnc": "#FF3357",
    "cc": "#33b4ff"
}

def get_color(config):
    
    strategy_name = config.get('strategy', {}).get('name', 'Unknown')
    advers_fraction = config.get('adversaries_fraction', 0)
    base_color = strategy_colors.get(strategy_name, '#000000')  # Default to black if strategy not found

    # Adjust color brightness based on adversary fraction (darker for higher fractions)
    def adjust_color_brightness(base_color, adversary_fraction):
        """Adjusts the color brightness based on adversary fraction."""
        base_color_rgb = mcolors.hex2color(base_color)
        if adversary_fraction == 0:
            return base_color
        # darkened_rgb = [max(0, c * (1 - adversary_fraction)) for c in base_color_rgb]
        adjustment_factor = ((adversary_fraction - 0.1) / 0.75)
        new_rgb = [c + (1 - c) * adjustment_factor for c in base_color_rgb]
        new_hex = mcolors.to_hex(new_rgb)
        return new_hex
    
    adjusted_color = adjust_color_brightness(base_color, advers_fraction)
    return adjusted_color