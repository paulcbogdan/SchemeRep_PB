def print_colored_list(numbers, vmin=None, vmax=None):
    """
    Print a list of numbers with a smooth color spectrum based on their value.

    Parameters:
    - numbers: List of floats to print
    - vmin: Minimum value for color scaling (default: min of the list)
    - vmax: Maximum value for color scaling (default: max of the list)
    """
    # If vmin or vmax are not specified, use list min and max
    if vmin is None:
        vmin = min(numbers)
    if vmax is None:
        vmax = max(numbers)

    # Normalize the numbers
    normalized = [(num - vmin) / (vmax - vmin) if vmax != vmin else 0 for num in numbers]

    out_str = ''
    # Print the colored numbers
    for num, norm_value in zip(numbers, normalized):
        # Ensure norm_value is between 0 and 1
        norm_value = max(0, min(1, norm_value))
        norm_value = 1 - norm_value  # Invert the color spectrum

        # Create a color spectrum from green (low) to yellow to red (high)
        if norm_value < 0.5:
            # Green to Yellow
            r = int(norm_value * 2 * 255)
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int((1 - (norm_value - 0.5) * 2) * 255)
            b = 0

        # Using ANSI color codes that are more widely supported
        color_code = f'\033[38;5;{_rgb_to_256(r, g, b)}m'

        out_str += f"{color_code}{num:.2f}, "

    #     print(f"{color_code}{num:.2f}", end=", ")
    #
    # # Reset color
    # print('\033[0m')
    out_str = out_str[:-2]
    out_str += '\033[0m'
    return out_str



def _rgb_to_256(r, g, b):
    """
    Convert RGB values to the nearest 256-color palette value.

    Args:
    r, g, b: Integer values between 0 and 255

    Returns:
    Closest color in the 256-color palette
    """
    # Clamp values to 0-255 range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # 6x6x6 color cube
    r = int(r / 51)
    g = int(g / 51)
    b = int(b / 51)

    return 16 + (r * 36) + (g * 6) + b


# Example usage
if __name__ == "__main__":
    # Sample list of numbers
    sample_list = [1.23, 3.45, 2.34, 4.56, 5.67, 0.12, 6.78, 7.89, 8.90, 9.01, 10.12, 11.23]

    # Print with default min and max
    print("Default coloring:")
    print_colored_list(sample_list)

    # Print with specified min and max
    print("\nCustom range coloring:")
    print_colored_list(sample_list, vmin=2, vmax=8)