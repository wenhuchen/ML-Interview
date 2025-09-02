"""
Utility functions for the package
"""


def greet(name="World"):
    """
    Return a greeting message

    Args:
        name (str): Name to greet, defaults to "World"

    Returns:
        str: Greeting message
    """
    return f"Hello, {name}!"


def format_number(number, decimal_places=2):
    """
    Format a number to a specified number of decimal places

    Args:
        number (float): Number to format
        decimal_places (int): Number of decimal places, defaults to 2

    Returns:
        str: Formatted number string
    """
    return f"{number:.{decimal_places}f}"
