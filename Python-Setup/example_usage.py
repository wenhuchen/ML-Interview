#!/usr/bin/env python3
"""
Example usage of My Package
"""

from my_package import Calculator, greet, format_number


def main():
    """Demonstrate package functionality"""
    print("=== My Package Example ===\n")
    
    # Example 1: Using the Calculator
    print("1. Calculator Example:")
    calc = Calculator(100)
    print(f"Starting with: {calc.get_result()}")
    
    result = calc.add(50).multiply(2).subtract(25).divide(5).get_result()
    print(f"After operations: {result}")
    print()
    
    # Example 2: Using utility functions
    print("2. Utility Functions:")
    print(greet("Python Developer"))
    print(f"Formatted number: {format_number(3.14159265359, 6)}")
    print()
    
    # Example 3: Interactive calculator
    print("3. Interactive Calculator:")
    calc2 = Calculator(0)
    operations = [
        ("+10", "Add 10"),
        ("*3", "Multiply by 3"),
        ("-5", "Subtract 5"),
        ("/2", "Divide by 2")
    ]
    
    for operation, description in operations:
        if operation.startswith('+'):
            calc2.add(float(operation[1:]))
        elif operation.startswith('-'):
            calc2.subtract(float(operation[1:]))
        elif operation.startswith('*'):
            calc2.multiply(float(operation[1:]))
        elif operation.startswith('/'):
            calc2.divide(float(operation[1:]))
        
        print(f"{description}: {calc2.get_result()}")
    
    print(f"Final result: {calc2.get_result()}")


if __name__ == "__main__":
    main()
