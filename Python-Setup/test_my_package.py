#!/usr/bin/env python3
"""
Simple test script for My Package
"""

from my_package import Calculator, greet, format_number


def test_calculator():
    """Test the Calculator class"""
    print("Testing Calculator...")

    # Test basic operations
    calc = Calculator(10)
    assert calc.get_result() == 10

    calc.add(5)
    assert calc.get_result() == 15

    calc.multiply(2)
    assert calc.get_result() == 30

    calc.subtract(10)
    assert calc.get_result() == 20

    calc.divide(4)
    assert calc.get_result() == 5.0

    # Test chaining
    result = Calculator(0).add(5).multiply(3).subtract(2).get_result()
    assert result == 13

    # Test reset
    calc.reset()
    assert calc.get_result() == 0

    print("✓ Calculator tests passed!")


def test_utils():
    """Test utility functions"""
    print("Testing utilities...")

    # Test greet function
    assert greet() == "Hello, World!"
    assert greet("Alice") == "Hello, Alice!"

    # Test format_number function
    assert format_number(3.14159) == "3.14"
    assert format_number(3.14159, 4) == "3.1416"

    print("✓ Utility tests passed!")


def main():
    """Run all tests"""
    print("Running My Package tests...\n")

    try:
        test_calculator()
        test_utils()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
