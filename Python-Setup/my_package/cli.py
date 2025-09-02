"""
Command-line interface for My Package
"""

import argparse
from .calculator import Calculator
from .utils import greet, format_number


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="My Package CLI")
    parser.add_argument("--greet", "-g", help="Name to greet")
    parser.add_argument("--calculate", "-c", help="Simple calculation (e.g., '2+3*4')")
    parser.add_argument("--format", "-f", type=float, help="Number to format")

    args = parser.parse_args()

    if args.greet:
        print(greet(args.greet))

    elif args.calculate:
        try:
            # Simple evaluation for demonstration
            result = eval(args.calculate)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error calculating: {e}")

    elif args.format is not None:
        print(f"Formatted: {format_number(args.format)}")

    else:
        # Interactive calculator mode
        print("Interactive Calculator Mode")
        print("Enter 'quit' to exit")

        calc = Calculator()
        while True:
            try:
                user_input = input("Enter calculation (e.g., +5, *2, /3): ").strip()

                if user_input.lower() == "quit":
                    break

                if user_input.startswith("+"):
                    value = float(user_input[1:])
                    calc.add(value)
                elif user_input.startswith("-"):
                    value = float(user_input[1:])
                    calc.subtract(value)
                elif user_input.startswith("*"):
                    value = float(user_input[1:])
                    calc.multiply(value)
                elif user_input.startswith("/"):
                    value = float(user_input[1:])
                    calc.divide(value)
                elif user_input.lower() == "reset":
                    calc.reset()
                elif user_input.lower() == "result":
                    print(f"Current result: {calc.get_result()}")
                    continue
                else:
                    print("Invalid input. Use +, -, *, /, reset, or result")
                    continue

                print(f"Current value: {calc.get_result()}")

            except (ValueError, KeyboardInterrupt):
                print("Invalid input or interrupted")
                break


if __name__ == "__main__":
    main()
