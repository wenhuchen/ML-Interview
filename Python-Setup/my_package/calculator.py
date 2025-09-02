"""
Calculator module providing basic arithmetic operations
"""


class Calculator:
    """A simple calculator class"""

    def __init__(self, initial_value=0):
        self.value = initial_value

    def add(self, x):
        """Add a number to the current value"""
        self.value += x
        return self

    def subtract(self, x):
        """Subtract a number from the current value"""
        self.value -= x
        return self

    def multiply(self, x):
        """Multiply the current value by a number"""
        self.value *= x
        return self

    def divide(self, x):
        """Divide the current value by a number"""
        if x == 0:
            raise ValueError("Cannot divide by zero")
        self.value /= x
        return self

    def get_result(self):
        """Get the current calculated value"""
        return self.value

    def reset(self):
        """Reset the calculator to initial value"""
        self.value = 0
        return self
