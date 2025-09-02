# My Package

A simple example Python package that demonstrates how to create and install packages using `setup.py` or `pyproject.toml`.

## Features

- **Calculator**: A simple calculator class with basic arithmetic operations
- **Utilities**: Helper functions for common tasks

## Installation

### Using setup.py directly

```bash
python setup.py install
```

### Install the package:
If there is an pyproject.toml, this file will be prioritized before setup.py.

```bash
pip install -e .
```

Or install in development mode:
```bash
pip install -e .[dev]
```

### After Installation

There will be an .egg-info folder under the current folder, or in your system path.

- It contains information about the installed package, so Python tools (like pip, setuptools, or dependency resolvers) know what’s installed and what version it is.
- It acts like a "package manifest" rather than actual code.

## Usage

```python
from my_package import Calculator, greet

# Use the calculator
calc = Calculator(10)
result = calc.add(5).multiply(2).get_result()
print(result)  # Output: 30

# Use utility functions
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

## Development

### Install development dependencies

```bash
pip install -e .[dev]
```

### Code formatting

```bash
black my_package/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
