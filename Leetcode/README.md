# Coding Interview Problems and Implementations

This directory contains implementations of coding problems commonly encountered in technical interviews, particularly those asked at top tech companies like OpenAI and game development scenarios.

## Overview

The problems here focus on fundamental computer science concepts including data structures, algorithms, game logic, and iterator design patterns. These implementations demonstrate problem-solving skills essential for software engineering interviews.

## Directory Structure

### `games/`
Contains implementations of classic programming games and challenges.

#### `snake.py` - Snake Game Implementation
**Classic Snake Game in Terminal**

A complete implementation of the Snake game that runs in the terminal:

**Core Features:**
- **Real-time gameplay**: Uses keyboard library for real-time input handling
- **Dynamic board rendering**: Terminal-based display with character representation
- **Collision detection**: Wall boundaries and self-collision detection  
- **Food mechanics**: Random fruit placement and snake growth
- **Direction controls**: Arrow key input with invalid move prevention

**Key Programming Concepts:**
- **Game loop**: Continuous update-render cycle
- **State management**: Board state, snake position, and direction tracking
- **Input handling**: Asynchronous keyboard event processing
- **Terminal graphics**: ANSI escape codes for screen clearing and rendering
- **Coordinate system**: 2D grid navigation and boundary checking

**Implementation Details:**
```python
# Board representation: '.' = empty, 'X' = fruit, '@' = snake
# Snake: List of (row, col) tuples representing body segments
# Movement: Head advancement with tail removal (unless fruit eaten)
```

### `openai/`
Contains advanced programming challenges typical of AI/ML company interviews.

#### `resume_iterator.py` - Resumable Iterator Pattern
**Advanced Iterator Design Challenge**

This problem tests understanding of iterator protocols, state management, and complex data processing patterns:

**Problem Parts:**

1. **General Iterator Testing Framework**
   - Design a test function that validates resumable iterator behavior
   - Must test both forward iteration and state restoration
   - Verifies that resuming from saved state produces identical output

2. **ListIterator Implementation**
   - Basic resumable iterator over a list
   - State management with position tracking
   - `get_state()` and `set_state()` functionality

3. **JsonlFileIterator Implementation**
   - File-based iterator over JSON Lines format
   - Must handle file I/O operations resumably
   - State includes file position for resumption

4. **MultiJsonlFileIterator Implementation**
   - Iterator across multiple JSONL files
   - Complex state management across file boundaries
   - Must handle empty files gracefully
   - State includes current file index and position within file

**Key Concepts Tested:**
- **Iterator Protocol**: `__iter__()`, `__next__()`, `StopIteration`
- **State Serialization**: Capturing and restoring complex internal state
- **File I/O Management**: Resumable file reading with position tracking
- **Error Handling**: Graceful handling of missing/empty files
- **Design Patterns**: Template method pattern with abstract base class

## Interview Preparation Insights

### Problem-Solving Approaches

1. **Game Development (Snake)**
   - Tests ability to model complex state
   - Requires understanding of game loops and real-time systems
   - Demonstrates proficiency with coordinate systems and collision detection

2. **Iterator Design (Resumable Iterator)**
   - Evaluates understanding of Python's iteration protocol
   - Tests ability to design complex stateful objects  
   - Requires careful consideration of edge cases and error handling

### Technical Skills Demonstrated

- **Data Structures**: Lists, tuples, coordinate systems
- **Object-Oriented Design**: Abstract base classes, inheritance
- **File I/O**: Stream processing, position management
- **State Management**: Serialization/deserialization of complex state
- **Real-time Programming**: Asynchronous input handling
- **Testing**: Comprehensive test case design

### Common Interview Themes

1. **System Design**: How to structure resumable, stateful systems
2. **Error Handling**: Graceful degradation and edge case management
3. **Performance**: Efficient iteration without unnecessary memory usage
4. **Extensibility**: Design patterns that allow for easy extension

## Running the Code

### Snake Game:
```bash
cd games/
python snake.py
# Use arrow keys to control the snake
# Collect 'X' symbols to grow
# Avoid walls and self-collision
```

**Requirements:**
- `keyboard` library: `pip install keyboard`
- Terminal with ANSI escape code support

### Resumable Iterator:
```bash
cd openai/
python resume_iterator.py
# Implement the missing methods and test
```

The iterator problems are designed as coding challenges - complete the implementation by filling in the missing methods based on the provided test cases and requirements.

## Key Takeaways

- **Game Programming**: Requires strong understanding of state management and real-time systems
- **Advanced Iterators**: Tests deep Python knowledge and ability to design complex, stateful objects
- **Code Organization**: Both problems require clean separation of concerns and modular design
- **Testing**: Emphasis on comprehensive testing including edge cases and state verification

These problems represent the type of technical challenges commonly encountered in interviews at top-tier technology companies, particularly those focusing on systems programming and AI/ML applications.