# import pytest
# it = iter([1,2,3])
# assert it.__next__() == 1
# assert it.__next__() == 2
# assert it.__next__() == 3
# with pytest.raises(StopIteration):
#     it.__next__()

# it = resumable_iter([1,2,3])
# assert it.__next__() == 1
# s = it.get_state()
# assert it.__next__() == 2
# assert it.__next__() == 3
# it.set_state(s)  # go back to previous point of iteration!
# assert it.__next__() == 2
# assert it.__next__() == 3

from _typeshed import FileDescriptor


class ResumableIterator():
  def __iter__(self):
    return self

  def __next__(self):
    pass

  def get_state(self):
    pass

  def set_state(self, state):
    pass

##############################################
# PART 1: Implement general purpose test_resumable_iter
# that iterates over the inputed list and at each point:
# (1) checks the value is what's expected and
# (2) saves a state and later verifies that resuming 
# from the state matches same output.
# Usage:
#.  expected = ["o", "p", "e", "n"]
#.  it = ResumableIterator(expected)
#.  test_resumable_iter(it, expected) <--
##############################################
def test_resumable_iter(it: ResumableIterator, expected: list):
    # test simple iteration matches expected, collecting states
    # TODO

    pass

    # test resuming from collected states matches expected
    # TODO
    pass



##############################################
# PART 2: Implement ListIterator
##############################################

class ListIterator(ResumableIterator):
  def __init__(self, contents: list):
    pass

  def __next__(self):
    pass

  def get_state(self):
    pass

  def set_state(self, state):
    pass



##############################################
# PART 3: Implement MultiJsonFileIterator
##############################################


# d0.jsonl:
#     {"item": 1}
#     {"item": 2}

class JsonlFileIterator(ResumableIterator):
    def __init__(self, filename):
        ...
    ...

it = JsonlFileIterator("d0.jsonl")
s = it.get_state()
assert next(it) == {'item': 1}
assert next(it) == {'item': 2}
it.set_state(s)
assert next(it) == {'item': 1}


filenames = [f"d{i}.jsonl" for i in range(5)]
multifile_it = ??

assert next(multifile_it) == {'item': 1}
s = multifile_it.get_state()
assert next(multifile_it) == {'item': 2}
assert next(multifile_it) == {'item': 3}
multifile_it.set_state(s)
assert next(multifile_it) == {'item': 2}
assert next(multifile_it) == {'item': 3}

# d0.jsonl:
#     {"item": 1}
#     {"item": 2}
# d1.jsonl:
#     {"item": 3}
#     {"item": 4}
#     {"item": 5}
# d2.jsonl:
#     {"item": 6}
#     {"item": 7}
# d3.jsonl: <empty>
# d4.jsonl:
#     {"item": 8}


class MultiJsonlFileIterator(ResumableIterator):
    def __init__(self, filename):
        pass

    def __next__(self):
        pass

    def get_state(self):
        pass

    def set_state(self, state):
        pass