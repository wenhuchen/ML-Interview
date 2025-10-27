from dataclasses import dataclass

@dataclass
class Node:
	name: str
	folder: bool
	children: list


a6 = Node(name='A6.py', folder=False, children=[])
a5 = Node(name='A5', folder=True, children=[a6])
a4 = Node(name='A4', folder=True, children=[a5])
a3 = Node(name='A3', folder=True, children=[])
a21 = Node(name='A21.py', folder=False, children=[])
a20 = Node(name='A20.cc', folder=False, children=[])
a2 = Node(name='A2', folder=True, children=[a20, a21])
a1 = Node(name='A1', folder=True, children=[])
a = Node(name='A', folder=True, children=[a1, a2, a3])
root = Node(name='/data/', folder=True, children=[a, a4])


def traverse(prefix: str, node: Node):
	print(prefix + node.name)
	for c in node.children:
		traverse(prefix=prefix + '  ', node=c)

traverse('', root)

