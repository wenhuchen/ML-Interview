import random
from typing import List
import math
import time

class Node:
    def __init__(self, value: int):
        self.value = value
        self.connected = []
    
    def add_connected(self, node: 'Node'):
        self.connected.append(node)

    def __str__(self):
        return f'Node({self.value})'

    def __repr__(self):
        return self.__str__()


def test_group(nodes):
    global counter
    global gorundtruth_set

    counter += 1
    if set(nodes) & gorundtruth_set:
        return False
    else:
        return True


def base_case(nodes: List[int]) -> List[Node]:
    if len(nodes) == 3:
        if test_group([nodes[0], nodes[1]]):
            return [nodes[2]]
        else:
            if test_group([nodes[0], nodes[2]]):
                if test_group([nodes[1], nodes[2]]):
                    return [nodes[0]]
                else:
                    return [nodes[1], nodes[2]]
            else:
                return [nodes[1]]

def find_singletons(nodes: List[Node], good_group: List[Node]) -> List[Node]:
    if not nodes:
        return []
    elif len(nodes) <= 2:
        brokens = []
        for node in nodes:
            if not test_group(good_group + [node]):
                brokens.append(node)
        return brokens
    else:
        if not test_group(nodes):
            left_singletons = find_singletons(nodes[:len(nodes) // 2], good_group)
            right_singletons = find_singletons(nodes[len(nodes) // 2:], good_group)
            return left_singletons + right_singletons
        else:
            return []


def find_good_group(nodes: List[Node]) -> List[Node]:
    if not nodes:
        return []
    elif len(nodes) == 0:
        return []
    elif len(nodes) == 1:
        return []
    else:
        if test_group(nodes):
            return nodes
        else:
            tmp = find_good_group(nodes[:len(nodes) // 2])
            if tmp:
                return tmp
            else:
                tmp = find_good_group(nodes[len(nodes) // 2:])
                if tmp:
                    return tmp
                else:
                    return []


if __name__ == '__main__':
    num_nodes = 200000
    nodes = []
    broken_num = 10

    start = time.perf_counter()
    for i in range(1, num_nodes + 1):
        node = Node(i)
        nodes.append(node)

    broken_nodes = random.choices(nodes, k=broken_num)
    good_nodes = set(nodes) - set(broken_nodes)
    for node in nodes:
        if node not in broken_nodes:
            node.connected = good_nodes
        else:
            node.connected = set()
    print('preparation time: ', time.perf_counter() - start, 'seconds')

    start = time.perf_counter()
    groundtruth = broken_nodes
    gorundtruth_set = set(groundtruth)
    print('groundtruth: ', groundtruth)
    print('set construction time: ', time.perf_counter() - start, 'seconds')

    counter = 0

    start = time.perf_counter()
    good_group = find_good_group(nodes)
    singletons = find_singletons(nodes, good_group)
    print('processing time: ', time.perf_counter() - start, 'seconds')

    # print('singletons: ', singletons)
    print('counter: ', counter)
    print('complexity: ', len(singletons) * math.log(len(nodes)))
    assert len(singletons) == len(groundtruth)
    for singleton in singletons:
        assert singleton in groundtruth
