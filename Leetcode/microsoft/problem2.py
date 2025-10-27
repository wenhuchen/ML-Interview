"""
Given a singly linked list, write a function to check if it is a palindrome. The solution must have a time complexity of O(n) and a space complexity of O(1). The linked list node is defined as follows:

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
Example
Input: 1 -> 2 Output: False

Input: 1 -> 2 -> 2 -> 1 Output: True

Constraints
The number of nodes in the list is in the range [1, 10^5].
0 <= Node.val <= 9
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_ListNode(input_str):
	root = ListNode(input_str[0])
	prev = root
	for num in input_str[1:]:
		tmp = ListNode(num)
		prev.next = tmp
		prev = tmp
	return root

root = create_ListNode([1,2,3,4,5,5,4,3,2,1])


def see_the_node(root):
	while root:
		print(root.val)
		root = root.next
	print('--------')

def detect(root: ListNode):
	if root is None:
		return True
	if root.next is None:
		return False

	left = None
	mid = root
	right = mid.next

	while mid.val != right.val:
		mid.next = left
		left = mid
		mid = right
		right = mid.next
		if right is None:
			return False

	assert mid.val == right.val
	# Reaching the middle
	mid.next = left
	left = mid
	print('left val:', left.val, 'right val:', right.val)

	print('left chain: ', see_the_node(left))
	print('right chain: ', see_the_node(right))

	while left and right:
		if left.val == right.val:
			left = left.next
			right = right.next
		else:
			return False

	if left is None and right is None:
		return True

	return False


print(detect(root))

