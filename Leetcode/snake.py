import random
import time
import sys
import copy
import keyboard

def replace_str(s: str, newstring: str, index: int):
	assert isinstance(s, str)
	assert isinstance(newstring, str)
	s = s[:index] + newstring + s[index + 1:]
	return s


def init_board_and_snake(n: int, num_fruits: int):
	array = ['.'*n]*n

	# Initialize the snake to be horizontal;
	snake = [(random.choice(range(0, n)), random.choice(range(0, n - 1)))]
	snake.append((snake[-1][0], snake[-1][1] + 1))

	# Initialize the fruit
	for _ in range (num_fruits):
		x, y = random.choice(range(0, n)), random.choice(range(0, n))
		if (x, y) not in snake:
			array[x] = replace_str(array[x], 'X', y)

	return array, snake


def render_snake(array: list[str], snake: list[tuple]):
	new_array = copy.deepcopy(array)
	for body in snake:
		new_array[body[0]] = replace_str(new_array[body[0]], '@', body[1])
	return new_array


def plot_board(array: list[str]):
	array = [' '.join(x) for x in array]
	output = '\n'.join(array)
	sys.stdout.write("\033[H\033[J")
	sys.stdout.write(output)
	sys.stdout.flush()


def move(snake: list[tuple], direction: str, n: int):
	head = snake[-1]
	removed = snake.pop(0)
	if direction == 'right':
		if head[1] < n - 1:
			snake.append((head[0], head[1] + 1))
	elif direction == 'left':
		if head[1] > 0:
			snake.append((head[0], head[1] - 1))
	elif direction == 'up':
		if head[0] > 0:
			snake.append((head[0] - 1, head[1]))		
	elif direction == 'down':
		if head[0] < n - 1:
			snake.append((head[0] + 1, head[1]))
	return snake, removed


def handle_key_event(event):
	global prev
	if event.name == 'up':
		if prev not in ['up', 'down']:
			prev = 'up'
	elif event.name == 'down':
		if prev not in ['up', 'down']:
			prev = 'down'
	elif event.name == 'left':
		if prev not in ['left', 'right']:
			prev = 'left'
	elif event.name == 'right':
		if prev not in ['left', 'right']:
			prev = 'right'


if __name__ == '__main__':
	N = 20
	FRUITS = 14
	board, snake = init_board_and_snake(N, FRUITS)
	prev = 'right'

	keyboard.hook(handle_key_event)

	while True:
		plot_board(render_snake(board, snake))
		snake, removed = move(snake, prev, N)
		if not snake:
			print("FAILED!!")
			break

		if board[snake[-1][0]][snake[-1][1]] == 'X':
			board[snake[-1][0]] = replace_str(board[snake[-1][0]], '.', snake[-1][1])
			snake.insert(0, removed)

		time.sleep(1)
		# keyboard.wait()