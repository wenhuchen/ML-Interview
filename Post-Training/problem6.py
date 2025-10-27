"""
Problem 6: Derive and implement the NumPy backward pass for scaled dot-product
attention (including a stable softmax and its Jacobian-vector product), and
verify gradients by comparing with PyTorch autograd.
"""

import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple
import math

npt_f32 = npt.NDArray[np.float32]

def softmax(v: npt_f32):
	max_v = v.max(-1)
	normed_v = v - max_v[:, None]
	exp_v = np.exp(normed_v)
	softmax = exp_v / exp_v.sum(-1)[:, None]
	return softmax


def softmax_backward(v: npt_f32, grad_v: npt_f32):
	"""
	- v; A numpy array of shape [N, M]
	- grad_v: A numpy array of shape [n, M]
	"""
	N, M = v.shape
	grad = np.zeros((N, M), dtype=np.float32)
	for i in range(N):
		tmp = np.diag(v[i]) - v[i][:, None] @ v[i][None, :]
		grad[i] = grad_v[i] @ tmp
	return grad


def scaled_dot_product_attention_regular_backward(
    grad: npt_f32,
    q: npt_f32,
    k: npt_f32,
    v: npt_f32,
) -> Tuple[npt_f32, npt_f32, npt_f32]:
    """
    Self-attention backward

    Inputs:
    - grad: A numpy array of shape (N, D)
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M, D)
    - v: A numpy array of shape (M, D)

    Returns:
    - dq, dk, dv
    """
    D = q.shape[1]
    # [N, M]
    attention = softmax(q @ k.T / np.sqrt(D))
    grad_v = attention.T @ grad
    grad_attention = grad @  v.T
    grad_pre_attention = softmax_backward(attention, grad_attention)
    grad_k = q.T @ grad_pre_attention / np.sqrt(D)
    grad_q = grad_pre_attention @ k / np.sqrt(D)
    return grad_q, grad_k, grad_v


def torch_version_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
	D = q.shape[1]
	pre_attention = q @ k.T / math.sqrt(D)
	attention = torch.softmax(pre_attention, -1)
	output = attention @ v
	return output


if __name__ == "__main__":
	dim = 8
	grad = np.random.rand(15, dim).astype('float32')
	q = np.random.rand(15, dim).astype('float32')
	k = np.random.rand(10, dim).astype('float32')
	v = np.random.rand(10, dim).astype('float32')

	grad_q, grad_k, grad_v = scaled_dot_product_attention_regular_backward(
		grad=grad,
		q=q,
		k=k,
		v=v
	)

	q_tensor = torch.from_numpy(q).requires_grad_(True)
	k_tensor = torch.from_numpy(k).requires_grad_(True)
	v_tensor = torch.from_numpy(v).requires_grad_(True)
	out = torch_version_sdpa(
		q=q_tensor,
		k=k_tensor,
		v=v_tensor,
	)
	out.backward(torch.from_numpy(grad).float())

	grad_q_torch = torch.from_numpy(grad_q).float()

	print(grad_q_torch)
	print(q_tensor.grad)

	# check elementwise closeness
	is_close = torch.allclose(grad_q_torch, q_tensor.grad, rtol=1e-3, atol=1e-8)
	print('is close:', is_close)
