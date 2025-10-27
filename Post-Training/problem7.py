"""
Problem 7: Implement memory-efficient scaled dot-product attention by splitting
keys/values into sections and combining per-chunk results, along with a
numerically stable softmax suitable for large sequences.
"""

import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple
import math

npt_f32 = npt.NDArray[np.float32]

def softmax_memory_efficient(s: npt_f32, dim: int):
    """
    Inputs:
    - s: A numpy array of shape (N, M)
    - dim: Dimension to do softmax over
    """

    # *****Implement softmax_memory_efficient*****
    # We recommend copy softmax and modify as needed

    # *****Implement softmax*****
    # s_max = s.max(axis=dim, keepdims=True) # [N, 1]
    # s = s - s_max # [N, M]
    exps = np.exp(s)
    s = exps/np.sum(exps, axis=dim, keepdims=True) # [N, M]

    pass


def scaled_dot_product_attention_regular_forward_memory_efficient(
    q: npt_f32,
    k: npt_f32,
    v: npt_f32,
):
    """
    Self-attention forward

    Inputs:
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M/s, D)
    - v: A numpy array of shape (M/s, D)

    Returns:
    - output: A numpy array of shape (N, D)
    """

    # *****Implement scaled_dot_product_attention_regular_forward_memory_efficient*****
    # We recommend copy scaled_dot_product_attention_regular_forward and modify as needed
    N, D = q.shape
    M, _ = k.shape
    att0 = np.exp(q @ k.T / np.sqrt(D)) # [N, M/s]
    att1 = att0 @ v # [N, M/s]
    softmax_denom = att0.sum(axis=-1 ,keepdims=True) # [N, 1]

    return softmax_denom, att1


def scaled_dot_product_attention_efficient_forward(
    q: npt_f32,
    k: npt_f32,
    v: npt_f32,
    s: int
) -> npt_f32:
    """
    Memory Efficient Self-attention forward
    Steps:
    (A) Split k and v into s sections.
    (B) Perform attention on (q, k_i, v_i) individually in a for loop.
    (C) Combine the results of attention.

    Inputs:
    - q: A numpy array of shape (N, D)
    - k: A numpy array of shape (M, D)
    - v: A numpy array of shape (M, D)
    - s: An integer stating number of sections to split k and v into.

    Returns:
    - output: A numpy array of shape (N, D)
    """

    # *****Implement scaled_dot_product_attention_efficient_forward*****
    N = q.shape[0]
    k_splits = np.split(k, s, axis=0) # [M/s, D]
    v_splits = np.split(v, s, axis=0)
    att = np.zeros_like(q) # [N, D]
    softmax_sum = np.zeros([N, 1])
    for i in range(s):
        softmax_denom, att1 = scaled_dot_product_attention_regular_forward_memory_efficient(q, k_splits[i], v_splits[i])
        softmax_sum += softmax_denom
        att += att1

    return att/softmax_sum

if __name__ == "__main__":
	dim = 8

	sequence_length = 12
	grad = np.random.rand(15, dim).astype('float32')
	q = np.random.rand(15, dim).astype('float32')

	k = np.random.rand(sequence_length, dim).astype('float32')
	v = np.random.rand(sequence_length, dim).astype('float32')

	output = scaled_dot_product_attention_efficient_forward(
		q=q,
		k=k,
		v=v,
		s=4
	)

	print(output)
