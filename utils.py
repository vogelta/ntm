"""
Utility functions for the Neural Turing Machine model.

For all task functions, inputs/outputs do not include start or stop tokens.
Output will have the stop token appended as part of the main function.

Input depth as given is the full depth including start and stop tokens, so the
functional input should be two rows less than this.
The length arguments given refer to the functional length.
"""

import numpy as np
import tensorflow as tf


def cosine_similarity(vector, matrix):
    """Calculates cosine similarity between a vector and the rows of a matrix.
    The length of the vector must equal number of rows of the matrix.
    Output has shape (1,ncols(matrix))"""
    dot_products = tf.matmul(vector,matrix)
    norm_vector = tf.sqrt(tf.reduce_sum(tf.square(vector)))
    norm_matrix = tf.sqrt(tf.reduce_sum(tf.square(matrix),0,keep_dims=True))
    return dot_products/(norm_vector*norm_matrix + 1e-10)


def circular_convolution(vector, shift_weights, shift_distances):
    """Calculates the circular convolution of vector according to the movement
    specified by shift_weights and shift_distances.
    'vector' is given as an (1,N) tensor."""
    N = tf.shape(vector)[1]
    output = []
    for i,distance in enumerate(shift_distances):
        indices = tf.range(distance+N,distance+2*N)
        indices = tf.mod(indices,N)
        output.append(shift_weights[0,i]*tf.gather(vector[0,:],indices))
    return tf.expand_dims(tf.add_n(output),0)


def copy_task(_, min_l, max_l, depth, stop_token, seed=None):
    """Produces one example of the copy-task.
    Returns:
    - length of inputs for this example
    - inputs as a list of length 'length'
    - target as a list of maximum output length for the given input length.
        stop_token is at the end.
    """
    if seed:
        np.random.seed(seed)
    length = np.random.randint(min_l, max_l+1)
    body = np.random.randint(0,2,size=(length,depth-2)).astype(float)
    body = np.hstack([body, np.zeros((length,2))])
    body = np.vsplit(body, length)
    return length, body, body+[stop_token]


def Linear(Input, OutputSize, Scope):
    """
    Creates a linear transformation with scope 'Scope'. We are always using a bias
    term, which is always initialized to zeros.

    Input: a 2D tensor of shape (BatchSize, InputSize)
    Returns: a 2D tensor of shape (BatchSize, OutputSize)
    """
    # Also I think we will need a 'reuse' input: on building the graph, we don't reuse,
    # but on running it we do? Or else no reuse on the first step, yes reuse on all after.
    InputSize = Input.get_shape().as_list()[1]
    with tf.variable_scope(Scope): # do we actually need to var scope here? or can we put it before the
    # call to Linear?
        Weights = tf.get_variable('Weight', [InputSize, OutputSize])
        Bias = tf.get_variable('Bias', [OutputSize], initializer=tf.zeros_initializer)
    return tf.matmul(Input,Weights) + Bias


def binary_cross_entropy_loss(logits, targets, mask=None):
    """Computes cross-entropy loss between logits and targets for the case where
    there are exactly two allowed classes 0 and 1. Logits are taken as the predicted
    probability    of class 0.

    Logits and targets are presented as lists of 2D tensors of the same shape
    (batch-size, output-size). Currently only implemented for batch-size of 1.

    If batch-size is greater than 1, then we have to consider that examples may have
    different lengths.
    """
    if mask is None:
        probs = [tf.log(tf.abs(l-t)) for l, t in zip(logits, targets)]
        loss = -tf.reduce_mean(tf.concat(0,probs))
    else:
        raise NotImplementedError
    return loss
