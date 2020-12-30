"""
Part of the NTM project: The NTM Cell, which performs one step of an NTM model.

We assume that NTM tasks require outputs of the same size as their inputs. Changing
this requires only adding a new OutputDepth parameter, and changing the output shape
of the controller.

We also assume that the default initializers will be sufficient for the weights, and
zero initializers sufficient for the biases.

For being able to handle batch-sizes greater than one, we need to update:
cosine_similarity and circular_convolution in utils.py
"""

import tensorflow as tf
from utils import circular_convolution, cosine_similarity, Linear


class CellConfig(object):
    """
    Hyperparameters for the NTM_Cell object.
    """
    MemoryDepth = 16
    MemoryLength = 64
    nReadHeads = 1
    nWriteHeads = 1

    ShiftOffsets = [-1,0,1]
    InputDepth = 8

    ControllerType = None # FeedForward or LSTM
    ControlHiddenSize = 256 # a list of the sizes of hidden layers for the controller
    OutputNonLinearity = tf.tanh
    # May also try sigmoid and ReLU activation functions.


def ReadMemory(Weights, Memory):
    """
    Performs the read-head operation for one head.
    Output has shape (1,MemoryDepth).
    """
    return tf.matmul(Weights,Memory,transpose_b=True)


def WriteMemory(Weights, Erase, Add, Memory):
    """
    Performs the write-head operation for one head.
    Weights has shape (1,MemoryLength)
    Erase, Add have shape (1,MemoryDepth)
    Output has shape (MemoryDepth, MemoryLength), like PrevMemory.
    """
    Memory -= tf.mul(Memory,tf.matmul(Erase,Weights,transpose_a=True))
    Memory += tf.matmul(Add,Weights,transpose_a=True)
    return Memory


class NTM_Cell(object):
    def __init__(self, CellConfig):
        self.Params = CellConfig


    def HeadUpdate(self, ControlState, PrevWeights, Memory, IsWrite = False):
        """
        For one head, takes the control state, previous weight and memory, and outputs
        the new weight, and for write-heads, the erase and add vectors as well.
        """
        KeyVector = tf.tanh(Linear(ControlState, self.Params.MemoryDepth, 'KeyVector'))
        KeyStrength = tf.nn.softplus(Linear(ControlState, 1, 'KeyStrength'))
        Gate = tf.sigmoid(Linear(ControlState, 1, 'Gate'))
        ShiftWeights = tf.nn.softmax(Linear(ControlState, len(self.Params.ShiftOffsets), 'ShiftWeights'))
        Sharpen = tf.nn.softplus(Linear(ControlState, 1, 'Sharpen'))+1.

        Weights = tf.exp(KeyStrength*cosine_similarity(KeyVector,Memory))
        Weights /= tf.reduce_sum(Weights,1)
        Weights = Gate*Weights + (1.0-Gate)*PrevWeights
        Weights = circular_convolution(Weights,ShiftWeights,self.Params.ShiftOffsets)
        Weights = tf.pow(Weights,Sharpen)
        Weights /= tf.reduce_sum(Weights,1)

        if IsWrite:
            Erase = tf.sigmoid(Linear(ControlState, self.Params.MemoryDepth, 'Erase'))
            Add = tf.tanh(Linear(ControlState, self.Params.MemoryDepth, 'Add'))
            return Weights, Erase, Add
        else:
            return Weights


    def step(self, CurrentInput, PrevState):
        """
        Takes an input and the previous 'state' of the NTM and returns the output
        and the next state. The 'states' are a tuple of two lists of weights (in order
        of read heads, then write heads), and the memory (as a matrix).

        We may also want to see how the memory is being accessed at each step, in which
        case we would append the read and write vectors to the output.

        Weights: list of shape (1,MemoryLength)
        Memory: shape (MemoryDepth, MemoryLength)
        CurrentInput: shape (1,InputDepth)
        """
        ReadWeights, WriteWeights, Memory = PrevState
        ReadInputs = [ReadMemory(W, Memory) for W in ReadWeights]
        ControlInput = tf.concat(1,ReadInputs+[CurrentInput])

        # now we should put in a control network that takes ControlInput-size inputs and
        ControlState = tf.tanh(Linear(ControlInput, self.Params.ControlHiddenSize, 'Controller'))
        # returns a 'control-state'

        Output = tf.sigmoid(Linear(ControlState, self.Params.InputDepth, 'Output'))

        NextReadWeights = []
        NextWriteWeights = []
        Adds = []
        Erases = []
        for i in xrange(self.Params.nReadHeads):
            with tf.variable_scope('ReadHead%d'%i):
                NextReadWeights.append(self.HeadUpdate(ControlState,ReadWeights[i],Memory))
        for i in xrange(self.Params.nWriteHeads):
            with tf.variable_scope('WriteHead%d'%i):
                W, E, A = self.HeadUpdate(ControlState,WriteWeights[i],Memory,IsWrite=True)
            NextWriteWeights.append(W)
            Erases.append(E)
            Adds.append(A)
        for i in xrange(self.Params.nWriteHeads):
            Memory = WriteMemory(NextWriteWeights[i], Erases[i], Adds[i], Memory)

        return Output, (NextReadWeights, NextWriteWeights, Memory)
