"""
Main body of the NTM model. Uses NTM_Cell objects, which hold head and controller
parameters and can perform one step of the NTM.

Currently, only the type of optimizer and overall learning rate is included in
config - other parameters of optimisers are left as default.

We follow a standard of capitalization for names of classes, lower case for instances
of those classes. Normal variables are using camel-case.

We assume that inputs for all tasks will begin with a 'start' token and finish with
an 'end' token - the task functions do not include these, however: we append them in
the course of operation.

Written with the understanding that creating an NTM_Cell object will at that time
only call the __init__ method, so no variables will be created. Instead, the
weights and biases will be added to the graph when step() is called, under the
scope at that time.

For some tasks, a 'mask' may be needed when the output length is dependent on some
property of the input. To correctly define the graph, we need to account for the
full possible length, but the loss may only reflect a portion of it. The idea of the
mask is to keep track of what part of the output is important to loss, by checking
when the stop-token appears in true output.
"""

import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from NTMCell import CellConfig, NTM_Cell
from utils import binary_cross_entropy_loss, copy_task


class Config(object):
    """The config class stores various hyperparameters and dataset parameters.
    Model objects are passed a Config() object at instantiation. We include
    batch-size for completeness, however only a size of 1 will work for now."""
    cellconfig = CellConfig()
    TrainSeed = None
    TestSeed = None
    MaxEpochs = 50
    EarlyStopping = 3
    EpochSize = 2000
    TestSize = 50
    LearningRate = 1e-4
    BatchSize = 1

    TaskName = 'Copy'
    TaskFunction = copy_task
    def TaskOutputLength(self, Length):
        return Length+1
    MaxOutputLength = 21 # only for Copy of MaxLength 20
    TaskMaskNeeded = False
    Zeros = tf.zeros([BatchSize,cellconfig.InputDepth])
    npStartToken = np.zeros([BatchSize,cellconfig.InputDepth])
    npStartToken[:,-2] = 1.
    StartToken = tf.convert_to_tensor(npStartToken, dtype=tf.float32)
    npStopToken = np.zeros([BatchSize,cellconfig.InputDepth])
    npStopToken[:,-1] = 1.
    StopToken = tf.convert_to_tensor(npStopToken, dtype=tf.float32)
    MinLength = 5
    MaxLength = 20
    TestLengths = [10, 20, 30, 40]


class NTM_Model(object):
    """
    Creates an NTM model defined by a given NTMCell.
    First we build the computational graph with a 'ladder'-type structure up to
    the maximum length, including optimiser functions at each length if the model
    is to be trainable (which is the case unless the model has Test=True).
    """
    def __init__(self, config, Test=False):
        self.config = config
        self.cellconfig = self.config.cellconfig
        self.optimizer = tf.train.RMSPropOptimizer(self.config.LearningRate)
        self.build_graph()


    def get_initial_state(self, Reuse=None):
        """
        Gets the initial state (weights and memory) of the NTM model.
        """
        trunc = tf.truncated_normal_initializer(stddev=0.5)
        steps = tf.cast(tf.reverse(tf.range(0,limit=self.cellconfig.MemoryLength),[True]),tf.float32)
        with tf.variable_scope('Initializer', reuse=Reuse):
            Memory = tf.get_variable('InitialMemory',
                [self.cellconfig.MemoryDepth,self.cellconfig.MemoryLength],tf.float32,
                initializer = trunc)
            ReadWeights = []
            WriteWeights = []
            for i in xrange(self.cellconfig.nReadHeads):
                WeightInit = tf.get_variable('ReadWeightInit%d'%i,
                    [self.config.BatchSize,self.cellconfig.MemoryLength], tf.float32,
                    initializer = trunc)
                ReadWeights.append(tf.nn.softmax(WeightInit+steps))
            for i in xrange(self.cellconfig.nWriteHeads):
                WeightInit = tf.get_variable('WriteWeightInit%d'%i,
                    [self.config.BatchSize,self.cellconfig.MemoryLength], tf.float32,
                    initializer = trunc)
                WriteWeights.append(tf.nn.softmax(WeightInit+steps))
        return ReadWeights, WriteWeights, Memory


    def build_graph(self):
        """
        Builds graph for the NTM model.

        The Inputs and Outputs will be given separately for each timestep, so for
        consistency with this, the graph will also given predicted outputs as a list.
        """
        self.ntm_cell = NTM_Cell(self.cellconfig)

        State = self.get_initial_state(Reuse=None)
        with tf.variable_scope('NTM'):
            _, State = self.ntm_cell.step(self.config.StartToken, State)

        self.InputPlaceholders = []
        self.OutputPlaceholders = []
        for InputLength in xrange(self.config.MaxLength):
            Input = tf.placeholder(tf.float32,
                [self.config.BatchSize,self.cellconfig.InputDepth], 'Input%d'%InputLength)
            self.InputPlaceholders.append(Input)
        for OutputLength in xrange(self.config.MaxOutputLength):
            TrueOutput = tf.placeholder(tf.float32,
                [self.config.BatchSize,self.cellconfig.InputDepth], 'Output%d'%OutputLength)
            self.OutputPlaceholders.append(TrueOutput)

        self.Outputs = {}
        self.Losses = {}
        self.TrainOps = {}
        trainables = tf.trainable_variables()

        with tf.variable_scope('NTM', reuse=True):
            for Length in xrange(1,self.config.MaxLength+1):
                _, State = self.ntm_cell.step(self.InputPlaceholders[Length-1], State)
                if Length >= self.config.MinLength:
                    _, OutState = self.ntm_cell.step(self.config.StopToken, State)
                    OutputList = []
                    OutputLength = self.config.TaskOutputLength(Length)
                    for OutputStep in xrange(OutputLength):
                        Output, OutState = self.ntm_cell.step(self.config.Zeros,OutState)
                        OutputList.append(Output)
                        Mask = None
                        if self.config.TaskMaskNeeded:
                            raise NotImplementedError
                    self.Outputs[Length] = OutputList
                    TrueOutput = self.OutputPlaceholders[:OutputLength]
                    Loss = binary_cross_entropy_loss(OutputList, TrueOutput, Mask)
                    Grads = []
                    for Grad in tf.gradients(Loss, trainables):
                        if Grad is not None:
                            Grads.append(tf.clip_by_value(Grad,-10.,10.))
                        else:
                            Grads.append(Grad)
                    self.Losses[Length] = Loss
                    self.TrainOps[Length] = self.optimizer.apply_gradients(zip(Grads,trainables))


    def run_epoch(self, session, test=False, verbose=200):
        """
        Runs one training or testing epoch of the model. If train_op is None, the
        model will be run for TestSize examples with fixed parameters, otherwise we
        train for EpochSize examples.
        """
        minl, maxl, depth = self.config.MinLength, self.config.MaxLength, self.cellconfig.InputDepth
        examples, seed = self.config.EpochSize, self.config.TrainSeed
        if test:
            examples, seed = self.config.TestSize, self.config.TestSeed
        TotalLoss = []
        for i in xrange(examples):
            Length, Inputs, Labels = self.config.TaskFunction(minl, maxl, depth, self.config.npStopToken, seed)
            feed = {Input: Vec for Input, Vec in zip(self.InputPlaceholders, Inputs)}
            feed.update({Output: Vec for Output, Vec in zip(self.OutputPlaceholders,Labels)})
            if not test:
                loss, _ = session.run([self.Losses[Length],self.TrainOps[Length]], feed_dict=feed)
            else:
                loss = session.run([self.Losses[Length]], feed_dict=feed)
            TotalLoss.append(loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(i,examples,np.mean(TotalLoss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.mean(TotalLoss)


    def generate(self, session):
        """
        Builds the graph and runs the NTM model for an example of each length in
        TestLengths, and saves the result as a figure.
        """
        depth, seed = self.cellconfig.InputDepth, self.config.TestSeed
        for Length in self.config.TestLengths:
            _, Inputs, Labels = self.config.TaskFunction(Length,Length,depth,self.config.npStopToken,seed)
            InputPlaceholders = []
            LabelPlaceholders = []
            Outputs = []
            State = self.get_initial_state(Reuse=True)
            with tf.variable_scope('NTM', reuse=True):
                _, State = self.ntm_cell.step(self.config.StartToken, State)
                for i in xrange(Length):
                    Input = tf.placeholder(tf.float32, [1,self.cellconfig.InputDepth])
                    InputPlaceholders.append(Input)
                    _, State = self.ntm_cell.step(Input, State)
                _, State = self.ntm_cell.step(self.config.StopToken, State)
                for i in xrange(self.config.TaskOutputLength(Length)):
                    Output, State = self.ntm_cell.step(self.config.Zeros, State)
                    Outputs.append(Output)
                    LabelPlaceholders.append(tf.placeholder(tf.float32, [1,self.cellconfig.InputDepth]))
            Mask = None
            if self.config.TaskMaskNeeded:
                """If we add a mask, we may also need to change evaluations below to return
                the mask too, so that we can accurately display the output."""
                raise NotImplementedError
            Loss = binary_cross_entropy_loss(Outputs, LabelPlaceholders, Mask)
            feed = {Holder: Vec for Holder, Vec in zip(InputPlaceholders, Inputs)}
            feed.update({Holder: Vec for Holder, Vec in zip(LabelPlaceholders, Labels)})
            Outputs, Loss = session.run([Outputs, Loss], feed_dict = feed)
            #print 'Test Length {} : Loss = {}'.format(Length, Loss)
            Labels = np.concatenate(Labels).T
            Outputs = np.concatenate(Outputs).T
            plt.clf()
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.imshow(Labels, cmap=plt.cm.jet, interpolation='none')
            ax0.set_title('Targets')
            ax0.yaxis.set_visible(False)
            ax1.imshow((1.-Outputs), cmap=plt.cm.jet, interpolation='none')
            ax1.set_title('Outputs')
            ax1.yaxis.set_visible(False)
            plt.savefig('{} Task: Length {}'.format(self.config.TaskName, Length))


def Test_NTM():
    """
    Trains and Tests an NTM model.
    """
    config = Config()
    with tf.variable_scope('NTM_Model') as scope:
        model = NTM_Model(config)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            BestLoss = float('inf')
            BestEpoch = 0
            session.run(init)
            for epoch in xrange(config.MaxEpochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                TrainLoss = model.run_epoch(session)
                print 'Training Loss: {}'.format(TrainLoss)
                if TrainLoss < BestLoss:
                    BestLoss = TrainLoss
                    BestEpoch = epoch
                    saver.save(session, './%s_ntm.weights'%config.TaskName)
                if epoch - BestEpoch > config.EarlyStopping:
                    break
                print 'Total Time: {}'.format(time.time()-start)
            # saver.restore(session, './%s_ntm.weights'%config.TaskName)
            TestLoss = model.run_epoch(session, test=True)
            print 'Test Loss: {}'.format(TestLoss)
            model.generate(session)


if __name__ == "__main__":
    Test_NTM()
