import os
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from mxnet import gluon, nd, autograd
from mxnet.gluon.data.vision.transforms import ToTensor, Resize



####################
# Hyper parameters #
ctx = mx.gpu()
# Common
epoch = 20
batch_size = 4


# Optimizer
lr = 0.00005
optimizer = "sgd"
momentum = 0.9

# ConvLSTM
size = 50
depth = 5
target = 1
channel = 1
kernel = (5,5,5)
i2h_pad = (2,2,2)
# Hyper parameters #
####################



#################
# Build network #
class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = gluon.rnn.HybridSequentialRNNCell()
            
            self.encoder.add(gluon.contrib.rnn.Conv3DLSTMCell(input_shape=[channel,depth,size,size], hidden_channels=24,
                                      i2h_kernel=kernel, i2h_pad=i2h_pad, h2h_kernel=kernel))
            
            self.encoder.add(gluon.contrib.rnn.Conv3DLSTMCell(input_shape=[24,depth,size,size], hidden_channels=16,
                                      i2h_kernel=kernel, i2h_pad=i2h_pad, h2h_kernel=kernel))
            
            self.encoder.add(gluon.contrib.rnn.Conv3DLSTMCell(input_shape=[16,depth,size,size], hidden_channels=12,
                                      i2h_kernel=kernel, i2h_pad=i2h_pad, h2h_kernel=kernel))
            
            self.conv = gluon.nn.Conv3D(channels=1, kernel_size=(depth, 1, 1))
      
            
    def hybrid_forward(self, F, x, state=None):
        x, state = self.encoder(x, state)
        
        concated = mx.nd.concat(state[0], state[2], state[4], dim=1)
        
        x = self.conv(concated)
        
        return x
net = Net()
# Build network #
#################



#########################
# Weight initialization #
net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)

state = net.encoder.begin_state(batch_size=batch_size, ctx=ctx)
x = nd.empty(shape=(batch_size,channel,depth,size,size), ctx=ctx)

out = net(x, state)
# Weight initialization #
#########################



###################
# Data processing #
def sliding_window(array, depth, target):
    length = len(array)-depth-target-1
    data, label = [], []

    trans = Resize(size)
    for i in range(length):
        data.append(trans(mx.nd.stack(*array[i:i+depth]).astype("float32")))
        label.append(trans(mx.nd.expand_dims(array[i+depth+target].astype("float32"), 0)))
    return gluon.data.ArrayDataset(data, label)

images = []
for filename in sorted(os.listdir("handmade")):
    images.append(mx.image.imread("handmade/"+filename, flag=0))

dataset = sliding_window(images, depth, target).transform_first(ToTensor())
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch="discard", shuffle=True)
# Data processing #
###################



###################
# Declare metrics #
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr, "momentum": 0.9})
# Declare metrics #
###################



for e in range(epoch):
   for data, label in dataloader:
      data = mx.nd.swapaxes(data.as_in_context(ctx), 1, 2)
      label = mx.nd.swapaxes(label.as_in_context(ctx), 1, 2)
      
      states = net.encoder.begin_state(batch_size=batch_size, ctx=ctx)
      
      with autograd.record():
         pred = net(data, states)
         losses = loss(pred, label)
      losses.backward()
      trainer.step(batch_size)
   print(e, mx.nd.sum(losses))
