import chainer
import numpy as np
import math


class ResBlock(chainer.Chain):
    def __init__(self):
        super(ResBlock, self).__init__()

        with self.init_scope():
            self.conv_1 = chainer.links.Convolution2D(256, 256, ksize=3, pad=1,
                                                      initialW=chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.conv_2 = chainer.links.Convolution2D(256, 256, ksize=3, pad=1,
                                                      initialW=chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.bias_1 = chainer.links.BatchNormalization(256)
            self.bias_2 = chainer.links.BatchNormalization(256)
            self.conv_shortcut = chainer.links.Convolution2D(256, 256, ksize=1, pad=0,
                                                             initialW=chainer.initializers.GlorotUniform())

    def __call__(self, x):
        residual = chainer.functions.relu(self.bias_1(x))
        residual = self.conv_1(
            chainer.functions.unpooling_2d(residual, 2, outsize=(residual.shape[2:][0] * 2, residual.shape[2:][1] * 2)))
        return self.conv_2(chainer.functions.relu(self.bias_2(residual))) \
               + self.conv_shortcut(
            chainer.functions.unpooling_2d(x, 2, outsize=(x.shape[2:][0] * 2, x.shape[2:][1] * 2)))


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.linear_1 = chainer.links.Linear(128, 4 * 4 * 256, initialW=chainer.initializers.GlorotUniform())
            self.relu_1 = chainer.functions.relu
            self.tanh_1 = chainer.functions.tanh
            self.bias_1 = chainer.links.BatchNormalization(256)
            self.conv_1 = chainer.links.Convolution2D(256, 3, ksize=3, stride=1, pad=1,
                                                      initialW=chainer.initializers.GlorotUniform())

            self.resblock_1 = ResBlock()
            self.resblock_2 = ResBlock()
            self.resblock_3 = ResBlock()

    def __call__(self, size=64, z=None, y=None):
        x = self.linear_1(self.xp.random.randn(size, 128).astype(self.xp.float32)) if z is None else self.linear_1(z)
        return self.tanh_1(self.conv_1(self.relu_1(self.bias_1(self.resblock_3(self.resblock_2(self.resblock_1(chainer.functions.reshape(x, (x.shape[0], -1, 4, 4)))))))))


class SpectralNormConv2D(chainer.links.connection.convolution_2d.Convolution2D):
    def __init__(self, in_c, out_c, k, p, init_w):
        super(SpectralNormConv2D, self).__init__(in_c, out_c, k, 1, p, False, init_w, None)
        self.p = p

        self.u = np.random.normal(size=(1, out_c)).astype(np.float32)
        self.register_persistent('u')

    @property
    def W_bar(self):
        W_reshape = self.W.reshape(self.W.shape[0], -1)
        xp = chainer.cuda.get_array_module(W_reshape.data)
        if self.u is None:
            self.u = xp.random.normal(size=(1, W_reshape.shape[0])).astype(xp.float32)
        u = self.u
        sn = chainer.cuda.reduce('T x', 'T out', 'x * x', 'a + b', 'out = sqrt(a)', 0, 'norm_sn')
        divide = chainer.cuda.elementwise('T x, T norm, T eps', 'T out', 'out = x / (norm + eps)', 'div_sn')
        for i in range(1):
            v = divide(xp.dot(u, W_reshape.data), sn(xp.dot(u, W_reshape.data)), 1e-12)
            u = divide(xp.dot(v, W_reshape.data.transpose()), sn(xp.dot(v, W_reshape.data.transpose())), 1e-12)
        s = chainer.functions.array.broadcast.broadcast_to((chainer.functions.sum(
            chainer.functions.linear(u, chainer.functions.transpose(W_reshape)) * v)).reshape((1, 1, 1, 1)), self.W.shape)
        if chainer.config.train:
            self.u[:] = u
        return self.W / s

    def __call__(self, x):
        if self.W.data is None:
            super(SpectralNormConv2D, self)._initialize_params(x.shape[1])
        return chainer.functions.connection.convolution_2d.convolution_2d(
            x, self.W_bar, self.b, 1, self.p)


class DiscResBlock(chainer.Chain):
    def __init__(self, down):
        super(DiscResBlock, self).__init__()
        self.down = down
        with self.init_scope():
            self.relu_1 = chainer.functions.relu
            self.relu_2 = chainer.functions.relu

            self.conv_1 = SpectralNormConv2D(128, 128, 3, 1, chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.conv_2 = SpectralNormConv2D(128, 128, 3, 1, chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.conv_3 = SpectralNormConv2D(128, 128, 1, 0, chainer.initializers.GlorotUniform())

    def __call__(self, x):
        residual = self.conv_2(self.relu_2(self.conv_1(self.relu_1(x))))

        if self.down:
            return chainer.functions.average_pooling_2d(residual, 2) + chainer.functions.average_pooling_2d(self.conv_3(x), 2)
        else:
            return residual + x


class OptDiscResBlock(chainer.Chain):
    def __init__(self):
        super(OptDiscResBlock, self).__init__()
        with self.init_scope():
            self.relu_1 = chainer.functions.relu
            self.conv_1 = SpectralNormConv2D(3, 128, 3, 1, chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.conv_2 = SpectralNormConv2D(128, 128, 3, 1, chainer.initializers.GlorotUniform(math.sqrt(2)))
            self.conv_3 = SpectralNormConv2D(3, 128, 1, 0, chainer.initializers.GlorotUniform())

    def __call__(self, x):
        return chainer.functions.average_pooling_2d(self.conv_2(self.relu_1(self.conv_1(x))), 2)\
               + self.conv_3(chainer.functions.average_pooling_2d(x, 2))


class SpectralNormFC(chainer.links.connection.linear.Linear):
    def __init__(self):
        super(SpectralNormFC, self).__init__(128, 1, True, chainer.initializers.GlorotUniform(), None)
        self.u = np.random.normal(size=(1, 1)).astype(np.float32)
        self.register_persistent('u')

    @property
    def W_bar(self):
        W_reshape = self.W
        xp = chainer.cuda.get_array_module(W_reshape.data)
        if self.u is None:
            self.u = xp.random.normal(size=(1, W_reshape.shape[0])).astype(xp.float32)
        u = self.u
        sn = chainer.cuda.reduce('T x', 'T out', 'x * x', 'a + b', 'out = sqrt(a)', 0, 'norm_sn')
        divide = chainer.cuda.elementwise('T x, T norm, T eps', 'T out', 'out = x / (norm + eps)', 'div_sn')
        for i in range(1):
            v = divide(xp.dot(u, W_reshape.data), sn(xp.dot(u, W_reshape.data)), 1e-12)
            u = divide(xp.dot(v, W_reshape.data.transpose()), sn(xp.dot(v, W_reshape.data.transpose())), 1e-12)
        s = chainer.functions.array.broadcast.broadcast_to((chainer.functions.sum(
            chainer.functions.linear(u, chainer.functions.transpose(W_reshape)) * v)).reshape((1, 1)), self.W.shape)
        self.u[:] = u
        return self.W / s


    def __call__(self, x):
        if self.W.data is None:
            super(SpectralNormFC, self)._initialize_params(x.size // x.shape[0])
        return chainer.functions.connection.linear.linear(x, self.W_bar, self.b)


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.relu_1 = chainer.functions.relu
            self.opt_1 = OptDiscResBlock()
            self.res_1 = DiscResBlock(True)
            self.res_2 = DiscResBlock(False)
            self.res_3 = DiscResBlock(False)
            self.fc = SpectralNormFC()

    def __call__(self, x, y=None):
        return self.fc(chainer.functions.sum(self.relu_1(self.res_3(self.res_2(self.res_1(self.opt_1(x))))), axis=(2, 3)))
