# code borrowed from https://github.com/hvy/chainer-inception-score
import math

import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import Variable
import numpy as np


class Mixed(Chain):
    def __init__(self, trunk):
        super().__init__()
        for name, link in trunk:
            self.add_link(name, link)
        self.trunk = trunk

    def __call__(self, x):
        hs = []
        for name, _ in self.trunk:
            h = getattr(self, name)(x)
            hs.append(h)
        return F.concat(hs)


class Tower(Chain):
    def __init__(self, trunk):
        super().__init__()
        for name, link in trunk:
            if not name.startswith('_'):
                self.add_link(name, link)
        self.trunk = trunk

    def __call__(self, x):
        h = x
        for name, f in self.trunk:
            if not name.startswith('_'):  # Link
                if 'bn' in name:
                    h = getattr(self, name)(h)
                else:
                    h = getattr(self, name)(h)
            else:  # AveragePooling2D, MaxPooling2D or ReLU
                h = f(h)
        return h


def _average_pooling_2d(h):
    return F.average_pooling_2d(h, 3, 1, 1)


def _max_pooling_2d(h):
    return F.max_pooling_2d(h, 3, 1, 1)


def _max_pooling_2d_320(h):
    return F.max_pooling_2d(h, 3, 2, 0)


class Inception(Chain):
    def __init__(self):
        super().__init__(
            conv=L.Convolution2D(3, 32, 3, stride=2, pad=0),
            conv_1=L.Convolution2D(32, 32, 3, stride=1, pad=0),
            conv_2=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            conv_3=L.Convolution2D(64, 80, 1, stride=1, pad=0),
            conv_4=L.Convolution2D(80, 192, 3, stride=1, pad=0),
            bn_conv=L.BatchNormalization(32),
            bn_conv_1=L.BatchNormalization(32),
            bn_conv_2=L.BatchNormalization(64),
            bn_conv_3=L.BatchNormalization(80),
            bn_conv_4=L.BatchNormalization(192),
            mixed=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(192, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(192, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(192, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(192, 32, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(32)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_1=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(256, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(256, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_2=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(288, 48, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(48)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(48, 64, 5, stride=1, pad=2)),
                    ('bn_conv_1', L.BatchNormalization(64)),
                    ('_relu_1', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=1, pad=1)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_3=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(288, 384, 3, stride=2, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(288, 64, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(64)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(64, 96, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(96)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(96, 96, 3, stride=2, pad=0)),
                    ('bn_conv_2', L.BatchNormalization(96)),
                    ('_relu_2', F.relu)
                ])),
                ('pool', Tower([
                    ('_pooling', _max_pooling_2d_320)
                ]))
            ]),
            mixed_4=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 128, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(128)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(128, 128, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(128)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(128, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 128, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(128)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(128, 128, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(128)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(128, 128, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(128)),
                    ('_relu_2', F.relu),
                    ('conv_3', L.Convolution2D(128, 128, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(128)),
                    ('_relu_3', F.relu),
                    ('conv_4', L.Convolution2D(128, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_5=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(160, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(160)),
                    ('_relu_2', F.relu),
                    ('conv_3', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(160)),
                    ('_relu_3', F.relu),
                    ('conv_4', L.Convolution2D(160, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_6=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(160, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 160, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(160)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(160)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(160, 160, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(160)),
                    ('_relu_2', F.relu),
                    ('conv_3', L.Convolution2D(160, 160, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(160)),
                    ('_relu_3', F.relu),
                    ('conv_4', L.Convolution2D(160, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_7=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu),
                    ('conv_3', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_3', L.BatchNormalization(192)),
                    ('_relu_3', F.relu),
                    ('conv_4', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_4', L.BatchNormalization(192)),
                    ('_relu_4', F.relu)
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_8=Mixed([
                ('tower', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(192, 320, 3, stride=2, pad=0)),
                    ('bn_conv_1', L.BatchNormalization(320)),
                    ('_relu_1', F.relu)
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(768, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(192, 192, (1, 7), stride=1, pad=(0, 3))),
                    ('bn_conv_1', L.BatchNormalization(192)),
                    ('_relu_1', F.relu),
                    ('conv_2', L.Convolution2D(192, 192, (7, 1), stride=1, pad=(3, 0))),
                    ('bn_conv_2', L.BatchNormalization(192)),
                    ('_relu_2', F.relu),
                    ('conv_3', L.Convolution2D(192, 192, 3, stride=2, pad=0)),
                    ('bn_conv_3', L.BatchNormalization(192)),
                    ('_relu_3', F.relu)
                ])),
                ('pool', Tower([
                    ('_pooling', _max_pooling_2d_320)
                ]))
            ]),
            mixed_9=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(1280, 320, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(320)),
                    ('_relu', F.relu),
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(1280, 384, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.relu),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.relu),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.relu),
                        ]))
                    ]))
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(1280, 448, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(448)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(448, 384, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(384)),
                    ('_relu_1', F.relu),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.relu),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.relu),
                        ]))
                    ]))
                ])),
                ('tower_2', Tower([
                    ('_pooling', _average_pooling_2d),
                    ('conv', L.Convolution2D(1280, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            mixed_10=Mixed([
                ('conv', Tower([
                    ('conv', L.Convolution2D(2048, 320, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(320)),
                    ('_relu', F.relu),
                ])),
                ('tower', Tower([
                    ('conv', L.Convolution2D(2048, 384, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(384)),
                    ('_relu', F.relu),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.relu),
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.relu),
                        ]))
                    ]))
                ])),
                ('tower_1', Tower([
                    ('conv', L.Convolution2D(2048, 448, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(448)),
                    ('_relu', F.relu),
                    ('conv_1', L.Convolution2D(448, 384, 3, stride=1, pad=1)),
                    ('bn_conv_1', L.BatchNormalization(384)),
                    ('_relu_1', F.relu),
                    ('mixed', Mixed([
                        ('conv', Tower([
                            ('conv', L.Convolution2D(384, 384, (1, 3), stride=1, pad=(0, 1))),
                            ('bn_conv', L.BatchNormalization(384)),
                            ('_relu', F.relu)
                        ])),
                        ('conv_1', Tower([
                            ('conv_1', L.Convolution2D(384, 384, (3, 1), stride=1, pad=(1, 0))),
                            ('bn_conv_1', L.BatchNormalization(384)),
                            ('_relu_1', F.relu)
                        ]))
                    ]))
                ])),
                ('tower_2', Tower([
                    ('_pooling', _max_pooling_2d),
                    ('conv', L.Convolution2D(2048, 192, 1, stride=1, pad=0)),
                    ('bn_conv', L.BatchNormalization(192)),
                    ('_relu', F.relu)
                ]))
            ]),
            logit=L.Linear(2048, 1008)
        )

    def __call__(self, x, get_feature=False):
        """Input dims are (batch_size, 3, 299, 299)."""

        # assert x.shape[1:] == (3, 299, 299)

        x -= 128.0
        x *= 0.0078125

        h = F.relu(self.bn_conv(self.conv(x)))
        # assert h.shape[1:] == (32, 149, 149)

        h = F.relu(self.bn_conv_1(self.conv_1(h)))
        # assert h.shape[1:] == (32, 147, 147)

        h = F.relu(self.bn_conv_2(self.conv_2(h)))
        # assert h.shape[1:] == (64, 147, 147)

        h = F.max_pooling_2d(h, 3, stride=2, pad=0)
        # assert h.shape[1:] == (64, 73, 73)

        h = F.relu(self.bn_conv_3(self.conv_3(h)))
        # assert h.shape[1:] == (80, 73, 73)

        h = F.relu(self.bn_conv_4(self.conv_4(h)))
        # assert h.shape[1:] == (192, 71, 71)

        h = F.max_pooling_2d(h, 3, stride=2, pad=0)
        # assert h.shape[1:] == (192, 35, 35)

        h = self.mixed(h)
        # assert h.shape[1:] == (256, 35, 35)

        h = self.mixed_1(h)
        # assert h.shape[1:] == (288, 35, 35)

        h = self.mixed_2(h)
        # assert h.shape[1:] == (288, 35, 35)

        h = self.mixed_3(h)
        # assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_4(h)
        # assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_5(h)
        # assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_6(h)
        # assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_7(h)
        # assert h.shape[1:] == (768, 17, 17)

        h = self.mixed_8(h)
        # assert h.shape[1:] == (1280, 8, 8)

        h = self.mixed_9(h)
        # assert h.shape[1:] == (2048, 8, 8)

        h = self.mixed_10(h)
        # assert h.shape[1:] == (2048, 8, 8)

        h = F.average_pooling_2d(h, 8, 1)
        # assert h.shape[1:] == (2048, 1, 1)
        h = F.reshape(h, (-1, 2048))

        if get_feature:
            return h
        else:
            h = self.logit(h)
            h = F.softmax(h)

            # assert h.shape[1:] == (1008,)

            return h


def inception_score(generator, minibatch, directory, num_image, split):
    @chainer.training.make_extension()
    def eval_inception():
        model = Inception()
        chainer.serializers.load_hdf5(directory, model)
        model.to_gpu()
        images = []
        for i in range(0, num_image, minibatch):
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = generator(minibatch)
            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            images.append(x)
        images = np.asarray(images)
        _, _, _, h, w = images.shape
        images = images.reshape((num_image, 3, h, w)).astype(np.float32)
        n, c, w, h = images.shape

        xp = model.xp
        y_softmax = xp.empty((n, 1008), dtype=xp.float32)  

        for i in range(int(math.ceil(float(n) / float(minibatch)))):
            start = (minibatch * i)
            end = min(minibatch * (i + 1), n)

            images_b = Variable(xp.asarray(images[start:end])) 
            images_b = F.resize_images(images_b, (299, 299)) if (w, h) != (299, 299) else images_b

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y = model(images_b)
            y_softmax[start:end] = y.data
        scores = xp.empty((split), dtype=xp.float32) 
        y_softmax = y_softmax[:, 1:1001]  

        for i in range(split):
            temp = y_softmax[(i * n // split):((i + 1) * n // split), :]
            scores[i] = xp.exp(xp.mean(xp.sum((temp * (xp.log(temp) - xp.log(xp.expand_dims(xp.mean(temp, 0), 0)))), 1)))

        chainer.reporter.report({
            'inception_score': xp.mean(scores)
        })

    return eval_inception
