import chainer
import numpy as np

class CIFAR10(chainer.dataset.dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        self.dataset, _ = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=255)

    def __len__(self):
        return len(self.dataset)

    def get_example(self, x):
        scaled = np.asarray(self.dataset[x][0] / 128. - 1., np.float32) + np.random.uniform(size=(self.dataset[x][0] / 128. - 1.).shape, low=0., high=1. / 128)
        return scaled, self.dataset[x][1]
