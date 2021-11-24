import chainer
import numpy as np

class CIFAR10(chainer.dataset.dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        self.dataset, _ = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=255)

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image = np.asarray(self.dataset[i][0] / 128. - 1., np.float32)
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image, self.dataset[i][1]
