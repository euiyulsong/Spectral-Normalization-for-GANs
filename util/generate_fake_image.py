import chainer
import numpy as np
from PIL import Image

def generate(generator, dst):
    @chainer.training.make_extension()
    def generate(trainer):
        np.random.seed(0)
        images = []
        for i in range(0, 50000, 100):
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                image = generator(100)
            image = np.asarray(np.clip(chainer.cuda.to_cpu(image.data) * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            images.append(image)
        images = np.asarray(images)
        _, _, _, h, w = images.shape
        images = images.reshape((10, 10, 3, h, w)).transpose(0, 3, 1, 4, 2).reshape((10 * h, 10 * w, 3))
        Image.fromarray(images).save(dst + '/image{:0>8}.png'.format(trainer.updater.iteration))
    return generate