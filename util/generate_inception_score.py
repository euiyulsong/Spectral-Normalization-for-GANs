import chainer
import numpy as np
import math
#from util.inception import Inception

def inception_score(generator, minibatch, directory, num_image, split):
    @chainer.training.make_extension()
    def eval_inception(trainer=None):
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

            images_b = chainer.Variable(xp.asarray(images[start:end]))
            images_b = chainer.functions.resize_images(images_b, (299, 299)) if (w, h) != (299, 299) else images_b

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
