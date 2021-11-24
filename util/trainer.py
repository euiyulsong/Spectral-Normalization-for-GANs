import chainer
import numpy as np

class SNGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, iterator, optimizer_generator, optimizer_discriminator, generator, discriminator):
        self.iterator = iterator
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.generator = generator
        self.discriminator = discriminator

        super(SNGANUpdater, self).__init__(iterator=iterator, optimizer=optimizer_generator)

    def create_fake_x(self, resolution=None):
        resolution = 128 if resolution is None else resolution
        generator = self.generator
        x = generator(resolution, y=None)
        return x, None

    def update_core(self):
        generator = self.generator
        optimizer_generator = self.optimizer_generator

        discriminator = self.discriminator
        optimizer_discriminator = self.optimizer_discriminator
        xp = discriminator.xp
        for i in range(5):
            if i == 0:
                generator.cleargrads()

                gen_x, gen_y = self.create_fake_x()
                discriminator_generated = discriminator(gen_x, y=gen_y)
                generator_loss = -chainer.functions.mean(discriminator_generated)
                generator_loss.backward()
                optimizer_generator.update()
                chainer.reporter.report({'generator_loss': generator_loss})

            mini_batch = self.get_iterator('main').next()
            x = []
            for j in range(len(mini_batch)):
                x.append(np.asarray(mini_batch[j][0]).astype(np.float32))
            x = chainer.Variable(xp.asarray(x))
            discriminator_actual = discriminator(x, y=None)
            gen_x, gen_y = self.create_fake_x(resolution=len(x))
            discriminator_generated = discriminator(gen_x, y=gen_y)
            gen_x.unchain_backward()
            discriminator.cleargrads()

            discriminator_loss = chainer.functions.mean(chainer.functions.relu(1. - discriminator_actual)) + chainer.functions.mean(chainer.functions.relu(1. + discriminator_generated))
            discriminator_loss.backward()
            optimizer_discriminator.update()
            chainer.reporter.report({'discriminator_loss': discriminator_loss})