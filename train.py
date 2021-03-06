import model
import chainer
from util.trainer import SNGANUpdater
from util.dataset import CIFAR10
from util.generate_fake_image import generate
from util.generate_inception_score import inception_score

if __name__ == '__main__':
    chainer.cuda.get_device_from_id(0).use()
    generator = model.Generator().to_gpu(device=0)
    discriminator = model.Discriminator().to_gpu(device=0)
    optimizer_generator = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_discriminator = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5, beta2=0.9)
    optimizer_generator.setup(generator)
    optimizer_discriminator.setup(discriminator)

    dataset = CIFAR10()
    iterator = chainer.iterators.MultiprocessIterator(dataset, 64, n_processes=6)

    updater = SNGANUpdater(iterator=iterator, optimizer_generator=optimizer_generator, optimizer_discriminator = optimizer_discriminator, generator=generator, discriminator=discriminator )
    trainer = chainer.training.Trainer(updater, (50000, 'iteration'), out="./checkpoint")
    report_keys = ["generator_loss", "discriminator_loss", "inception_score"]

#    trainer.extend(inception_score(generator, 100, "util/inception_model", 1000, 1), trigger=(1000, 'iteration'), priority=chainer.training.extension.PRIORITY_WRITER)
    trainer.extend(generate(generator, "./images"), trigger=(1000, 'iteration'), priority=chainer.training.extension.PRIORITY_WRITER)
    trainer.extend(chainer.training.extensions.PrintReport(report_keys), trigger=(100, 'iteration'))
    trainer.extend(chainer.training.extensions.LogReport(keys=report_keys, trigger=(100, 'iteration')))

    trainer.extend(chainer.training.extensions.LinearShift('alpha', (0.0001, 0.), (0, 50000), optimizer_generator))
    trainer.extend(chainer.training.extensions.LinearShift('alpha', (0.0001, 0.), (0, 50000), optimizer_discriminator))

    trainer.run()
