from GAN_CelebA.gan_celeba import GAN_CELEBA
from Settings.settings import *
from Utils.dataset_utils import *


if __name__ == "__main__":
    noise_dim = 300
    gan_celeba = GAN_CELEBA(noise_dim)
    gan_celeba.load_weights(CHECKPOINT_PREFIX)

    # generate 36 new images.
    input_noise = tf.random.normal((36, noise_dim))
    generated_images = np.clip(gan_celeba.generate_images(input_noise), 0, 1)

    visualize_images(generated_images)
