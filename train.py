from Utils.dataset_utils import *
from Settings.settings import *
from GAN_CelebA.gan_celeba import GAN_CELEBA
from tqdm import tqdm
from zipfile import ZipFile


def train_model(gan_celeba: GAN_CELEBA, N, batch_size=32, epochs=6,
                load_weights=False):
    train_logger = logging.getLogger("Train")
    train_logger.setLevel(LOG_LEVEL)
    if load_weights:
        train_logger.info("Loading weights from previous checkpoint.")
        if os.path.exists(CHECKPOINT_PREFIX):
            gan_celeba.load_weights(CHECKPOINT_PREFIX)
            train_logger.info(
                "Weights were successfully loaded from previous checkpoint!")
        else:
            train_logger.warning("Previous checkpoint not found!")
    train_logger.info(f"Training on {N} samples.")
    for epoch in range(epochs):
        train_logger.info(f"Starting epoch {epoch+1}/{epochs}")
        gen_loss_history = []
        disc_loss_history = []
        for index_start in tqdm(range(0, N, batch_size)):
            index_end = min(N, index_start + batch_size)
            batched_images = load_batch_from_celebA_dir(
                list(range(index_start, index_end)))
            gen_loss, disc_loss = gan_celeba.train_step(batched_images)
            gen_loss_history.append(gen_loss)
            disc_loss_history.append(disc_loss_history)
        train_logger.info(
            f"Average generator loss of the epoch: {np.array(gen_loss).mean()}")
        train_logger.info(
            f"Average discriminator loss of the epoch: {np.array(disc_loss).mean()}")
        gan_celeba.save_weights(CHECKPOINT_PREFIX)
    train_logger.info("Model trained successfully!")


class CELEBA_ZIP_DOES_NOT_EXIST(Exception):
    def __init__(self, celeba_zip_path=CELEBA_ZIP):
        self.celeba_zip_path = celeba_zip_path
        self.message = f"Celeba zip file does not exist at {self.celeba_zip_path}." + \
            "\n Please download from https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ."
        super().__init__(self.message)


if __name__ == "__main__":
    N = 202599
    input_noise_dim = 300
    gan_celeba = GAN_CELEBA(input_noise_dim)

    main_logger = logging.getLogger("Main")
    main_logger.setLevel(LOG_LEVEL)

    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    if not os.path.exists(CELEBA_DIR):
        main_logger.warning("The celeba directory does not exist!")
        if os.path.exists(CELEBA_ZIP):
            main_logger.warning("Extracting data from celeba zip file.")
            with ZipFile(CELEBA_ZIP, "r") as zipObj:
                zipObj.extractall(DATASETS_DIR)
        else:
            raise CELEBA_ZIP_DOES_NOT_EXIST()

    train_model(gan_celeba, N, load_weights=True)
