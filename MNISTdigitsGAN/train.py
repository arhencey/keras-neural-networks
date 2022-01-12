from prepare_data import get_data
from model import get_generator, get_discriminator, get_gan
import numpy as np

# select real samples from the dataset
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples. with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in the latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = np.zeros((n_samples, 1))
    return X, y

# train the generator and the discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(batch_per_epoch):
            # get randomly selected real samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate fake examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>{}, {}/{}, d1={:.3f}, d2={:.3f}, g={:.3f}'.format(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save('generator.h5')

def main():
    # load dataset
    dataset = get_data()

    # create models
    latent_dim = 100
    discriminator = get_discriminator()
    generator = get_generator(latent_dim)
    gan_model = get_gan(generator, discriminator)

    # train model
    train(generator, discriminator, gan_model, dataset, latent_dim)

if __name__=='__main__':
    main()
