import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from train import generate_latent_points

def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    #plt.show()
    plt.savefig('generated_digits.png')

# load model
model = load_model('cgan_generator.h5')
# generate images
latent_points, labels = generate_latent_points(100, 100)
# specify labels
labels = np.asarray([x for _ in range(10) for x in range(10)])
# generate images
X = model.predict([latent_points, labels])
# scale from [-1, 1] to [0, 1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 10)

