import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(
    (train_images, train_labels),
    (test_images, test_labels),
) = tf.keras.datasets.mnist.load_data()
# Using tensorflow online dataset library to load MNIST
train_images = train_images / 255.0
test_images = test_images / 255.0
# Convert into float type
num_train = train_images.shape[0]
num_test = test_images.shape[0]
img_shape = (test_images.shape[1], test_images.shape[2])
num_pixel = test_images.shape[1] * test_images.shape[2]
# Acquiring basic info
train_images = np.reshape(train_images, (num_train, num_pixel))
test_images = np.reshape(test_images, (num_test, num_pixel))
# Vectorizing 2-D images

num_chosen_test = 300
num_chosen_train = 10000
# Shrink dataset for faster validation of this model
train_index = np.random.choice(
    np.arange(num_train), size=num_chosen_train, replace=False
)
test_index = np.random.choice(
    np.arange(num_test), size=num_chosen_test, replace=False)
# Randomly acquire a INDEX
train_images = train_images[train_index]
train_labels = train_labels[train_index]
test_images = test_images[test_index]
test_labels = test_labels[test_index]
# Use INDEX to make slices


def KNN(train_labels, train_images, test_images, k):
    result = np.zeros((num_chosen_test), dtype="uint8")
    plt.style.use("dark_background")
    for i in range(num_chosen_test):
        X = np.reshape(
            np.tile(test_images[i],
                    num_chosen_train), (num_chosen_train, num_pixel)
        )
        # Create a matrix of repeating vectors of the i-th image to avoid using another loop
        X = (X - train_images) ** 2
        X = np.sum(X, axis=1)
        # Sum results
        topk = train_labels[(np.argsort(X))[0:k]]
        # Find nearest neighbors
        #show_list = train_images[np.argsort(X)[0:k]]
        # for j in range(k):
        #    plt.subplot(num_chosen_test, k, i * k + j + 1)
        #    plt.imshow(np.reshape(show_list[j] * 255, img_shape))
        #    plt.xticks([])
        #    plt.yticks([])
        # Output k-nearest images
        result[i] = np.argmax(np.bincount(topk))
        # Count & make predictions
    # plt.show()
    return result


k_range = range(2, 5)
accuracy = []
plt.style.use("dark_background")
for i in k_range:
    result = KNN(train_labels, train_images, test_images, i)
    diff = result - test_labels
    cur_accu = (np.bincount(diff)[0]) / num_chosen_test
    print("K = %d, accuracy = %f" % (i, cur_accu))
    accuracy.append(cur_accu)
