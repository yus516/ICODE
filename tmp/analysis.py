import matplotlib.pyplot as plt
import numpy as np

def downsample_matrix(matrix):
    # Ensure the matrix is 136x136
    assert matrix.shape == (136, 136), "Input matrix must be 136x136"

    # Sum adjacent rows and columns
    new_matrix = matrix[:68, :68] + matrix[68:, :68] + matrix[:68, 68:] + matrix[68:, 68:]

    return new_matrix

def my_norm(matrix):
    # print(np.min(matrix), np.max(matrix))
    result = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    print(np.min(result), np.max(result))
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

# Load the matrix
a_hat_l_0 = np.load("tmp/ramp/cyberattck9_ramp_9_1_agc_a_hat_l_1-68_variable_0.npy")

# Ensure the shape is (136, 136)
assert a_hat_l_0.shape == (136, 136), "Matrix must be 136x136"

# Downsample by summing every 2x2 block
dm = downsample_matrix(a_hat_l_0)


# Plot the downsampled matrix
plt.figure(figsize=(8, 8))
plt.imshow(dm, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Sum of 2x2 blocks")
plt.title("Downsampled 68x68 Matrix")
plt.show()

plt.savefig('tmp/ramp/foo.png', bbox_inches='tight')


plt.figure()
fig, ax = plt.subplots(1, 10, figsize=(20,2))

ll = []

for i in range(10):

    a_hat_l_ = np.load("tmp/ramp/cyberattck9_ramp_9_1_agc_a_hat_l_1-68_variable_" + str(i) + ".npy")

    # Ensure the shape is (136, 136)
    assert a_hat_l_.shape == (136, 136), "Matrix must be 136x136"

    # Downsample by summing every 2x2 block
    downsampled_matrix = downsample_matrix(a_hat_l_)
    flat_arr = downsampled_matrix.flatten()
    # Plot the downsampled matrix
    thresh = np.partition(flat_arr, -300)[-300]
    ax[i].imshow((downsampled_matrix > thresh) * 1, cmap="viridis", interpolation="nearest")
plt.savefig('tmp/ramp/root-cause-graph.png', bbox_inches='tight')

plt.figure()
fig, ax = plt.subplots(1, 10, figsize=(20,2))

dm1 = my_norm(dm)

for i in range(10):

    a_hat_l_ = np.load("tmp/ramp/cyberattck9_ramp_9_1_agc_a_hat_l_1-68_variable_" + str(i) + ".npy")

    # Ensure the shape is (136, 136)
    assert a_hat_l_.shape == (136, 136), "Matrix must be 136x136"

    # Downsample by summing every 2x2 block
    downsampled_matrix = downsample_matrix(a_hat_l_)
    downsampled_matrix1 = my_norm(downsampled_matrix)

    # Plot the downsampled matrix
    l = np.sum(np.abs(dm1 - downsampled_matrix1), axis=0)
    print(l.shape)
    ll.append(l[8])
    ax[i].plot(l)
plt.savefig('tmp/ramp/root-cause-new.png', bbox_inches='tight')

plt.figure()
plt.plot(ll)
plt.savefig('tmp/ramp/root-cause-change-by-time.png', bbox_inches='tight')