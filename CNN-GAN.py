import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Set hyperparameters
batch_size = 100
z_dimensions = 100
# Define placeholders for input images and noise vector
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z_placeholder = tf.placeholder(tf.float32, shape=[None, z_dimensions])
# Create generator and discriminator networks
def generator(z, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # First convolutional layer
        x1 = tf.layers.conv2d_transpose(z, 64, 4, strides=1, padding='valid')
        x1 = tf.layers.batch_normalization(x1, training=True)
        x1 = tf.nn.leaky_relu(x1)
        # 4x4x64 now
        x2 = tf.layers.conv2d_transpose(x1, 32, 4, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.nn.leaky_relu(x2)
        # 7x7x32 now
        x3 = tf.layers.conv2d_transpose(x2, 1, 4, strides=2, padding='same')
        # 14x14x1 now
        # Output layer
        logits = tf.tanh(x3)
        return logits
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Input layer is 28x28x1
        x1 = tf.layers.conv2d(x, 32, 4, strides=2, padding='same')
        x1 = tf.nn.leaky_relu(x1)
        # 14x14x32
        x2 = tf.layers.conv2d(x1, 64, 4, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.nn.leaky_relu(x2)
        # 7x7x64
        x3 = tf.layers.conv2d(x2, 128, 4, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.nn.leaky_relu(x3)
        # 4x4x128
        x4 = tf.layers.flatten(x3)
        logits = tf.layers.dense(x4, 1)
        # Output layer
        return logits
# Generate random noise vector for generator input
def generate_noise(batch_size, z_dimensions):
    return np.random.normal(0, 1, size=[batch_size, z_dimensions])
# Generate fake images using generator
generated_images = generator(z_placeholder)
# Get logits for real and fake images using discriminator
real_logits = discriminator(x_placeholder)
fake_logits = discriminator(generated_images, reuse=True)
# Define loss functions for generator and discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
# Define optimizers for generator and discriminator
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'Discriminator' in var.name]
g_vars = [var for var in tvars if 'Generator' in var.name]
d_train_step = tf.train.AdamOptimizer(0.0003).minimize(d_loss, var_list=d_vars)
g_train_step = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
# Define accuracy calculation
correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(tf.ones_like(real_logits), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initialize global variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Train the GAN
for i in range(5000):
    # Get real images and generate noise
    batch = mnist.train.next_batch(batch_size)
    real_images = batch[0].reshape([batch_size, 28, 28, 1])
    noise = generate_noise(batch_size, z_dimensions)
    # Train discriminator and generator
    sess.run(d_train_step, feed_dict={x_placeholder: real_images, z_placeholder: noise})
    sess.run(g_train_step, feed_dict={z_placeholder: noise})
    # Print losses and accuracy every 100 steps
    if i %! (MISSING)== 0:
        dLoss, gLoss, acc = sess.run([d_loss, g_loss, accuracy], feed_dict={x_placeholder: real_images, z_placeholder: noise})
        print("Iteration {}: Discriminator Loss: {:.4f}, Generator Loss: {:.4f}, Accuracy: {:.4f}".format(i, dLoss, gLoss, acc))
# Test the generator by generating a sample image
noise = generate_noise(1, z_dimensions)
generated_image = sess.run(generated_images, feed_dict={z_placeholder: noise})
generated_image = generated_image.reshape([28, 28])
plt.imshow(generated_image, cmap='Greys')
plt.show()
