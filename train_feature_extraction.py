import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open("train.p", mode='rb') as f:
    train = pickle.load(f)
X, y = train['features'], train['labels']
# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape,y_train.shape)
# TODO: Define placeholders and resize operation.
labels = tf.placeholder(tf.int64, (None))
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
resize = tf.image.resize_images(features,[227,227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
nb_classes = 43
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape,stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
print(logits, labels)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 128
def evaluate(X_data, y_data, sess):
    num_examples = X_data.shape[0]
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * batch_x.shape[0])
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
EPOCHS = 10
num_examples = X_train.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Training...")
    print()
    for i in range(EPOCHS):
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})

        validation_accuracy = evaluate(x_test, y_test, sess)
        print("EPOCH {} ... Time {}".format(i+1, time.time() - t0))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
