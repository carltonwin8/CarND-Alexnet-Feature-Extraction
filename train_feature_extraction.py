import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128

with open('./train.p', 'rb') as f:
    data = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'])
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features,(227,227))

fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape,stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
training_operation = opt.minimize(loss_op, var_list=[fc8W, fc8b])

correct_prediction = tf.equal(tf.arg_max(logits, 1), labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X, y, sess):
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        batch_x, batch_y = X[offset:end], y[offset:end]
        loss, accuracy = sess.run([loss_op, accuracy_operation], feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss /X.shape[0], total_accuracy / X.shape[0]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(epochs):
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})

        validation_accuracy, loss_acc = evaluate(X_val, y_val, sess)
        print("EPOCH {} ... Time {}".format(i+1, time.time() - t0))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
