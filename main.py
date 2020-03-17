import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data',one_hot = True)
train_data = mnist.train.images
train_label = mnist.train.labels
test_data = mnist.test.images
test_label = mnist.test.labels

# 每张图片的像素大小是 28 * 28，在提供的数据集中已经转为 1 * 784（28 * 28）的向量,方便矩阵乘法处理
y = tf.compat.v1.placeholder(tf.float32, [None, 784])
# 输出是每一张图1*10 的one-hot向量
b = tf.compat.v1.placeholder(tf.float32, [None, 10])



Weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))



# 思考输出的a的shape：n * 10, n时输入的数据的数量
a = tf.compat.v1.nn.softmax(tf.matmul(y, Weight) + bias)

# 损失函数
loss =  tf.compat.v1.norm(b-tf.matmul(y, Weight) - bias,ord=2)
# 使用梯度下降的方法进行参数优化
learning_rate = 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 判断是否预测结果与正确结果是否一致
# predres 数组的输出是 [False, True...] 类似格式
predres = tf.compat.v1.equal(tf.argmax(a, 1), tf.argmax(b, 1))

# 计算正确率
#使用cast转化为浮点数 则 True会被转化为 1.0, False 0.0
# 这些数据求均值,这个均值表示所有数据中有多少个1 ，也就是正确个数
accruracy = tf.compat.v1.reduce_mean(tf.cast(predres, tf.float32))

init_param = tf.global_variables_initializer()


train_epochs = 100
batch_size = 100

with tf.compat.v1.Session() as sess:
    sess.run(init_param)

    for epoch in range(train_epochs):
        print("epoch:" + str(epoch))
        avg_loss = 0.
        # 计算训练数据可以划分为多少个batch大小的组
        batch_num = int(mnist.train.num_examples / batch_size)


        for i in range(batch_num):
            # mnist.train.next_batch实现：第i个循环取第i个batch_size的数据
            batch_y, batch_b = mnist.train.next_batch(batch_size)

            feed_dict = {y: batch_y, b: batch_b}
            sess.run(optimizer, feed_dict)
            output_dict = {y: batch_y, b: batch_b}
            # 累计计算总的损失值
            avg_loss += sess.run(loss, feed_dict=output_dict) / batch_num



        # feed_train = {y: train_data[1: 100], b: train_label[1: 100]}
        feedt_test = {y: mnist.test.images, b: mnist.test.labels}

        train_acc = sess.run(accruracy, feed_dict=train_data)
        test_acc = sess.run(accruracy, feed_dict=feedt_test)

        print("loss: %.9f train_acc: %.3f test_acc: %.3f" %
              (avg_loss, train_acc, test_acc))