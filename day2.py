# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# %%
cifar_10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_10.load_data()

# %%
# 共10类图片，0-9如下：
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# %%
# 训练集包含60000张32*32的图片
train_images.shape
# %%
# 标签0-9，60000张图一共60000个没啥说的
len(train_labels)
# %%
train_labels
# %%
# 画张图看看，像素的范围为[0, 255]
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# 像素归一化，归一化后的像素范围为[0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0
# 肉眼可见图变灰了
plt.figure(figsize=(10, 10))
train_labels_squeezed = np.squeeze(train_labels)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    print(train_labels_squeezed[i])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels_squeezed[i]])
    
plt.show()
# %%
# 进行一个极其简单的神经网络的搭建
# Flatten层把二维的输入转为一维向量
# Dense层即全连接层，第一层Dense作为隐藏层，激活函数为relu
# 最后一层Dense作为输出，输出每个类别的概率
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
# %%
# 编译模型
# 优化器选择adam（不知道选啥的时候用adam就完事了）
# 损失函数选用SparseCategoricalCrossentropy
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
# fit模型，开始训练啦！
cnn_model.fit(train_images, train_labels, epochs=10, batch_size=32)
# %%
# 测试模型
test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# %%
# 这里直接接一个softmax把输出转化成概率
probability_model = tf.keras.Sequential([cnn_model, 
                                         tf.keras.layers.Softmax()])
# %%
predictions = probability_model.predict(test_images)
# %%
predictions[0]
# %%
# argmax把输出最大概率的元素，得到结果
np.argmax(predictions[0])
# %%
test_labels[0]
# %%
# 下面都是可视化，感兴趣的自己看看
def plot_image(i, predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = img[i]  # 选择第i张图像

    plt.imshow(img, cmap=None)

    predicted_label = np.argmax(predictions_array)
    true_label = true_label[i]  # 提取标量整数值
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    predicted_class = class_names[predicted_label]
    predicted_prob = predictions_array[predicted_label] * 100

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_class, predicted_prob, class_names[int(true_label)]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    bar_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    bar_plot[predicted_label].set_color('red')
    bar_plot[int(true_label)].set_color('blue')


# 显示第一张图像的预测结果
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels[i])
plt.show()

# 显示第12张图像的预测结果
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels[i])
plt.show()
# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels[i])
plt.tight_layout()
plt.show()
# %%
# Code here!
