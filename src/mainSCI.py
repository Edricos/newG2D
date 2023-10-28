import numpy as np

x = 200
while True:
    if (720 - x) % 16 == 0 and (1280 - x) % 64 == 0:
        break
    x += 1
    if x > 500: break
print(x)


# %%

#  Copyright 2018 Algolux Inc. All Rights Reserved.
#  修改自dataset_util.py
import os
import cv2
import numpy as np

crop_size = 128


# crop_size = 0


def read_gated_image(base_dir, gta_pass, img_id, data_type, num_bits=10, scale_images=False,
                     scaled_img_width=None, scaled_img_height=None,
                     normalize_images=False):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir, gta_pass, 'gated%d_10bit' % gate_id)
        img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if data_type == 'real':
            img = img[crop_size:(img.shape[0] - crop_size), crop_size:(img.shape[1] - crop_size)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))

    img = np.concatenate(gated_imgs, axis=2)
    if normalize_images:
        mean = np.mean(img, axis=2, keepdims=True)
        std = np.std(img, axis=2, keepdims=True)
        img = (img - mean) / (std + np.finfo(float).eps)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img, axis=0)


def read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance, scale_images=False,
                  scaled_img_width=None,
                  scaled_img_height=None, raw_values_only=False):
    if data_type == 'real':
        depth_lidar1 = np.load(
            os.path.join(
                base_dir, gta_pass, "depth_hdl64_gated_compressed",
                img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size: (depth_lidar1.shape[0] - crop_size),
                       crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        # 数组的形状将变为(1, height, width, 1)
        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
            np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)
        # return depth_lidar1, gt_mask

    img = np.load(os.path.join(base_dir, gta_pass, 'depth_compressed', img_id + '.npz'))['arr_0']

    if raw_values_only:
        return img, None

    img = np.clip(img, min_distance, max_distance) / max_distance
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)

    return np.expand_dims(np.expand_dims(img, axis=2), axis=0), None

# %%
#测试数据读取

import matplotlib.pyplot as plt

base_dir = '../data/real'
gta_pass = ''
data_type = 'real'
img_id = '01151'
min_distance = 3.
max_distance = 150.

# 从数据文件中读取目标图像和激光雷达掩码
in_img = read_gated_image(base_dir, gta_pass, img_id, data_type)
gt_patch, lidar_mask = read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance,
                                     # raw_values_only=True
                                     )

# 测试保存concatenate后的数据
in_dir = '../data/real/gated'
depth_dir = '../data/real/depth'
np.save(in_dir + img_id + '.npy', in_img)
laod_in_img = np.load(in_dir + img_id + '.npy')

figShape = [720 - crop_size * 2, 1280 - crop_size * 2]
# 创建一个2x2的子图布局
plt.figure(figsize=(20, 20))
fig, axs = plt.subplots(2, 3)
# 在子图上绘图
axs[0, 0].set_title('Gated 1')
axs[0, 0].imshow(laod_in_img[:, :, :, 0:1].reshape(figShape))
axs[0, 0].axis('off')

axs[0, 1].set_title('Gated 2')
axs[0, 1].imshow(in_img[:, :, :, 1:2].reshape(figShape))
axs[0, 1].axis('off')

axs[0, 2].set_title('Gated 3')
axs[0, 2].imshow(in_img[:, :, :, 2:3].reshape(figShape))
axs[0, 2].axis('off')

y_coords, x_coords = np.where(lidar_mask.reshape(figShape))
valid_depths = np.where(lidar_mask, gt_patch, np.nan)
axs[1, 0].set_title('Sparse Lidar Depth Map')
axs[1, 0].imshow(np.zeros_like(gt_patch.reshape(figShape)), cmap='gray')
axs[1, 0].scatter(x_coords, y_coords, c=1. / (gt_patch.reshape(figShape)[lidar_mask.reshape(figShape)]), cmap='jet',
                  s=1, marker='o')
axs[1, 0].axis('off')

RGBimg = cv2.imread(os.path.join(base_dir, 'rgb_left_8bit', img_id + '.png'))
axs[1, 1].set_title('RGB')
axs[1, 1].imshow(RGBimg)
axs[1, 1].axis('off')
axs[1, 2].axis('off')
# 调整子图之间的间距
plt.tight_layout()
plt.show()

# %%

import tensorflow as tf
import os
import cv2
import numpy as np

base_dir = '../data/real'
gta_pass = ''
data_type = 'real'
img_id = '01151'
min_distance = 3.
max_distance = 150.

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

data_dir = '../data/real'
# sample_ids = [f.split('.')[0] for f in os.listdir(os.path.join(data_dir, 'gated0_10bit'))]

def readlines(filename):
    """
        Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

data_fpath = r'C:\Users\edric\PycharmProjects\newG2D\data\real\splits\data'
# 数据加载
fpath = os.path.join(data_fpath, r"{}_files.txt")
sample_ids = readlines(fpath.format(r"train"))
# sample_ids = readlines('../data/real/splits/data/train_files.txt')


def load_sample(sample_id):
    if isinstance(sample_id, tf.Tensor):
        sample_id = sample_id.numpy().decode('utf-8')
    input1_path = os.path.join(data_dir, 'gated0_10bit', sample_id + '.png')
    input2_path = os.path.join(data_dir, 'gated1_10bit', sample_id + '.png')
    input3_path = os.path.join(data_dir, 'gated2_10bit', sample_id + '.png')
    depth_path = os.path.join(data_dir, 'depth_hdl64_gated_compressed', sample_id + '.npz')

    # 设置阈值和normalizer
    threshold = 2 ** 10 - 1
    normalizer_value = 2 ** 10 - 1
    # 设置裁剪大小
    crop_size = 128

    input1 = tf.image.decode_png(tf.io.read_file(input1_path), channels=1)
    input1 = tf.cast(input1, tf.float32)
    # 获取图像的高度和宽度
    height, width, _ = input1.shape
    # 使用tf.image.crop_to_bounding_box裁剪图像
    input1 = tf.image.crop_to_bounding_box(input1, crop_size, crop_size,
                                           height - 2 * crop_size, width - 2 * crop_size)
    normalizer = tf.constant(normalizer_value, dtype=tf.float32)
    # 使用tf.where处理大于阈值的像素
    img_tensor = tf.where(input1 > threshold, normalizer, input1)
    # 将图像张量转换为float32并除以normalizer
    input1 = tf.cast(img_tensor, tf.float32) / normalizer

    input2 = tf.image.decode_png(tf.io.read_file(input2_path), channels=1)
    input2 = tf.cast(input2, tf.float32)
    # 获取图像的高度和宽度
    height, width, _ = input2.shape
    # 使用tf.image.crop_to_bounding_box裁剪图像
    input2 = tf.image.crop_to_bounding_box(input2, crop_size, crop_size,
                                           height - 2 * crop_size, width - 2 * crop_size)
    normalizer = tf.constant(normalizer_value, dtype=tf.float32)
    # 使用tf.where处理大于阈值的像素
    img_tensor = tf.where(input2 > threshold, normalizer, input2)
    # 将图像张量转换为float32并除以normalizer
    input2 = tf.cast(img_tensor, tf.float32) / normalizer

    input3 = tf.image.decode_png(tf.io.read_file(input3_path), channels=1)
    input3 = tf.cast(input3, tf.float32)
    # 获取图像的高度和宽度
    height, width, _ = input3.shape
    # 使用tf.image.crop_to_bounding_box裁剪图像
    input3 = tf.image.crop_to_bounding_box(input3, crop_size, crop_size,
                                           height - 2 * crop_size, width - 2 * crop_size)
    normalizer = tf.constant(normalizer_value, dtype=tf.float32)
    # 使用tf.where处理大于阈值的像素
    img_tensor = tf.where(input3 > threshold, normalizer, input3)
    # 将图像张量转换为float32并除以normalizer
    input3 = tf.cast(img_tensor, tf.float32) / normalizer

    depth = np.load(depth_path)['arr_0']
    depth = depth[crop_size: (depth.shape[0] - crop_size), crop_size: (depth.shape[1] - crop_size)]
    depth = np.expand_dims(depth, axis=2)  #改为[height, width, 1]
    gt_mask = (depth > 0.)
    depth = np.float32(np.clip(depth, min_distance, max_distance) / max_distance)
    y_true_with_mask = tf.concat([depth, gt_mask], axis=-1)

    inputs = tf.stack([input1, input2, input3], axis=-1)
    inputs = tf.squeeze(inputs, axis=-2)  #将数据形状更改为[height, width, 3],以适应网络
    return inputs, y_true_with_mask


inputs1, depth1 = load_sample(sample_ids[25])


dataset = tf.data.Dataset.from_tensor_slices(sample_ids)
dataset = dataset.map(
    lambda x: tf.py_function(
        load_sample, [x], [tf.float32, tf.float32]),
    # num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# dataset = (dataset.shuffle(1024).batch(4).prefetch(tf.data.experimental.AUTOTUNE))
dataset = (dataset.shuffle(1024).batch(1))


sample_ids_val = readlines(fpath.format("val"))
dataset_val = tf.data.Dataset.from_tensor_slices(sample_ids_val)
dataset_val = dataset_val.map(
    lambda x: tf.py_function(
        load_sample, [x], [tf.float32, tf.float32]),
)
dataset_val = (dataset_val.shuffle(1024).batch(1))

num_samples = len(sample_ids)  # 这应该是14227
num_test_samples = int(0.1 * num_samples)  # 10%的样本作为测试集
# test_dataset = dataset.take(num_test_samples)
# train_dataset = dataset.skip(num_test_samples)

# %%

#测试unet
import unet
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=9216)])
    except RuntimeError as e:
        print(e)

def rmse(y_true, y_combined_pred):
    y_pred = y_combined_pred[..., -1:]
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps))*150

def mean_relative_error(y_true, y_combined_pred):
    y_pred = y_combined_pred[..., -1:]
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]
    relative_error = tf.abs((y_pred - y_true) / (y_true + np.finfo(float).eps))
    return tf.reduce_sum(relative_error * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps)

def mae(y_true, y_combined_pred):
    y_pred = y_combined_pred[..., -1:]
    # tf.print(" Shape of y_true:", tf.shape(y_true))
    # tf.print(" Shape of y_pred:", tf.shape(y_pred))
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]  # 移除gt_mask以得到真实的y_true

    l1_loss = tf.reduce_sum(tf.abs(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps)
    return l1_loss*150

def mae_loss(y_true, y_combined_pred):
    y_pred = y_combined_pred[..., -1:]
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]  # 移除gt_mask以得到真实的y_true

    l1_loss = tf.reduce_sum(tf.abs(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps)
    return l1_loss

def tv_loss(input, g_output):
    dy_out, dx_out = tf.image.image_gradients(g_output)
    dy_out = tf.abs(dy_out)
    dx_out = tf.abs(dx_out)
    dy_input, dx_input = tf.image.image_gradients(tf.reduce_mean(input, axis=3, keepdims=True))
    ep_dy = tf.exp(-tf.abs(dy_input))
    ep_dx = tf.exp(-tf.abs(dx_input))
    grad_loss = tf.reduce_mean(tf.multiply(dy_out, ep_dy) + tf.multiply(dx_out, ep_dx))
    return grad_loss

def custom_l1_loss_lamda(y_true, y_combined_pred):
    y_pred = y_combined_pred[..., -1:]  # 输出只有一个通道
    inputs = y_combined_pred[..., :-1]
    # tf.print(" Shape of y_true:", tf.shape(y_true))
    # tf.print(" Shape of y_pred:", tf.shape(y_pred))
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]  # 移除gt_mask以得到真实的y_true

    l1_loss = tf.reduce_sum(tf.abs(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps)
    # 平滑损失的权重
    smooth_weight = 0.3
    # loss = l1_loss + smooth_weight * smoothness_loss(y_pred)
    loss = l1_loss + smooth_weight * tv_loss(input=inputs, g_output=y_pred)
    return loss

input_img = tf.keras.Input(shape=(464, 1024, 3))
# input_img = tf.keras.Input(shape=(720, 1280, 3))
outputs=unet.build_unet(input_img)

# 1.使用Lambda层将输入和输出连接起来
combined_output = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))([input_img, outputs])
model = tf.keras.Model(inputs=input_img, outputs=combined_output)

# model = tf.keras.Model(inputs=input_img, outputs=unet.build_unet(input_img))
csv_logger = CSVLogger('training_log.csv', append=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
              loss=mae_loss,
              metrics=[rmse, mean_relative_error, mae])
model.summary()


# %%
import unet
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=9216)])
    except RuntimeError as e:
        print(e)


def rmse_c(y_true, pred):
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps))*150

def mae_loss_c(y_true, y_pred):
    gt_mask = y_true[..., -1:]
    y_true = y_true[..., :-1]  # 移除gt_mask以得到真实的y_true

    l1_loss = tf.reduce_sum(tf.abs(y_pred - y_true) * gt_mask) / (tf.reduce_sum(gt_mask) + np.finfo(float).eps)
    return l1_loss

# 1. 准备数据
x_train = tf.constant([[1.], [2.], [3.], [4.]], dtype=tf.float32)
y_train = tf.constant([[0.], [-1.], [-2.], [-3.]], dtype=tf.float32)

# 2. 定义模型
input_img = tf.keras.Input(shape=(464, 1024, 3))
# input_img = tf.keras.Input(shape=(720, 1280, 3))
outputs=unet.build_unet(input_img)

model = tf.keras.Model(inputs=input_img, outputs=outputs)


# 4. 选择优化器
optimizer = tf.optimizers.SGD(learning_rate=0.1)

train_mae_metric = tf.keras.metrics.Mean(name='train_MAE')
train_rmse_metric = tf.keras.metrics.Mean(name='train_RMSE')
# test_mae_metric = tf.keras.metrics.Mean(name='test_MAE')

# 5. 手动训练模型
for epoch in range(3):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = mae_loss_c(y_batch, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        batch_mae = mae_loss_c(y_batch, y_pred)
        batch_rmse = rmse_c(y_batch, y_pred)
        train_mae_metric(batch_mae)
        train_rmse_metric(batch_rmse)

    print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    template = 'Epoch {}, Train MAE: {}, Train RMSE: {}'
    print(template.format(epoch + 1, train_mae_metric.result(), train_rmse_metric.result()))



# %%


history = model.fit(dataset,
                    # validation_data=dataset_val,
                    epochs=3,
                    batch_size=1,
                    callbacks=[csv_logger])

# %%

model.save_weights('newG2D-weight_03.h5')

# %%

inputs, _ = load_sample(img_id)
# 扩展数据的维度以匹配模型的输入维度
inputs = tf.expand_dims(inputs, axis=0)
# 使用模型进行预测
predictions = model.predict(inputs)[..., -1:]
# 获取预测的深度图
predicted_depth = predictions[0, :, :, 0]
print(predicted_depth.shape)
# 使用matplotlib显示图像
plt.imshow(1. / predicted_depth, cmap='viridis')  # 使用viridis颜色映射
# plt.colorbar()
plt.title("Predicted Depth Map")
plt.axis('off')
plt.show()




