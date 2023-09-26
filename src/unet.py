import tensorflow as tf


def lrelu(x):
    return tf.keras.layers.LeakyReLU(alpha=0.2)(x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(
        tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=pool_size, strides=pool_size,
                                             padding='same',
                                             kernel_initializer=tf.keras.initializers.Constant(value=deconv_filter))(x1)
    deconv_output = tf.concat([deconv, x2], axis=-1)
    return deconv_output


def build_unet(input_img):
    pool_size = 2

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(input_img)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv9)

    out = tf.keras.layers.Conv2D(1, (1, 1), activation=None, padding='same')(conv9)

    return out


# 检查可用的 GPU
# Tensorflow 2.13 支持 cuDNN：8.6、CUDA：11.8，
# 并且 TensorFlow 2.10 是在本机 Windows 上支持 GPU 的最后一个 TensorFlow 版本。
# 从 TensorFlow 2.11 开始，您需要在 WSL2 中安装 TensorFlow 才能使用 GPU。
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10096)])
  except RuntimeError as e:
    print(e)


input_img = tf.keras.Input(shape=(None, None, 3))
model = tf.keras.Model(inputs=input_img, outputs=build_unet(input_img))
model.summary()
