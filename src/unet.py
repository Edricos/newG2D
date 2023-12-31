import tensorflow as tf


# 定义Leaky ReLU激活函数
def lrelu(x):
    # return tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return tf.maximum(x * 0.2, x)


# 上采样并连接函数
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    # 定义反卷积的滤波器
    deconv_filter = tf.Variable(
        tf.random.truncated_normal(
            [pool_size, pool_size, output_channels, in_channels],
            stddev=0.02)
    )
    # 执行反卷积操作
    deconv = tf.keras.layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=pool_size,
        strides=pool_size,
        padding='same',
        kernel_initializer=tf.keras.initializers.Constant(value=deconv_filter)
    )(x1)
    # deconv = tf.keras.layers.Conv2DTranspose(
    #     filters=output_channels,
    #     kernel_size=(deconv_filter.shape[0], deconv_filter.shape[1]),
    #     strides=pool_size,
    #     padding='same'
    # )(x1)
    # 将反卷积的结果与x2连接
    deconv_output = tf.concat([deconv, x2], axis=-1)
    return deconv_output


# 构建U-Net网络结构
def build_unet(input_img):
    pool_size = 2

    # 第一层卷积
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(input_img)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv1)

    # 第二层卷积
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv2)

    # 第三层卷积
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv3)

    # 第四层卷积
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv4)

    # 第五层卷积
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv5)

    # 上采样并连接
    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, kernel_initializer='he_uniform', padding='same')(conv9)

    # 输出层
    out = tf.keras.layers.Conv2D(1, (1, 1), activation=None, padding='same')(conv9)

    # return out, up8, up7
    return out



def network(input_img, use_multi_scale, use_3dconv=False):
    # 检查可用的 GPU
    # # Tensorflow 2.13 支持 cuDNN：8.6、CUDA：11.8，
    # # 并且 TensorFlow 2.10 是在本地 Windows 上支持 GPU 的最后一个 TensorFlow 版本。
    # # 从 TensorFlow 2.11 开始，需要在 WSL2 中安装 TensorFlow 才能使用 GPU。
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10096)])
      except RuntimeError as e:
        print(e)

    # 测试
    # input_img = tf.keras.Input(shape=(1280, 720, 3))
    # model = tf.keras.Model(inputs=input_img, outputs=build_unet(input_img))
    # model.summary()

    if use_multi_scale:
        print('Using mult-scale')
        if use_3dconv:
            print('Using 3D convs')
            # out, conv8, conv7 = build_3d_conv_unet(input_img)
        else:
            out, conv8, conv7 = build_unet(input_img)
        out2 = tf.keras.layers.Conv2D(1,[3, 3], padding='SAME',
                                      activation=lrelu, bias_initializer=None, dilation_rate=1)(conv8)
        out3 = tf.keras.layers.Conv2D(1,[3, 3], padding='SAME',
                                      activation=lrelu, bias_initializer=None, dilation_rate=1)(conv7)
        out = {'output': out, 'half_scale': out2, 'fourth_scale': out3}
        return out

    else:
        print('not using multi-scale')
        if use_3dconv:
            print('Using 3D convs')
            # out, _, _ = build_3d_conv_unet(input_img)
        else:
            out, _, _ = build_unet(input_img)
        out = {'output': out}
        return out




