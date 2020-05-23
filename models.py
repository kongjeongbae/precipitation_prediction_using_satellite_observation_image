from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate, Input, GlobalAveragePooling2D, add, AveragePooling2D
from tensorflow.keras import Model


def build_model(input_layer, start_neurons):
    DROP_OUT = 0.01
    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu",
                   padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(DROP_OUT)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(DROP_OUT)(pool2)

    # 10 x 10
    convm = Conv2D(start_neurons * 4, (3, 3),
                   activation="relu", padding="same")(pool2)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DROP_OUT)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3),
                    activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3),
                    activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DROP_OUT)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3),
                    activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3),
                    activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(DROP_OUT)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(uconv1)

    return output_layer


def build_model2(input_layer, start_neurons):
    DROP_OUT = 0.3

    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu",
                   padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(DROP_OUT)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(DROP_OUT)(pool2)

    # 10 x 10 -> 5 x 5
    conv3 = Conv2D(start_neurons * 4, (3, 3),
                   activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3),
                   activation="relu", padding="same")(conv3)
    pool3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(pool3)
    pool3 = Dropout(DROP_OUT)(pool3)

    # 5x 5
    convm = Conv2D(start_neurons * 8, (3, 3),
                   activation="relu", padding="same")(pool3)

    # 5 x 5 -> 10 x 10
    deconv3 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DROP_OUT)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3),
                    activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3),
                    activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DROP_OUT)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3),
                    activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3),
                    activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DROP_OUT)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3),
                    activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3),
                    activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(DROP_OUT)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(uconv1)

    return output_layer


def resnet_model2(input_layer, start_neurons):
    # ResUnet 참조
    # https://www.researchgate.net/figure/Res-U-Net-architecture-a-basic-block-of-the-Stack-U-Net-model-Another-possible-basic_fig1_324859814
    # 위 논문 Fig 1. 에서는 BatchNormalization 도 안함

    # https://www.semanticscholar.org/paper/Weighted-Res-UNet-for-High-Quality-Retina-Vessel-Xiao-Lian/ff8b823d6f04a78bdb568a09139ef6d02111764e/figure/13
    # 이 논문에서도 BatchNorm 안함

    # 그런데 tensorflow Resnet layer를 보면 BatchNorm이 되어있기에 해보자.

    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu",
                   padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    pool1 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(pool1)
    add1 = add([conv2, pool1])
    conv3 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(add1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    add2 = add([conv3, add1])

    conv4 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(add2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    add3 = add([conv4, add2])
    conv5 = Conv2D(start_neurons * 2, (1, 1),
                   activation="relu", padding="same")(add3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(start_neurons * 8, (1, 1),
                   activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(start_neurons * 8, (1, 1),
                   activation="relu", padding="same")(add3)
    conv6 = BatchNormalization()(conv6)

    add4 = add([conv5, conv6])

    conv7 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(add4)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(start_neurons * 2, (1, 1),
                   activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(conv7)

    return output_layer


def res_u_net(input_layer, start_neurons):
    # ResUnet 참조
    # https://www.researchgate.net/figure/Res-U-Net-architecture-a-basic-block-of-the-Stack-U-Net-model-Another-possible-basic_fig1_324859814
    # 위 논문 Fig 1. 에서는 BatchNormalization 도 안함

    # https://www.semanticscholar.org/paper/Weighted-Res-UNet-for-High-Quality-Retina-Vessel-Xiao-Lian/ff8b823d6f04a78bdb568a09139ef6d02111764e/figure/13
    # 이 논문에서도 BatchNorm 안함

    # 그런데 tensorflow Resnet layer를 보면 BatchNorm이 되어있기에 해보자.

    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu",
                   padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)

    pool1 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(pool1)
    add1 = add([conv2, pool1])
    conv3 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(add1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    add2 = add([conv3, add1])

    conv4 = Conv2D(start_neurons * 1, (1, 1),
                   activation="relu", padding="same")(add2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 1, (3, 3),
                   activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)

    add3 = add([conv4, add2])
    conv5 = Conv2D(start_neurons * 2, (1, 1),
                   activation="relu", padding="same")(add3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(start_neurons * 2, (3, 3),
                   activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(start_neurons * 8, (1, 1),
                   activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(start_neurons * 8, (1, 1),
                   activation="relu", padding="same")(add3)
    conv6 = BatchNormalization()(conv6)

    add4 = add([conv5, conv6])
    conv7 = Conv2DTranspose(start_neurons * 2, (3, 3),
                            strides=(2, 2), padding="same")(add4)
    conv7 = Conv2D(start_neurons * 4, (1, 1),
                   activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(start_neurons * 2, (1, 1),
                   activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(conv7)

    return output_layer


def resnet_model(shape):
    inputs = Input(shape)

    bn = BatchNormalization()(inputs)
    conv0 = Conv2D(256, kernel_size=1, strides=1, padding='same',
                   activation='relu', kernel_initializer='he_normal')(bn)

    bn = BatchNormalization()(conv0)
    conv = Conv2D(128, kernel_size=2, strides=1, padding='same',
                  activation='relu', kernel_initializer='he_normal')(bn)
    concat = concatenate([conv0, conv], axis=3)

    bn = BatchNormalization()(concat)
    conv = Conv2D(64, kernel_size=3, strides=1, padding='same',
                  activation='relu', kernel_initializer='he_normal')(bn)
    concat = concatenate([concat, conv], axis=3)

    # 원래는 5였음 - 6분정도 걸리고 2로 줄이면 3분정도 걸림 /에폭당
    for i in range(9):
        bn = BatchNormalization()(concat)
        conv = Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu', kernel_initializer='he_normal')(bn)
        concat = concatenate([concat, conv], axis=3)

    bn = BatchNormalization()(concat)
    outputs = Conv2D(1, kernel_size=1, strides=1, padding='same',
                     activation='relu', kernel_initializer='he_normal')(bn)

    model = Model(inputs=inputs, outputs=outputs)

    return model

