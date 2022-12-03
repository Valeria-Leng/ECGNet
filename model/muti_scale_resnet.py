import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, AveragePooling2D, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# 继承Layer,建立resnet18和34卷积层模块


class CellBlock(layers.Layer):
    def __init__(self, filter_num, kernel_size, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num, kernel_size,
                            strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        # self.dropout1 = Dropout(rate=0.2)

        self.conv2 = Conv2D(filter_num, kernel_size, strides=1, padding='same')
        self.bn2 = BatchNormalization()

        # self.bn1 = BatchNormalization()
        # self.relu1 = Activation('relu')
        # self.conv1 = Conv2D(filter_num, kernel_size,
        #                     strides=stride, padding='same')
        # self.dropout1 = Dropout(rate=0.2)
        # self.bn2 = BatchNormalization()
        # self.relu2 = Activation('relu')
        # self.conv2 = Conv2D(filter_num, kernel_size, strides=1, padding='same')

        if stride != 1:
            # self.mp = AveragePooling2D((1, 1), strides=1, padding='same')
            # self.mp = MaxPooling2D((1, 1), strides=1, padding='same')
            self.residual = Conv2D(filter_num, (1, 1), strides=stride)
        else:
            # self.mp = lambda x: x
            self.residual = lambda x: x

    def call(self, inputs, training=True):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # r = self.mp(inputs)
        r = self.residual(inputs)

        x = layers.add([x, r])
        x = tf.nn.relu(x)

        # x = self.bn1(inputs)
        # x = self.relu1(x)
        # x = self.conv1(x)
        # # x = self.dropout1(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.conv2(x)

        # r = self.mp(inputs)
        # r = self.residual(r)

        # x = layers.add([x, r])
        # x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, 0.2)

        return x


class Route(layers.Layer):
    def __init__(self, kernel_size, layers_dims):
        super(Route, self).__init__()
        self.layers1 = self.build_cellblock(
            64, layers_dims[0], kernel_size)
        self.layers2 = self.build_cellblock(
            128, layers_dims[1], kernel_size, stride=2)
        self.layers3 = self.build_cellblock(
            256, layers_dims[2], kernel_size, stride=2)
        self.layers4 = self.build_cellblock(
            512, layers_dims[3], kernel_size, stride=2)

        # self.mp = MaxPooling2D((2, 2), strides=2, padding='same')
#
        self.avgpool = GlobalAveragePooling2D()

    def call(self, x, training=None):
        # x = self.stem(inputs)
        # print(x.shape)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        # x = self.mp(x)
        x = self.avgpool(x)
        # x = self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, kernel_size, stride=1):
        res_blocks = Sequential()
        # 每层第一个block stride可能为非1
        res_blocks.add(CellBlock(filter_num, kernel_size, stride))
        for _ in range(1, blocks):  # 每一层由多少个block组成
            res_blocks.add(CellBlock(filter_num, kernel_size, stride=1))

        return res_blocks
# 继承Model， 创建resnet18和34


class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes, activation):
        super(ResNet, self).__init__()
        # print(layers_dims[1])
        self.stem = Sequential([
            Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])  # 开始模块

        # self.Route1 = Route((3, 3), layers_dims)
        # self.Route2 = Route((5, 5), layers_dims)
        # self.Route3 = Route((7, 7), layers_dims)
        self.Route1 = Route((3, 3), layers_dims)
        self.Route2 = Route((5, 3), layers_dims)
        self.Route3 = Route((7, 3), layers_dims)
        # self.avgpool = GlobalAveragePooling2D()
        # self.dropout = Dropout(rate=0.2)
        # self.fc1 = Dense(128)
        # self.fc2 = Dense(64)
        self.fc3 = Dense(nb_classes, activation=activation)  # 分类
        # self.fc3 = Dense(nb_classes, activation='linear')  # 回归

    def call(self, inputs, training=True):
        x = self.stem(inputs)
        # print(x.shape)

        x1 = self.Route1(x)
        x2 = self.Route2(x)
        x3 = self.Route3(x)

        x = tf.concat((x1, x2, x3), 1)
        # x = tf.concat((x1, x2), 1)
        # x = self.dropout(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)

        return x

    # def build_cellblock(self, filter_num, blocks, stride=1):
    #     res_blocks = Sequential()
    #     res_blocks.add(CellBlock(filter_num, stride))  # 每层第一个block stride可能为非1

    #     for _ in range(1, blocks):  # 每一层由多少个block组成
    #         res_blocks.add(CellBlock(filter_num, stride=1))

    #     return res_blocks


def build_ResNet(NetName, nb_classes, activation):
    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3],
                     'Multi_scale_Resnet': [2, 2, 2, 2]}

    return ResNet(ResNet_Config[NetName], nb_classes, activation)


def Multi_scale_Resnet(nb_classes, activation):
    model = build_ResNet('Multi_scale_Resnet', nb_classes, activation=activation)
    # model = build_ResNet('Multi_scale_Resnet', 1)
    model.build(input_shape=(None, 5120, 12, 1))
    model.summary()
    return model


if __name__ == '__main__':
    model = Multi_scale_Resnet(nb_classes=2, activation='softmax')
    # model.summary()
