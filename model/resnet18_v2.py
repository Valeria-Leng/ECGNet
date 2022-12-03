import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

# 继承Layer,建立resnet18和34卷积层模块


class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()
# v1
        self.conv1 = Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')

        self.conv2 = Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
##################################################################################
        # self.bn1 = BatchNormalization()
        # self.relu1 = Activation('relu')
        # self.dropout1 = Dropout(rate=0.2)
        # self.conv1 = Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        # self.bn2 = BatchNormalization()
        # self.relu2 = Activation('relu')
        # self.dropout2 = Dropout(rate=0.2)
        # self.conv2 = Conv2D(filter_num, (3, 3), strides=1, padding='same')

        if stride != 1:
            self.residual = Conv2D(filter_num, (1, 1), strides=stride)
        else:
            self.residual = lambda x: x

    def call(self, inputs, training=None):
        # v1
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)
        ###########################################
        # x = self.bn1(inputs)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.conv1(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.conv2(x)
        # r = self.residual(inputs)
        # x = layers.add([x, r])
        # output = x

        return output


# 继承Model， 创建resnet18和34


class ResNet_v2(models.Model):
    def __init__(self, layers_dims, nb_classes, activation):
        super(ResNet_v2, self).__init__()

        self.stem = Sequential([
            Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])  # 开始模块

        self.layer1 = self.build_cellblock(64, layers_dims[0])
        self.layer2 = self.build_cellblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(512, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        # self.fc = Dense(nb_classes, activation='softmax')

        self.fc = Dense(nb_classes, activation = activation)
        # self.fc = Dense(nb_classes, activation='linear')

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride))  # 每层第一个block stride可能为非1

        for _ in range(1, blocks):  # 每一层由多少个block组成
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks


def build_ResNet(NetName, nb_classes, activation):
    ResNet_Config = {'ResNet18': [2, 2, 2, 2],
                     'ResNet34': [3, 4, 6, 3]}

    return ResNet_v2(ResNet_Config[NetName], nb_classes, activation)


def resnet18_v2(nb_classes, activation):
    model = build_ResNet('ResNet18', nb_classes, activation)
    # model = build_ResNet('ResNet34', 2)
    model.build(input_shape=(None, 5120, 12, 1))
    model.summary()

    return model


if __name__ == '__main__':
    resnet18_v2(2, activation='softmax')
