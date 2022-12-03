import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
from datetime import datetime
from model.resnet18_v2 import resnet18_v2
from model.DNN import get_model
from model.muti_scale_resnet import Multi_scale_Resnet
import argparse
from gender_train import processing, buildModel_2DCNN
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report
from gender_train import spilt_train_test

parser = argparse.ArgumentParser()
parser.add_argument('--path_dir', default='./save_weight/age/')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--image_dir', default='./images/')
args = parser.parse_args()




# test_path = 'data/X_test.txt'
records_path = './records/'
csv_path = './records_normal_raw.csv'


def load_test(df_train):
    gender = np.array(df_train.Sex, dtype=np.float32)
    gender = tf.convert_to_tensor(gender, dtype=tf.float32)
    gender = tf.keras.utils.to_categorical(gender, num_classes=2)
    with open('./data/df_test.txt', 'rb') as f:
        ecg_test = pickle.load(f)
    return ecg_test, gender

def load_model(model_name, model_path, FPR, TPR, AUC):
    df_train, df_val, df_test = spilt_train_test(csv_path)
    ecg_test, gender_test = load_test(df_test)
   

    new_model = model_name
    new_model.load_weights(model_path)
    new_model.predict(ecg_test)
    ##################################################
    new_model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
    score = new_model.evaluate(ecg_test, gender_test)
    print(score)
    y_pred = new_model.predict(ecg_test)

    y_pro = [x[1] for x in y_pred]
    y_pred = np.argmax(y_pred, axis=-1)

    y_true = np.argmax(gender_test, axis=-1)
    # y_true = np.array(y_true)
    # print(y_true, y_pred)
    print(classification_report(y_true, y_pred, digits=4))
    fpr, tpr, thresholds = roc_curve(y_true, y_pro)
    area = auc(fpr, tpr)
    print('AUC=', area)
    FPR.append(fpr)
    TPR.append(tpr)
    AUC.append(area)
    return FPR, TPR, AUC


if __name__ == '__main__':
    FPR = []
    TPR = []
    AUC = []

    # load weights and calculate AUC
    load_model(
        buildModel_2DCNN(2, last_layer='softmax'), './save_weight/gender/CNN_86.70%_model.h5', FPR, TPR, AUC)
    load_model(
        get_model(2, last_layer='softmax'), './save_weight/gender/DNN_89.10%_model.h5', FPR, TPR, AUC)
    load_model(
        resnet18_v2(2, activation='softmax'), './save_weight/gender/Resnet_91.15%_model.h5', FPR, TPR, AUC)
    load_model(
        Multi_scale_Resnet(2, activation='softmax'), './save_weight/gender/MultiScaleResNet_91.95%_model.h5', FPR, TPR, AUC)
    print(FPR, TPR, AUC)

    #plot ROC curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(FPR[0], TPR[0], 'r',
             label='CNN (AUC = {:.3f})'.format(AUC[0]))
    plt.plot(FPR[1], TPR[1], 'b',
             label='DNN (AUC = {:.3f})'.format(AUC[1]))
    plt.plot(FPR[2], TPR[2], 'y',
             label='ResNet (AUC = {:.3f})'.format(AUC[2]))
    plt.plot(FPR[3], TPR[3], 'g',
             label='Multi-Scale-Resnet (AUC = {:.3f})'.format(AUC[3]))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    TIMESTAMP = "{0:%Y-%m-%dT%H:%M:%S}".format(datetime.now())
    print(TIMESTAMP)
    image_savedir = os.path.join('./images/','gender_' + TIMESTAMP + '_AUC.png')

    plt.savefig(image_savedir)
