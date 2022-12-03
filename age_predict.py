import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
from model.resnet18_v2 import resnet18_v2
from model.DNN import get_model
from model.muti_scale_resnet import Multi_scale_Resnet
import argparse
from model.CNN import buildModel_2DCNN
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from gender_train import spilt_train_test
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--path_dir', default='./save_weight/age/')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--image_dir', default='./images/')
args = parser.parse_args()




# test_path = 'data/X_test.txt'
records_path = './records/'
csv_path = './records_normal_raw.csv'

def coeff_determination(y_true, y_pred):

    SS_res = K.sum(K.square(y_true-y_pred))

    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res/(SS_tot + K.epsilon()))

def load_test(df_train):
    age = np.array(df_train.Age/100.0, dtype=np.float32)
    age = tf.convert_to_tensor(age, dtype=tf.float32)
    with open('./data/df_test.txt', 'rb') as f:
        ecg_test = pickle.load(f)
    return ecg_test, age

def load_age_model(model_name, model_path):
    df_train, df_val, df_test = spilt_train_test(csv_path)
    ecg_test, age_test = load_test(df_test)
   

 
    print(ecg_test.shape, age_test.shape)
    new_model = model_name
    new_model.load_weights(model_path)
    predicted = new_model.predict(ecg_test)
    new_model.compile(optimizer='adam',
                      loss='mse', metrics=['mae', coeff_determination])

    score = new_model.evaluate(ecg_test, age_test)

    print(score)
    # #####################################################
    # plt.figure()
    # plt.scatter(age_test*100, predicted*100)
    # x = np.linspace(0, 1, 100)
    # y = x
    # plt.plot(x*100, y*100, color='red', linewidth=1.0,
    #             linestyle='--', label='line')
    # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # # plt.legend(["y = x"])
    # plt.title("Multi-Scale-ResNet Estimated Age vs Chronological Age")
    # plt.xlabel('Chronological Age (years)')
    # plt.ylabel('Multi-Scale-ResNet Estimated Age (years)')
    # plt.savefig('./images/test_MultiScaleResNet' + '.png', dpi=200,
    #             bbox_inches='tight', transparent=False)
    #####################################################


if __name__ == '__main__':
 

    # load weights
    load_age_model(
        buildModel_2DCNN(1, last_layer='linear'), './save_weight/age/CNN_8.366568386554718_model.h5')
    load_age_model(
        get_model(1, last_layer='linear'), './save_weight/age/DNN_7.848995178937912_model.h5')
    load_age_model(
        resnet18_v2(1, activation='linear'), './save_weight/age/ResNet_7.5802043080329895_model.h5')
    load_age_model(
        Multi_scale_Resnet(1, activation='linear'), './save_weight/age/MultiScaleResNet_7.520367205142975_model.h5')

   
