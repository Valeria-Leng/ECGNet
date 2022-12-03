import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

import pandas as pd
import h5py
# import keras
import numpy as np
import pywt
from scipy.stats import zscore
import pickle
from datetime import datetime
from model.resnet18_v2 import resnet18_v2
from model.DNN import get_model
from model.muti_scale_resnet import Multi_scale_Resnet
from model.CNN import buildModel_2DCNN
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
import argparse
from tensorflow.keras.optimizers import SGD
# from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

records_path = './records/'
csv_path = './records_normal_raw.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--path_dir', default='./save_weight/gender/')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--log_dir', default='./log/')
# parser.add_argument('--image_dir', default='./images/')
args = parser.parse_args()


# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# exit()
# def main():
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)])
            # tf.config.gpu_options.per_process_gpu_memory_fraction = 0.5
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental

    except RuntimeError as e:
        print(e)
        exit(-1)


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
    
def spilt_train_test(path):
    #Read the first column marker of csv file 读csv文件存储第一列标号
    df = pd.read_csv(path)
    # print(df)
    df_train = df.groupby('Age_group', group_keys='False',as_index=False).apply(lambda x :x.sample(replace=False,random_state=1,frac=0.8))
    #Get the index of the training set and delete it from the original table to get the test set获取训练集的index后从原表中删去得到测试集
    train_index = df_train.index.droplevel()
    # print(train_index, df.index)
    df.drop(index=train_index, axis=0, inplace=True)
    df_test = df
    # print(df_test)
    #Divide the training set and validation set according to 8:2 进一步按照8：2划分训练集和验证集
    df_val = df_train.groupby('Age_group', group_keys='False',as_index=False).apply(lambda x :x.sample(replace=False,random_state=1,frac=0.2))
    # print(df_val)
    #drop多余的第一列索引
    # idv = df_val.index.levels[0]
    val_index = df_val.index.droplevel()
    # print(val_index)
    # print(df_train.index)
    df_train.drop(index = val_index, inplace=True)
    # print(df_train)
    return df_train, df_val, df_test

def records_pre (df_train, records_path):

    # print(df_train)

    # print(globals())

    Xsavedir = os.path.join('./data', namestr(df_train, globals())[0]+'.txt')
    
    # print(Xsavedir)
    
    gender = np.array(df_train.Sex, dtype=np.float32)
    gender = tf.convert_to_tensor(gender, dtype=tf.float32)
    gender = tf.keras.utils.to_categorical(gender, num_classes=2)

    age = np.array(df_train.Age/100.0, dtype=np.float32)
    age = tf.convert_to_tensor(age, dtype=tf.float32)

    # #####################################################################

    # ECG_SET = df_train.ECG_ID
    # ecg = []
    # for ecg_id in ECG_SET:
    #     ecg_path = os.path.join(records_path, ecg_id + '.h5')
    #     # print(ecg_path)
    #     with h5py.File(ecg_path, 'r') as f:
    #         signal= f['ecg'][()]
    #         signal= np.transpose(signal)#transpose to make shape be ( , 12)
    #         # print(signal.dtype)
    #         if(signal.shape != (5000, 12)):#only leave (5000,12)
    #             x_train = signal[:5000]
    #         else: 
    #             x_train = signal
    #         # print(x_train)
    #         # exit()
    #         ###################################################小波变换
    #         coeffs = pywt.wavedec(x_train, 'sym8', level=10)
    #         # print(len(coeffs))#返回level+1个系数，第一个为近似系数CA1,后面level个为细节系数
    #         coeffs[1]=np.zeros_like(coeffs[1])
    #         # print(coeffs[1])
    #         # exit(0)
    #         coeffs[2]=np.zeros_like(coeffs[2])
    #         coeffs[8]=np.zeros_like(coeffs[8])
    #         coeffs[9]=np.zeros_like(coeffs[9])
    #         coeffs[10]=np.zeros_like(coeffs[10])
    #         x_train_pre = pywt.waverec(coeffs, 'sym8')
    #         # print('wavelet done!')
    #         # print(x_train_pre)
    #         x_train_pre = zscore(x_train_pre)
    #         # print(x_train_pre)
    #         # print('zscore done!')
    #         # exit(0)
    #         ####################################################
    #         x_train_pre = np.pad(x_train_pre, ((0, 120), (0, 0)), 'constant') 
    #         ecg.append(x_train_pre)
    #         # print(ecg)
    #         # exit(0)
    #         f.close()
    # ecg = np.array(ecg, dtype=np.float32)
    # ecg = tf.convert_to_tensor(ecg, dtype=tf.float32)
    # ecg = tf.reshape(ecg, [ecg.shape[0], 5120, 12, 1])
    # ###########################################################
    # # save data as txt formal
  
    # with open(Xsavedir, 'wb') as f:
    #     data = pickle.dump(ecg, f)
    #     print(f'{namestr(df_train, globals())[0]} has saved!')
    
    # ########################################################################################################
    #load data
    with open(Xsavedir, 'rb') as f:
        X_ecg = pickle.load(f)

    return X_ecg, gender, age



if __name__=='__main__':
   

    df_train, df_val, df_test = spilt_train_test(csv_path)
    print(f'The Training set has {len(df_train)} records')
    print(f'The Validation set has {len(df_val)} records')
    print(f'The Test set has {len(df_test)} records')

    ecg_train, gender_train, age_train = records_pre(df_train, records_path)
    # print(ecg_train.shape, gender_train.shape, age_train.shape)
    ecg_val, gender_val, age_val = records_pre(df_val, records_path)
    ecg_test, gender_test, age_test = records_pre(df_test, records_path)
    
    # print(gender_train)

    ########################
    accuracy = []
    loss = []
    avg_accuracy = 0
    avg_loss = 0
    index = 1
    epochs = 30
    for index in range(3):
        print(index, epochs)
        TIMESTAMP = "{0:%Y-%m-%dT%H:%M:%S/}".format(datetime.now())
        # log_dir = args.log_dir + TIMESTAMP
        callbacks_list = [
            # '''Interrupts training when improvement stops
            # '''
            tf.keras.callbacks.EarlyStopping(

                monitor='val_loss',

                patience=8,
            ),

            # TensorBoard(log_dir=log_dir,  # log 目录
            #             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
            #             write_graph=True,  # 是否存储网络结构图
            #             write_grads=True,  # 是否可视化梯度直方图
            #             write_images=True,  # 是否可视化参数
            # #             ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=4, mode='auto', min_lr=0.000001),
        ]
        ####################################################
        model = Multi_scale_Resnet(2, activation='softmax')
        # model = get_model(2, last_layer='softmax')
        # model = resnet18_v2(2, activation='softmax')
        # model = buildModel_2DCNN(2, last_layer='softmax')
        ###################################################
        learning_rate = 0.01
        decay_rate = learning_rate / epochs
        momentum = 0.8
        sgd = SGD(learning_rate=learning_rate, momentum=momentum,
                  decay=decay_rate, nesterov=False)
        ##################################################
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(ecg_train, gender_train, batch_size=16,
                            validation_data=(ecg_val, gender_val), epochs=epochs,  callbacks=callbacks_list, shuffle=True)
        score = model.evaluate(ecg_test, gender_test)
        acc = str(score[1]*100)
        print(acc)
        model.save_weights(args.path_dir + 'DNN_' +
                           acc + '%_model.h5')
        

    


