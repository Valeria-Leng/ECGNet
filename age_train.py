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
from keras import backend as K
import pickle
from datetime import datetime
from gender_train import buildModel_2DCNN
from model.resnet18_v2 import resnet18_v2
from model.DNN import get_model
from model.muti_scale_resnet import Multi_scale_Resnet
import argparse
# from dataloader import namestr, spilt_train_test, records_pre
records_path = './records/'
csv_path = './records_normal_raw.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--path_dir', default='./save_weight/age/')
parser.add_argument('--batch_size', default=64)
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
    #Get the index of the training set and delete it from the original table to get the test set 获取训练集的index后从原表中删去得到测试集
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


def coeff_determination(y_true, y_pred):

    SS_res = K.sum(K.square(y_true-y_pred))

    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res/(SS_tot + K.epsilon()))


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
    epochs = 32
    
    for index in range(5):
        print(index, epochs)
        TIMESTAMP = "{0:%Y-%m-%dT%H:%M:%S}".format(datetime.now())
        save_path = os.path.join(args.path_dir, 'MultiScaleResNet_' + TIMESTAMP + '_model.h5')
        # print(save_path)
        file = os.getcwd()
        if not os.path.exists(save_path):
            os.mknod(save_path)
        # log_dir = args.log_dir + TIMESTAMP
        callbacks_list = [
            # '''Interrupts training when improvement stops
            # '''
            tf.keras.callbacks.EarlyStopping(

                monitor='val_loss',

                patience=8,
            ),
            tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                 monitor="val_loss",
                                 mode = "min",
                                 save_weights_only=True,
                                 save_best_only=False,
                                 verbose=1,
                                 save_freq=5580
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
     
        model = Multi_scale_Resnet(1, activation='linear')
        # model = get_model(1, last_layer='linear')
        # model = resnet18_v2(1, activation='linear')
        # model = buildModel_2DCNN(1, last_layer='linear')
        model.compile(optimizer='adam',
                      loss='mse', metrics=['mae', coeff_determination])
        history = model.fit(ecg_train, age_train, batch_size=16,
                            validation_data=(ecg_val, age_val), epochs=epochs,  callbacks=callbacks_list, shuffle=True)
       
        scores = model.evaluate(ecg_test, age_test, verbose=0)

        print('Model evaluation: ', scores)
    ######################################################
        # predicted = model.predict(x_test)
        # plt.figure()
        # plt.scatter(y_test*100, predicted*100)
        # x = np.linspace(0, 1, 100)
        # y = x
        # plt.plot(x*100, y*100, color='red', linewidth=1.0,
        #          linestyle='--', label='line')
        # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # # plt.legend(["y = x"])
        # plt.title("Multi-Scale-ResNet Estimated Age vs Chronological Age")
        # plt.xlabel('Chronological Age (years)')
        # plt.ylabel('Multi-Scale-ResNet Estimated Age (years)')
        # plt.savefig('./images/test_' + str(index) + '.png', dpi=200,
        #             bbox_inches='tight', transparent=False)
    ######################################################
        index = index+1
        acc = str(scores[1]*100)
        model.save_weights(args.path_dir + 'MultiScaleResNet_' +
                           acc + '_model.h5')
        # print(acc)
        
        

    


