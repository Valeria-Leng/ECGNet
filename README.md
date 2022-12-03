# ECGNet
- Age and sex estimation based on standard 12-lead ECG



- Input：shape = (N, 5120，12，1). The input tensor should contain 5000 points of ECG tracing sampled at 500 Hz (i.e. 10 seconds of signal) and filled to 5120 using 0.
The second dimension of the tensor contains the points of 12 different leads. These leads are arranged in the following order：{di, dii, diii, avr, avl, avf, v1, v2, v3, v4, v5, v6}.
All signals are represented as 32-bit floating point numbers.
- Output：shape=（N，1）or（N，2）. The output is the predicted age or sex based on ECG.

## document

The code was tested on Python 3.9.13 and tensorflow 2.8.0.


data：Saving and loading the training set/validation set/test set as txt.
Please open this link to dowmload it：https://drive.google.com/drive/folders/1MN0-3j-r9P2wz1BHssOrTyDgdVABEFWm?usp=share_link

- ``df_train.txt``

- `df_val.txt``

- ``df_test.txt``


images：save images


model: Four model scripts are included.

- ``model.CNN.py``: Reproduce the CNN model from Zachi et al.

- ``model.DNN.py``: The DNN model from the study of Emilly et al.

- ``model.muti_scale_resnet.py``: The framework of Multi_Scale_Resnet.

- ``model.resnet18_v2.py``: The framework of Resnet18.


records：Norm category of ECG records. 
The dataset we used is a public large-scale multi-label 12-lead ECG database containing 25770 ECG records from 24666 patient acquired from Shandong Provincial Hospital (SPH) between 2019/08 and 2020/08.
You can download SPH dataset by this: https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1.

save_weight: Storing the trained weights.


- ``age_train.py`` and ``gender_train.py``: Pre-processing of raw ECG signals and training weights.

- ``age_predict.py`` and ``gender_predict.py``: The results of model predictions on the test set.

- ``README.txt``

- ``records_normal_raw.csv``: Only norm category of ECG records are contained in table.
