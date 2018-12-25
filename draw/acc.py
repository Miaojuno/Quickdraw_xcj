import numpy as np
from keras.models import Sequential
from keras.layers import Dense, regularizers, Masking
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
import os
from read import readcsv


class test_acc():
    maxlen = 200
    count = 0

    def __init__(self, raw_test_path, model_path, classification_num, delfalse):

        self.raw_test_path=raw_test_path          #   测试集路径
        self.model_path=model_path          #   神经网络模型（存储/加载）路径
        self.classification_num=classification_num          #   分类数量
        self.delfalse=delfalse          #   是否过滤源数据中识别失败部分

        self.domain()

    def domain(self):

        model = load_model(self.model_path)

        acc=0
        for iii in range(1000):
            x_test, y_test = self.get_test_batch()
            loss, accuracy = model.evaluate(x_test, y_test)
            acc+=accuracy
        print('avg test accuracy: ', acc/1000)

    def get_test_batch(self):
        self.readclass = readcsv(path=self.raw_test_path,delfalse=self.delfalse)
        f = self.readclass.main()
        for f1 in f:
            date_x = f1[0]
            date_y = f1[1]
            x = []
            for one in date_x:  # 每幅画
                x_1 = []
                for line in one:  # 每一笔
                    for i in range(len(line[0])):  # 每个点
                        x_2 = []
                        x_2.append(line[0][i])
                        x_2.append(line[1][i])
                        if i == 0:  # 第三项为1->笔画第一点  0->非笔画第一点
                            x_2.append(1)
                        else:
                            x_2.append(0)
                        if len(x_2) < self.maxlen:
                            x_1.append(x_2)
                x.append(x_1)
            y = to_categorical(date_y, num_classes=self.classification_num)
            x = sequence.pad_sequences(x, value=-1, maxlen=200, padding='post')
            # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)  # 分割数据集
            return x, y


test_path=r'F:\Date\draw\123\csv_340_shuffle_test.csv'
modelpath=r'F:\Date\draw\result\re_model_340.h5'
test_acc( raw_test_path=test_path, model_path=modelpath, classification_num=340, delfalse=1)