import numpy as np
from keras.models import Sequential
from keras.layers import Dense, regularizers, Masking, Conv1D, MaxPooling1D
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


class model():

    count = 0

    def __init__(self, raw_train_path, raw_test_path, model_path, classification_num, delfalse, continue_train,lines,n_epochs,maxlen,lstm_units):
        self.raw_train_path=raw_train_path          #   训练集路径
        self.raw_test_path=raw_test_path          #   测试集路径
        self.model_path=model_path          #   神经网络模型（存储/加载）路径
        self.classification_num=classification_num          #   分类数量
        self.delfalse=delfalse          #   是否过滤源数据中识别失败部分
        self.continue_train=continue_train          #   是否是继续训练
        self.lines=lines//1000        #源数据行数
        self.n_epochs=n_epochs      #训练次数
        self.maxlen=maxlen      #输入图画最大点数
        self.lstm_units=lstm_units      #lstm神经元个数


        self.domain()

    def domain(self):
        if self.continue_train == 1:
            model = load_model(self.model_path)
            model.fit_generator(generator=self.get_train_batch(), steps_per_epoch=self.lines, epochs=self.n_epochs, verbose=1)
            model.save(self.model_path)
        else:
            model = Sequential()

            #conv+lstm
            # model.add(Conv1D(filters=128, kernel_size=3, activation='relu',padding="same",input_shape=(None,3)))
            # model.add(MaxPooling1D(pool_size=2,strides=2))
            # model.add(Masking(mask_value=-1, input_shape=(self.maxlen//2, 3)))
            # model.add(LSTM(units=self.lstm_units, input_shape=(self.maxlen//2, 3)))

            #lstm
            model.add(Masking(mask_value=-1, input_shape=(self.maxlen, 3)))
            model.add(LSTM(units=self.lstm_units, input_shape=(self.maxlen, 3)))

            # model.add(Dropout(0.2))
            model.add(Dense(self.classification_num, activation='softmax'))
            adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
            model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
            # model.fit(x_train, y_train , validation_data=(x_test,y_test), epochs=2, batch_size=500, verbose=1,shuffle=True)
            model.fit_generator(generator=self.get_train_batch(),steps_per_epoch=self.lines,epochs=self.n_epochs,verbose=1) #                               71-> 10015855
            model.save(self.model_path)


        # acc=0
        # for iii in range(10):
        #     x_test, y_test = self.get_test_batch()
        #     loss, accuracy = model.evaluate(x_test, y_test)
        #     print('test loss: ', loss)
        #     print('test accuracy: ', accuracy)
        #     acc+=accuracy
        # print('avg test accuracy: ', acc/10)


    def get_train_batch(self):
        self.readclass = readcsv(path=self.raw_train_path,delfalse=self.delfalse)
        f=self.readclass.main()
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
            x = sequence.pad_sequences(x, value=-1, maxlen=self.maxlen, padding='post')
            if self.raw_train_path==self.raw_test_path:
                x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=0)  # 分割数据集
            yield (x, y)


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
            x = sequence.pad_sequences(x, value=-1, maxlen=self.maxlen, padding='post')
            if self.raw_train_path==self.raw_test_path:
                x_train, x, y_train, y = train_test_split(x, y, test_size=0.1, random_state=0)  # 分割数据集
            return x, y

