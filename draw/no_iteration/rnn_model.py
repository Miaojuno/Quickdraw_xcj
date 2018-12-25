from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split

from no_iteration.preprocessing import loaddata


class rmodel():
    maxlen = 200
    def __init__(self,row_count):

        x, y, y_dict=self.reshapedate(row_count)
        x = sequence.pad_sequences(x,value=-1, maxlen=200, padding='post')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)#  , random_state=seed
        print(x_train.shape)
        print(x_test.shape)
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(self.maxlen, 3)))
        model.add(LSTM(units=100, input_shape=(self.maxlen, 3)))
        # model.add(Dropout(0.2))
        model.add(Dense(row_count, activation='softmax'))
        adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
        model.fit(x_train, y_train , validation_data=(x_test,y_test), epochs=2, batch_size=500, verbose=1)
        # model.save(r'F:\Date\draw\result\re1.h5')
        # model=load_model(r'F:\Date\draw\result\re3.h5')
        loss, accuracy = model.evaluate(x_test,y_test)
        print('test loss: ', loss)
        print('test accuracy: ', accuracy)
    def reshapedate(self,row_count):
        date = loaddata(row_count)
        date_x = date.x
        date_y = date.y
        y_dict=date.y_dict
        x = []
        for one in date_x:#每幅画
            x_1=[]
            for line in one:#每一笔
                for i in range(len(line[0])):#每个点
                    x_2 = []
                    x_2.append(line[0][i])
                    x_2.append(line[1][i])
                    if i==0:#第三项为1->笔画第一点  0->非笔画第一点
                        x_2.append(1)
                    else:
                        x_2.append(0)
                    if len(x_2)<self.maxlen:
                        x_1.append(x_2)
            x.append(x_1)

        y = to_categorical(date_y, num_classes=row_count)
        return x ,y ,y_dict

rmodel(row_count=5)