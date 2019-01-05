# Miaojuno/Quickdraw_xcj
项目简介：
--
通过在鼠标在页面上进行绘图，利用神经网络识别所画内容



源数据
--
来源：https://www.kaggle.com/c/quickdraw-doodle-recognition/data

![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1.PNG)

我使用了其中的train_simplified.zip 大小约为22GB，格式为countrycode,drawing,key_id,recognized,timestamp,word。
我只使用了其中的drawing，recognized，word 分别为轨迹坐标，判断成功的标签，lable

train_simplified.zip和源数据的区别是使用了Ramer-Douglas-Peucker algorithm对源数据的点进行了处理，若需要自己处理可以调用python中的rdp包，
在我的draw_test中我就是用它对自己的数据进行处理


shuffle处理：
--
由于机器条件和时间限制我只挑选了71个种类进行训练，340种类训练只迭代两次

由于源数据过大无法全部加载进入内存，而神经网络需要对数据进行shuffle处理，因此我通过draw/preprecess_shuffle/csv_merage.py对所需要的文件进行合并，
而draw/preprecess_shuffle/shuffle_occ.py是某大手写的文件shuffle方法，可以shuffle我71分类的源文件，

但是`在shuffle过大文件时会发生错误`，
可以通过\
部分合并+shuffle+分割+部分合并+shuffle+合并\
实现总体的shuffle

处理过的71分类源数据csv_71_shuffled.csv已上传至：https://pan.baidu.com/s/19eWRtCTtAaiMH7JRT34VZA



源数据处理：
--
draw/read.py用于读入源数据,将其处理为如下格式存入`迭代器`中\
x中的单一项形如：
```
[[[167, 109, 80, 69, 58, 31, 57, 117, 99, 52, 30, 6, 1, 2, 66, 98, 253, 254, 246, 182, 165], [140, 194, 227, 232, 229, 229, 206, 124, 123, 149, 157, 159, 153, 110, 82, 77, 74, 109, 121, 127, 120]], [[207, 207, 210, 221, 238], [74, 103, 114, 128, 135]], [[119, 107, 76, 70, 49, 39, 60, 93], [72, 41, 3, 0, 1, 5, 38, 70]]
```
y中的单一项形如：
```
'airplane'
```
draw/rnnmodel.py中使用get_train_batch和get_test_batch读取并且修改x的格式，x的单一项形如：
```
[[167,140,1],[109,194,0],[80,227,0],[69,229,0]....[165,120,0],[207,74,1],[207,103,0]....]
```
将标签值y进行one-hot处理为单一项形如:
```
[1,0,0,0,0,.......]
```
x为shape为(p,q,3)的二维列表，p代表每次迭代读取的个数，q代表该条源数据轨迹点的长度，而每个三元组中前两项为点的x，y坐标，第三项为1代表是笔画的开头，0则代表为笔画的过程点
同样将其存入`迭代器`



模型：
--
模型搭建时，第一次尝试中使用了单层的lstm模型，2epoch后测试集精确率大约为0.768，模型保存于model/model1，
```
#lstm
model.add(Masking(mask_value=-1, input_shape=(self.maxlen, 3)))
model.add(LSTM(units=self.lstm_units, input_shape=(self.maxlen, 3)))
model.add(Dense(self.classification_num, activation='softmax'))
adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

model.fit_generator(generator=self.get_train_batch(),steps_per_epoch=self.lines,epochs=self.n_epochs,verbose=1)                         
model.save(self.model_path)
```
后改用一层卷积加上一层lstm，2epoch后测试集精确率大约为0.817，4epoch后测试集精确率大约为0.844,模型保存于model/model2
```
#conv+lstm
model.add(Conv1D(filters=128, kernel_size=3, activation='relu',padding="same",input_shape=(None,3)))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Masking(mask_value=-1, input_shape=(self.maxlen//2, 3)))
model.add(LSTM(units=self.lstm_units, input_shape=(self.maxlen//2, 3)))
model.add(Dense(self.classification_num, activation='softmax'))
adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit_generator(generator=self.get_train_batch(), steps_per_epoch=self.lines, epochs=self.n_epochs,
            verbose=1)
model.save(self.model_path)
```

340分类的2epoch精确率大约为63%，模型保存于model/model3

在我的cnn+lstm模型中，卷积层filters=128，窗口为3，进行padding，使用relu作为激活函数，

使用maxpooling，步长为2，窗口为2，因此在进入lstm层时长度减半，

masking层用于处理边长序列，使其都为指定长度输入lstm层，我在这里使用了150作为maxlen，

因此lstm层inputshape变为了(?，3)，lstm神经元个数我使用了200，

全连接层的激活函数使用softmax作为我的分类模型的激活函数，此时shape变为了71项的向量

我使用adamw作为神经网络优化算法，并且略微调大了学习率

使用fit_generator作为训练函数，可以通过generator调用get_train_batch函数迭代的读入数据，规避内存不够的问题



模型调用参数说明：
--
raw_train_path          #   训练集路径

raw_test_path          #   测试集路径

model_path          #   神经网络模型（存储/加载）路径

delfalse          #   是否过滤源数据中识别失败部分

continue_train          #   是否是继续训练

lines        #源数据行数

n_epochs      #训练次数

maxlen      #输入图画最大点数

lstm_units      #lstm神经元个数

若是源数据没有手动分为训练及测试集，即训练集路径=测试集路径，会自动随机分割10%作为测试集，\
因为源数据中存在一些图是Google的模型难以识别的\
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/2-1.PNG)\
即源数据中recognized为false的条目，该部分源数据作为干扰项也许会降低训练效果，delfase=1时会过滤该部分源数据，\
continue_train=1时代表我已经存在该神经网络模型，并且需要在该模型基础上进行继续训练，\
maxlen代表每个图rdp处理后的轨迹点数


ui：
--
draw/drawing.py调用cv2实现了鼠标绘图并记录轨迹点的功能,并在draw/drawing_test.py中被调用，\
再draw/drawing_test.py中通过rdp对轨迹点进行预处理，再转换为三元组形式，\
作为源数据输入神经网络模型，输出所有类别的可能性列表，选取列表中十个最大的，也就是神经网络识别认为最有可能性的十项,\
然后通过tkinter库制作简单ui展示输出结果，绘图完成后按C让神经网络识别图形，识别后通过点击按钮继续绘图


复现步骤：
---
1解压kaggle下载的simple数据，\
2Csv_merge.py中的第一个路径改为你所挑选的所有分类的源数据文件所在文件夹，第二个路径修改位存储路径（rawdate.csv）\
3在命令提示符中使用python shuffle_ooc.py rawdate.csv > rawdate_shuffled.csv 命令进行shuffle，注意py文件和两个csv文件需要完整路径，或是存储在同一个目录下，在该目录下运行命令提示符，会输出文件行数（文件过大时需要分割shuffle再合并）\
4调用rnnmodel.py进行模型训练，注意所有参数都需要填写，会输出测试集的训练准确率，也可以调用acc.py进行精确率测试\
5调用draw_test.py可以进行简单绘图并识别\


我挑选的71种类型：
--
```
'mushroom': 0, 'moon': 1, 'bread': 2, 'rain': 3, 'hand': 4, 'ice cream': 5, 'tree': 6, 'hamburger': 7,
'cloud': 8, 'basketball': 9, 'mountain': 10, 'finger': 11, 'tiger': 12, 'fork': 13, 'star': 14,
'baseball': 15, 'house': 16, 'cake': 17, 'castle': 18, 'line': 19, 'bear': 20, 'arm': 21, 'bus': 22,
'bridge': 23, 'wheel': 24, 'fish': 25, 'sun': 26, 'calculator': 27, 'pencil': 28, 'bed': 29, 'key': 30,
'river': 31, 'chair': 32, 'circle': 33, 'face': 34, 'airplane': 35, 'pig': 36, 'banana': 37, 'car': 38,
'bicycle': 39, 'cat': 40, 'bee': 41, 'clock': 42, 'door': 43, 'fence': 44, 'guitar': 45, 'dog': 46,
'tooth': 47, 'baseball bat': 48, 'camel': 49, 'train': 50, 'camera': 51, 'table': 52, 'eye': 53,
'shoe': 54, 'axe': 55, 'grass': 56, 'foot': 57, 'hospital': 58, 'apple': 59, 'cell phone': 60,
'beard': 61, 'cup': 62, 'elephant': 63, 'umbrella': 64, 'rabbit': 65, 'flower': 66, 't-shirt': 67,
'bird': 68, 'watermelon': 69, 'hammer': 70
```


部分灵魂画作的识别：
--
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-2.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-3.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-4.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-5.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-6.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-7.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-8.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-9.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-10.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-11.PNG)
![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-12.PNG)
