# Quickdraw_xcj
项目简介：
  通过在鼠标在页面上进行绘图，利用神经网络识别所画内容



源数据来源：https://www.kaggle.com/c/quickdraw-doodle-recognition/data

![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1.PNG)

我使用了其中的train_simplified.zip 大小约为22GB，格式为countrycode,drawing,key_id,recognized,timestamp,word。
我只使用了其中的drawing，recognized，word 分别轨迹坐标，判断成功怕标签，lable
和源数据的区别是使用了Ramer-Douglas-Peucker algorithm对源数据的点进行了处理，若需要自己处理可以调用python中的rdp包，
在我的draw_test中我就是用它对自己的数据进行处理

--------------------------------------------------------------------------------------------------------

shuffle处理：
由于机器条件和时间限制我只挑选了300多种类之中的71种进行训练
由于源数据过大无法全部加载进入内存，而神经网络需要对数据进行shuffle处理，因此我通过draw/preprecess_shuffle/csv_merage.py对所需要的文件进行合并，
而draw/preprecess_shuffle/shuffle_occ.py是某大手写的文件shuffle方法，可以shuffle我71分类的源文件，

但是在shuffle所有分类文件时会发生错误，
于是我通过多次分割合并shuffle实现总体的shuffle，所有源数据位于shuffled_date

--------------------------------------------------------------------------------------------------------


源数据处理：
draw/read.py用于读入源数据，并且存入迭代器中，draw/rnnmodel.py中使用get_train_batch和get_test_batch读取并且修改格式，
具体格式大致类似于[[23,24,1],[25,27,0],[30,37,0]...........]，其中每个三元组中前两项为点的x，y坐标，
第三项为1代表是笔画的开头，0则代表为笔画的过程点

--------------------------------------------------------------------------------------------------------


模型：
模型搭建时，第一次尝试中使用了单层的lstm模型，2epoch后测试集精确率大约为0.768，模型保存于model/model1，

后改用一层卷积加上一层lstm，2epoch后测试集精确率大约为0.817，4epoch后测试集精确率大约为0.844,模型保存于model/model2，

![Image text](https://github.com/Miaojuno/Quickdraw_xcj/blob/master/img/1-1.PNG)

在我的cnn+lstm模型中，卷积层filters=128，窗口为3，进行padding，使用relu作为激活函数，

使用maxpooling，步长为2，窗口为2，因此在进入lstm层时长度减半，

lstm神经元个数我使用了200，

全连接层的激活函数使用softmax，

使用adamw作为神经网络优化算法，并且略微调大了学习率

使用fit_generator作为训练函数，可以通过函数迭代的读入数据


--------------------------------------------------------------------------------------------------------

模型调用参数说明：
raw_train_path          #   训练集路径

raw_test_path          #   测试集路径

model_path          #   神经网络模型（存储/加载）路径

delfalse          #   是否过滤源数据中识别失败部分

continue_train          #   是否是继续训练

lines        #源数据行数

n_epochs      #训练次数

maxlen      #输入图画最大点数

lstm_units      #lstm神经元个数

若是源数据没有手动分为训练及测试集，即训练集路径=测试集路径，会自动随机分割10%作为测试集，
因为源数据中存在一些输入但是Google的模型难以识别，因此该部分也许会干扰训练，delfase=1时会过滤该部分源数据，
continue_train=1时代表我已经存在该模型，并且需要在该模型基础上进行继续训练，
maxlen输入画图最大点数代表rdp处理后的点数

--------------------------------------------------------------------------------------------------------

ui：
draw/drawing.py调用cv2实现了鼠标绘图并读取所有点的功能,并在draw/drawing_test.py中调用，对点进行预处理，
作为源数据输入神经网络模型，选取结果类别中十个可能性最大的，通过tkinter作为简单ui输出结果

--------------------------------------------------------------------------------------------------------

部分灵魂化作识别：
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
