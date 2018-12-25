import heapq

from drawing import Mouse
from keras.models import load_model
import numpy as np
from keras_preprocessing import sequence
from rdp import rdp
from tkinter import *

model=load_model(r'F:\Date\draw\result\conv+lstm\re_model_71_4.h5')
# model2=load_model(r'F:\Date\draw\result\re_model_71_2.h5')
# model3=load_model(r'F:\Date\draw\result\conv+lstm\re_model_71_2.h5')

def gettrack():
    mn = Mouse()
    mn.create_image()
    mn.track_list_list
    ntrack_list = []
    pmin = 255
    pmax = 0
    for line in mn.track_list_list:
        ls = rdp(line, epsilon=2)
        for i, it in enumerate(ls):
            ll = []
            if it[0] > pmax:
                pmax = it[0]
            if it[1] > pmax:
                pmax = it[1]
            if it[0] < pmin:
                pmin = it[0]
            if it[1] < pmin:
                pmin = it[1]
            ll.append(it[0])
            ll.append(it[1])
            if i == 0:
                ll.append(1)
            else:
                ll.append(0)
            ntrack_list.append(ll)
    pmax = pmax - pmin
    mpower = 255 / pmax
    track = []
    for line in ntrack_list:
        ll = []
        ll.append(int(mpower * (line[0] - pmin)))
        ll.append(int(mpower * (line[1] - pmin)))
        ll.append(line[2])
        track.append(ll)
    return track

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]



#classfiy50
# y_dict = {'belt': 0, 'beard': 1, 'bathtub': 2, 'basket': 3, 'bulldozer': 4, 'bottlecap': 5, 'blackberry': 6,
#           'banana': 7, 'bee': 8, 'baseball': 9, 'boomerang': 10, 'asparagus': 11, 'alarm clock': 12, 'bus': 13,
#           'cactus': 14, 'bandage': 15, 'animal migration': 16, 'bear': 17, 'birthday cake': 18, 'bracelet': 19,
#           'bread': 20, 'baseball bat': 21, 'bicycle': 22, 'backpack': 23, 'basketball': 24, 'bowtie': 25,
#           'anvil': 26, 'blueberry': 27, 'angel': 28, 'bat': 29, 'ambulance': 30, 'axe': 31, 'apple': 32,
#           'bucket': 33, 'ant': 34, 'beach': 35, 'bush': 36, 'bench': 37, 'broom': 38, 'barn': 39, 'binoculars': 40,
#           'butterfly': 41, 'broccoli': 42, 'book': 43, 'arm': 44, 'brain': 45, 'airplane': 46, 'bridge': 47,
#           'bird': 48, 'bed': 49}


#classfiy71

y_dict = {'mushroom': 0, 'moon': 1, 'bread': 2, 'rain': 3, 'hand': 4, 'ice cream': 5, 'tree': 6, 'hamburger': 7,
          'cloud': 8, 'basketball': 9, 'mountain': 10, 'finger': 11, 'tiger': 12, 'fork': 13, 'star': 14,
          'baseball': 15, 'house': 16, 'cake': 17, 'castle': 18, 'line': 19, 'bear': 20, 'arm': 21, 'bus': 22,
          'bridge': 23, 'wheel': 24, 'fish': 25, 'sun': 26, 'calculator': 27, 'pencil': 28, 'bed': 29, 'key': 30,
          'river': 31, 'chair': 32, 'circle': 33, 'face': 34, 'airplane': 35, 'pig': 36, 'banana': 37, 'car': 38,
          'bicycle': 39, 'cat': 40, 'bee': 41, 'clock': 42, 'door': 43, 'fence': 44, 'guitar': 45, 'dog': 46,
          'tooth': 47, 'baseball bat': 48, 'camel': 49, 'train': 50, 'camera': 51, 'table': 52, 'eye': 53,
          'shoe': 54, 'axe': 55, 'grass': 56, 'foot': 57, 'hospital': 58, 'apple': 59, 'cell phone': 60,
          'beard': 61, 'cup': 62, 'elephant': 63, 'umbrella': 64, 'rabbit': 65, 'flower': 66, 't-shirt': 67,
          'bird': 68, 'watermelon': 69, 'hammer': 70}


while 1:

    track=gettrack()


    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]


    for i in range(150 - len(track)):
        track.append([-1, -1, -1])
    x = []
    x.append(track)
    x = np.array(x)

    predict_lables_list = model.predict(x)
    predict_lables_list = predict_lables_list.tolist()
    predict_lables_list = predict_lables_list[0]
    max_num_index_list = map(predict_lables_list.index, heapq.nlargest(10, predict_lables_list))
    max_num_index_list = list(max_num_index_list)
    s=''
    for i,p in enumerate(max_num_index_list):
        print(i,':',get_key(y_dict, p)[0],end=' ')
        s +=str(i)
        s +=  ':'
        s += get_key(y_dict, p)[0]
        s += '---'
        s += '---'
        s += str(heapq.nlargest(10, predict_lables_list)[i])
        s += '\n\n'
    print()

    root = Tk()
    root.geometry('450x600+300+100')
    root.title("")
    t1 = Label(root, text=s, font=("黑体", 15)).grid(row=1, column=1, sticky=W)
    b1 = Button(root, text="确定", command=root.destroy, font=("黑体", 15)).grid(row=7, column=1, sticky=E)
    root.mainloop()

    # predict_lables_list = model3.predict(x)
    # predict_lables_list = predict_lables_list.tolist()
    # predict_lables_list = predict_lables_list[0]
    # max_num_index_list = map(predict_lables_list.index, heapq.nlargest(10, predict_lables_list))
    # max_num_index_list = list(max_num_index_list)
    # for i, p in enumerate(max_num_index_list):
    #     print(i, ':', get_key(y_dict, p)[0], end=' ')
    # print()
    #
    # predict_lables_list = model2.predict(x)
    # predict_lables_list = predict_lables_list.tolist()
    # predict_lables_list = predict_lables_list[0]
    # max_num_index_list = map(predict_lables_list.index, heapq.nlargest(10, predict_lables_list))
    # max_num_index_list = list(max_num_index_list)
    # for i, p in enumerate(max_num_index_list):
    #     print(i, ':', get_key(y_dict, p)[0], end=' ')
    # print()