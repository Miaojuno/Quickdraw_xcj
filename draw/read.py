import os
import pandas as pd
import json
import numpy as np
class readcsv():
    x = []
    y = []
    y_dict = {}

    def __init__(self,path,delfalse):
        self.path=path
        self.delfalse = delfalse


    def main(self):
        self.reader = pd.read_csv(self.path, usecols=[1,3, 5], iterator=True)
        while 1:
            try:
                df = self.reader.get_chunk(1000)
                self.load_one_data(df)
                print(self.y_dict)
                yield (self.x,self.y)
            except StopIteration:
                self.reader.close()
                self.reader = pd.read_csv(self.path, usecols=[1,3, 5], iterator=True)



    def load_one_data(self,df):#加载一个文件的数据到x,y,y_dict
        self.x = []
        self.y = []
        for indexs in df.index:
            if df.loc[indexs].values[0]=='drawing':
                continue
            if self.delfalse:
                if df.loc[indexs].values[1]==False:
                    continue
            list_list = json.loads(df.loc[indexs].values[0])
            self.x.append(list_list)
            lable=df.loc[indexs].values[2]
            if lable not in self.y_dict:
                self.y_dict[lable]=len(self.y_dict)
            self.y.append(self.y_dict[lable])

