
import os
import pandas as pd
import json

class loaddata():
    x=[]
    y=[]
    y_dict={}
    def __init__(self,num):
        file_names = self.loadfile_name(r'F:\Date\draw\rawdate')
        data=[]
        for i,file_name in enumerate(file_names):
            if i==num:
                break
            else:
                self.load_one_data(file_name)


    def loadfile_name(self,file_dir):
        for root, dirs, files in os.walk(file_dir):
            return files

    def load_one_data(self,file_name):
        df = pd.read_csv('F:\\Date\\draw\\rawdate\\'+file_name)
        for indexs in df.index:
            list_list = json.loads(df.loc[indexs].values[1])
            self.x.append(list_list)

            lable=df.loc[indexs].values[5]
            if lable not in self.y_dict:
                self.y_dict[lable]=len(self.y_dict)
            self.y.append(self.y_dict[lable])

