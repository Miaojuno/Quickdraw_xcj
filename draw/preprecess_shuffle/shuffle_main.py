import os
import glob
import pandas as pd

# #shuffle
# csvx_list = glob.glob(r'F:\Date\draw\123\csv_tem2_*.csv')
#
# for i in csvx_list:
#     os.system('python F:/untitled6/draw/shuffle_ooc.py '+i+' > '+i[:-4]+'_shuffle.csv')

#4970789 4970789


#split
# csvx_list2 = glob.glob(r'F:\Date\draw\123\csv_tem_*_shuffle.csv')
# c_list=[6979738,7324982,7795711,7564194,7251304,7016636,5775354]
# for k,path in enumerate(csvx_list2):
#     with open(r'F:\Date\draw\123\csv_tem_'+str(k)+'_shuffle.csv') as f:
#         num=0
#         f2=open(r'F:\Date\draw\123\0\csv_tem_'+str(k)+'_0.csv','a')
#         for i,line in enumerate(f.readlines()):
#             c=i//(c_list[k]//10)
#             if c ==10:
#                 break
#             if c!=num:
#                 num=c
#                 f2.close()
#                 f2 = open('F:/Date/draw/123/'+str(num)+'/csv_tem_'+str(k)+'_'+str(num)+'.csv','a')
#             f2.write(line)
#         f2.close()


os.system('python F:/untitled6/draw/preprecess_shuffle/shuffle_ooc.py F:/untitled6/draw/123/csv_71.csv > F:/untitled6/draw/123/csv_71_shuffle.csv')


