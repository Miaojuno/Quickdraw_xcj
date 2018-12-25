import glob


csvx_list = glob.glob('F:/Date/draw/myrawdate/*.csv')
print('发现CSV文件%s个' % len(csvx_list))
for count,i in enumerate(csvx_list):
    print(i, end=':')
    fr = open(i, 'r').read()
    with open(r'F:\Date\draw\123\csv_71.csv', 'a') as f:
        f.write(fr)
    print('成功')
print('写入完毕')