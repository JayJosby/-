from numpy import *
f = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第一次实验\Mmodel.txt', 'r', encoding='utf-8')
dict = eval(f.read())

f = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第三次实验\Mmodel20000.txt', 'r', encoding='utf-8')
dict = eval(f.read())
mark = 0
for key in dict.keys():
    if mark < 5:
        print(key,'=\n  ',dict[key])
    else:
        mark+=1