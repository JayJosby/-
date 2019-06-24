import numpy as np

f =open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000.txt')
wordList = []
num = 0
line = f.readline()
while line:
    list = line.split()
    if len(list) == 1:
        w2 = line.split()[0]
    elif len(list) == 2:
        w1 = line.split()[0]
        w2 = line.split()[1]
    wordList.append(w2)
    line = f.readline()
    print(w2)

wordList = np.array(wordList)
#showWordList(wordList)
np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000word.npy', wordList)

#initWordList(wordList)  #初始化存储列表
a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000word.npy')
wordList = a.tolist()

print(wordList)


