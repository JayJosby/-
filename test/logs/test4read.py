import numpy as np
#
# list = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000word.npy')
# fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000word.txt',mode='w')
# list = list.tolist()
# print(list)
# for word in list:
#     print(word)
#     fw.write(word+'\n')
# fw.close()



dict = {'a':[1],'b':[2],'c':[3]}
if 'g' in dict:
    print('ok')

# f =open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\w.txt')
# #fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000word.txt',mode='w')
# line = f.readline()
#
# while line:
#     print(line)
#     line = f.readline()
# while line:
#     w1 = line.split()[0]
#     w2 = line.split()[1]
#     mark = 0
#     if w1 in wordList:
#         print(w1, w2)
#     if w1 == 'the' and w2 == 'word' :
#         dict = {}
#         text = []
#         voice = []
#         print('开始第一部分的读取')
#         l = f.readline()
#         print(l)
#         l2 = f.readline()
#         print(l2)
#         l3 = f.readline()
#         print(l3)
#         num = 0
#     elif w1 == 1.0:
#         dict = {}
#         text = []
#         voice = []
#     elif w1 == 'the' and w2 == 'temp':
#         mark = 1
#         print('第一部分单词总数为', num,',开始第二部分读取')
#         # line = f.readline()
#         # list = line.split()
#         # print(line)
#         l = f.readline()
#         print(l)
#         l2 = f.readline()
#         print(l2)
#         l3 = f.readline()
#         print(l3)
#         num = 0
#     elif w1 == 'the' and w2 == 'pinyin':
#         mark = 1
#         print('第二部分单词总数为', num,', 开始第三部分读取')
#         l = f.readline()
#         print(l)
#         l2 = f.readline()
#         print(l2)
#         l3 = f.readline()
#         print(l3)
#         num = 0
#     else:
#         # list = line.split()
#         # for i in range(len(list)):
#         #     if i == 0:
#         #         word = list[i]
#         #     else:
#         #         text.append(list[i])
#         # #print(word,mark)
#         # sub_dict['text'] = text
#         # dict[word] = sub_dict
#         text = []
#         sub_dict = {}
#         num += 1
#     line = f.readline()
# print('单词总数为,', num)