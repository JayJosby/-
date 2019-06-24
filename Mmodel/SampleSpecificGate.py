import tensorflow as tf
import nltk
import numpy as np
from nltk.corpus import wordnet as wn


class SampleSpecificGate:
    def __init__(self, weight_text, weight_voice):
        # 注意，因为要执行tensor张量计算，所以在此类中，text和voice也应该是使用tf下的数据类型，placeholder
        # self.weight_text=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_text')
        # self.weight_voice=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_voice')
        self.representation4text = ''
        self.representation4voice = ''
        self.temp = ''
        self.w_text = tf.divide(weight_text, (weight_text + weight_voice))
        self.w_voice = tf.divide(weight_voice, (weight_text + weight_voice))
        # 为了避免反复读入词典存储文件，在此实例化存储列表对象
        self.wordList = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\wordList.npy')
        self.wordList = self.wordList.tolist()
        return

    def fuse(self, word, text, voice):
        index = self.wordList.index(word)
        w_text = self.w_text[index]  #这个地方需要用张量索引，====================可能很多地方都需要修改
        w_voice = self.w_voice[index]
        self.representation4text = tf.multiply(w_text, text)
        self.representation4voice = tf.multiply(w_voice, voice)
        ssgate_representation = tf.concat(values=[self.representation4text, self.representation4voice], axis=0)
        return ssgate_representation


def test():
    return 0

def main():
    print("we are in %s" % __name__)

if __name__ == '__main__':
    main()