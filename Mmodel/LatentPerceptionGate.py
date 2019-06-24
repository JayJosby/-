import tensorflow as tf
import numpy as np
from test import vectorOperations
from test import test4nltk


class LantentPerceptionGate:
    def __init__(self, weight_text, weight_voice):
        # 注意，因为要执行tensor张量计算，所以在此类中，text和voice也应该是使用tf下的数据类型，placeholder
        # self.weight_text=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_text')
        # self.weight_voice=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_voice')
        self.representation4text = ''
        self.representation4voice = ''
        self.temp = ''
        self.w_text = tf.divide(weight_text, (weight_text+weight_voice))
        self.w_voice = tf.divide(weight_voice, (weight_text+weight_voice))
        #为了避免反复读入词性频率存储文件，在此实例化存储列表对象
        self.wordList = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
        self.wordList = self.wordList.tolist()
        return


    def fuse(self, word, text, voice):
        # 将词性频率列表与权重列表做点乘，得到最终权重，这里的text是单个单词形式，tagWeight是词性频率向量，分别与各词性对应文本/语音权重点乘，得到最终权重比
        # 这个地方写的不对，tf.multiply(self.w_text, tagWeight),点乘之后得到的是一个向量，没有办法直接与text做乘法，需要对其平均求和
        tagWeight = test4nltk.getWordTagWeight(self.wordList, word)
        self.representation4text = tf.multiply(tf.divide(tf.reduce_sum(tf.multiply(self.w_text, tagWeight)), 35), text)
        self.representation4voice = tf.multiply(tf.divide(tf.reduce_sum(tf.multiply(self.w_voice, tagWeight)), 35), voice)
        lpgate_representation = tf.concat(values=[self.representation4text, self.representation4voice], axis=0)
        return lpgate_representation


    def printMessage(self):
        print('调用PerceptionGate中的打印方法')



def test():


    return None



def main():
  print("we are in %s"%__name__)
  test()

if __name__ == '__main__':
  main()