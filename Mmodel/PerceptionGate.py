import tensorflow as tf
import numpy as np
from test import vectorOperations



class PerceptionGate :
    '''
    #通过这个函数，将得到感知门中text/voice两个维度各自对应的权重，（我不太清楚参数传递对原数据的改变，。如果函数收到的是一个可变对象（比如字典或
    者列表）的引用，就能修改对象的原始值，相当于通过“传引用”来传递对象。如果函数收到的是一
    个不可变对象（比如数字、字符或者元组）的引用，就不能直接修改原始对象，相当于通过“传值’
    来传递对象，此时如果想改变这些变量的值，）
    '''
    '''
    在一个类中self.a和a完全不是一回事
    self.a定义在__init__中，可以在①类外和②类内其他方法中被访问
    
    '''
    def __init__(self, weight_text, weight_voice):
        # 注意，因为要执行tensor张量计算，所以在此类中，text和voice也应该是使用tf下的数据类型，placeholder
        # self.weight_text=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_text')
        # self.weight_voice=tf.placeholder(dtype=tf.float32, shape=[None, None], name='pGate_weight_voice')
        self.w_text = tf.divide(weight_text, (weight_text+weight_voice))
        self.w_voice = tf.divide(weight_voice, (weight_text+weight_voice))
        self.representation4text = ''
        self.representation4voice = ''
        self.temp = ''
        return

    def fuse(self, text, voice):
        # 对两种表示加权后相连接，得到第一个门的链接向量表示
        self.representation4text = tf.multiply(self.w_text, text)
        self.representation4voice = tf.multiply(self.w_voice, voice)
        pgate_representation = tf.concat(values=[self.representation4text, self.representation4voice], axis=0)
        return pgate_representation


    def printMessage(self):
        print('pGate融合结束')



def test():
    print()

def main():
  print("we are in %s"%__name__)
  test()

if __name__ == '__main__':
  main()