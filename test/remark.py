'''
常见操作:
pycharm操作技巧:
dot矩阵相乘
*  元素相乘
tuple list相互转换
tuple=tuple(list)
list=list(tuple)

python 函数的形式:
def functionname( parameters ):
   "函数_文档字符串"
   function_suite

python 引用其他py文件的函数:
import test即可
如果不在同一个package中，那么
from 包名 import test即可

想要查看函数的定义:ctrl+B
查看函数/类的引用:按下Ctrl+Alt+B (就是查看有哪些函数的参数是这个类)

以下划线开头的标识符是有特殊意义的。
以单下划线开头 _foo 的代表不能直接访问的类属性，需通过类提供的接口进行访问，不能用 from xxx import * 而导入。
以双下划线开头的 __foo 代表类的私有成员，以双下划线开头和结尾的 __foo__ 代表 Python 里特殊方法专用的标识，如 __init__() 代表类的构造函数。
----------------------------------------------------------------------
可以使用斜杠（ \）将一行的语句分为多行显示，但如果在{}内就不需要

----------------------------------------------------------------------
---------------------------python变量类型--------------------------
多个变量赋值:
a = b = c = 1
a, b, c = 1, 2, "john"
基本数据类型
Numbers（数字）
String（字符串）
List（列表）
Tuple（元组）
Dictionary（字典）
数字:（有四种:int long float complex（复数））
字符串:两种取值顺序，从左到右索引默认0开始的，最大范围是字符串长度少1，从右到左索引默认-1开始的，最大范围是字符串开头
print str           # 输出完整字符串
print str[0]        # 输出字符串中的第一个字符
print str[2:5]      # 输出字符串中第三个至第五个之间的字符串
print str[2:]       # 输出从第三个字符开始的字符串
print str * 2       # 输出字符串两次
print str + "TEST"  # 输出连接的字符串
---------------------------python运算符--------------------------
---------------------------python条件语句--------------------------
---------------------------python循环语句--------------------------
---------------------------python列表--------------------------
列表List
用于将元祖转化为列表list(tuple)
如何获取字符串长度len(string)
字符串如何转化为整数int(string,16/10)
---------------------------python元组--------------------------
---------------------------python字典--------------------------
---------------------------python类--------------------------
class Employee:
   '所有员工的基类'
   empCount = 0
   def __init__(self, name, salary):
      self.name = name				//注意到这里并没有提前声明变量
      self.salary = salary
      Employee.empCount += 1			//注意到这十分类似静态变量，如果想构造对象自己的属性
                                        //就要在__init__中写

   def displayCount(self):
     print "Total Employee %d" % Employee.empCount

   def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary

"创建 Employee 类的第一个对象"
emp1 = Employee("Zara", 2000)
"创建 Employee 类的第二个对象"
emp2 = Employee("Manni", 5000)
emp1.displayEmployee()
emp2.displayEmployee()
print "Total Employee %d" % Employee.empCount
empCount 变量是一个类变量，它的值将在这个类的所有实例之间共享。你可以在内部类或外部类使用 Employee.empCount 访问。
第一种方法__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法。
self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。

子类不重写 __init__，实例化子类时，会自动调用父类定义的 __init__。

你可以添加，删除，修改类的属性，如下所示：		//与java，C十分不同，可以动态添加类的属性
emp1.age = 7  # 添加一个 'age' 属性
emp1.age = 8  # 修改 'age' 属性
del emp1.age  # 删除 'age' 属性
你也可以使用以下函数的方式来访问属性：
getattr(obj, name[, default]) : 访问对象的属性。
hasattr(obj,name) : 检查是否存在一个属性。
setattr(obj,name,value) : 设置一个属性。如果属性不存在，会创建一个新属性。
delattr(obj, name) : 删除属性。
hasattr(emp1, 'age')    # 如果存在 'age' 属性返回 True。
getattr(emp1, 'age')    # 返回 'age' 属性的值
setattr(emp1, 'age', 8) # 添加属性 'age' 值为 8
delattr(emp1, 'age')    # 删除属性 'age'

Python内置类属性
__dict__ : 类的属性（包含一个字典，由类的数据属性组成）
__doc__ :类的文档字符串
__name__: 类名
__module__: 类定义所在的模块（类的全名是'__main__.className'，如果类位于一个导入模块mymod中，那么className.__module__ 等于 mymod）
__bases__ : 类的所有父类构成元素（包含了一个由所有父类组成的元组）

python对象销毁(垃圾回收)
Python 使用了引用计数这一简单技术来跟踪和回收垃圾。
在 Python 内部记录着所有使用中的对象各有多少引用。
一个内部跟踪变量，称为一个引用计数器。
当对象(比如一个具体的数字)被创建时， 就创建了一个引用计数(有多少个变量的值为这个对象)， 当这个对象不再需要时， 也就是说， 这个对象的引用计数变为0 时， 它被垃圾回收。但是回收不是"立即"的， 由解释器在适当的时机，将垃圾对象占用的内存空间回收。
析构方法, 删除一个对象
简单的调用方法 : del obj



类(Class):
类变量：
数据成员：
方法重写：
局部变量：
实例变量：
继承：
实例化：
实例化类其他编程语言中一般用关键字 new，但是在 Python 中并没有这个关键字，类的实例化类似函数调用方式。
以下使用类的名称 Employee 来实例化，并通过 __init__ 方法接收参数。
方法：
对象：


---------------------------python多线程--------------------------
---------------------------pythonJOSN--------------------------
np.eye():
函数的原型：numpy.eye(N,M=None,k=0,dtype=<class 'float'>,order='C)
返回的是一个二维2的数组(N,M)
np.identity():
这个函数和之前的区别在于，这个只能创建方阵，也就是N=M

print打印:print('Epochs:{}/{}'.format(e, epochs))
//Epochs:0/20 Iteration:100 Train loss: 0.01009926
//{ }代表输出位置
yield: 十分类似于return
一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
split():
str.split(str="", num=string.count(str)).str是分隔符，默认是所有的空字符，其中num是分割次数





tf.transpose(input, [dimension_1, dimenaion_2,..,dimension_n]):这个函数主要适用于交换输入张量的不同维度用的，如果输入张量是二维，就相当是转置。
tf.name_scope:类似tf.variable_scope，定义作用域
tf.nn.embedding_lookup（tensor, id）:
值得注意的试试，id是一个列表，而非直接类似于索引矩阵
tf.concat:
用于连接两个矩阵
tf.transpose(input, [dimension_1, dimenaion_2,..,dimension_n]):这个函数主要适用于交换输入张量的不同维度用的，如果输入张量是二维(就不用设定维度参数了)，就相当是转置。

tf.Variable用法：

v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')

v2=tf.Variable(tf.constant(2),name='v2')

v3=tf.Variable(tf.ones([4,3]),name='v3')


np.linspace               #产生线性间距向量numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

# python中的中括号[ ]：代表list列表数据类型-------------------------------
for size in [128]:
   print(size)            #128
   print(type(size))      #int
比如创建两个指定num_units(128和256)的lstmCell:
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]


b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
当i,j都缺省时，a[:]就相当于完整复制一份a了
b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.


①替换:ctrl+R=pychram中替换
②查看代码:选中代码view->quick definition
pycharm如何左移/右移代码？选中+Tab/shift+Tab
python大小写敏感，变量不用声明类型，但需要初始化(赋值)
pycharm中创建工程后，在第一个文件夹里右键new python file
类似①选择python版本和②导入库的操作在file-setting-project-project interpreter
打印就是一句print即可
如何获取矩阵尺寸？mat.shape 只要行数 mat.shape[0] 列数mat.shape[1]
矩阵操作看这个帖子https://www.cnblogs.com/chamie/p/4870078.html
tf.cast（x，type A),将对象x转换为A类型
tf.reshape(x,shape),将对象尺寸修改，可改变维度，卷积时常用图像1->4
如何在pycharm中打开terminal？View->Tool Windows->terminal
安装库的位置？python/Lib/site-package文件夹下面看到安装的库
在一个.py文件中，如果不是在定义函数，也就是说不是在def关键字的内嵌结构内，python会默认其余部分函数是main函数，并自动执行.
但正规工程中，一般都会将main函数写为：if__name__==__main__
def sayHello():
  str="hello"
  print(str);

if __name__ == "__main__":
  print ('This is main of module "hello.py"')
  sayHello()</code>

和上面一个意思，一个python的文件有两种使用的方法，第一是直接作为脚本执行，第二是import到其他的python脚本中被调用（模块重用）执行。因此if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程。

Tensorflow通过一个会话对象来管理Computational graph节点动态变换。
需要调用Session的run方法获得，其中可以传入两种参数:
①边/Tensor:如果传入的是Tensor对象，则是获得Tensor对象的数据
②节点/operation:如果传入的是操作节点，则会先获取节点返回的Tensor对象，再获取此对象的数据
可以用其他方法代替吗？为了避免一个变量来持有会话（？）
使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run()

如何打印张量Tensor?无论哪种方式都必须在Session中进行
①print(sess.run(a))
②with sess:
	print(a.run())//或者print(a.eval()),这使用了InteractiveSession获取上下文的Session,Tensor.eval() 和 Operation.run()
③sess.run(tf.Print(a,[a]))//tf.Print只是个OP


看看占位符的声明形式-----------------------------------

生成矩阵:
①生成全0矩阵
tf.zeros(shape, dtype, name)//tf.zeros([2, 3], int32)     # ==> [[0, 0, 0], [0, 0, 0]]
②生成与给定数据同尺寸的全0矩阵
tf.zeros_like(tensor, dtype, name)
③生成全1矩阵
tf.ones(shape, dtype, name)
④生成一个给定值的常量
tf.constant(value, dtype, shape, name)//a = tf.constant([[1, 2, 3], [4, 5, 6]], int32) # ==> [[1, 2, 3], [4, 5, 6]]
⑤生成一个全部为给定数字的数组
tf.fill(dims, value, name)
⑥生成符合正态分布的矩阵
tf.truncated_normal(shape,mean均值，stddev=0.1标准差)
⑦生成符合正态分布随机数矩阵
random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)


生成矩阵:numpy(把复杂的矩阵运算移到python外实现)
    ①np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    ②np.ones((3,3))、np.zeros((3,3))、np.eye(1,1)
    ③a.T 表示转置 a.I 表示逆矩阵
    ④与list有何区别？list存储的是对象(+指针)，更多内存
	⑤对应元素相乘用multiple(a1,a2)，矩阵相乘可直接写
	⑥数组相加不需要行数/列数一致，矩阵必须一致
	⑦squeeze()此方法删除维度为1的部分，如没有维度为1的行/列/页，不产生效果
	⑧argmax()此方法返回最大数的索引

字典中是有序的吗？按照hash来存储的，不是有序的，但可以用OederedDict实现排序
	①d=collections.OrderedDict()
	②按关键字key排序
	  dd = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
	  kd = collections.OrderedDict(sorted(dd.items(), key=lambda t: t[0]))
	③按照值value排序
	  vd = collections.OrderedDict(sorted(dd.items(),key=lambda t:t[1]))

在超类中实现init():
我们通过实现init()方法来初始化对象。当一个对象被创建，Python首先创建一个空对象，然后为那个新对象调用init()方法。__init__方法的第一个参数永远是self，表示创建的实例本身.
class Card:
  def __init__(self, rank, suit):
    self.suit = suit
    self.rank = rank
    self.hard, self.soft = self._points()

读取文本:read() readline() readlines()
	只有readline()是按行读取
	fh = open('c:\\autoexec.bat')
        for line in fh.readlines():
        print line

写入数据库:

消除\n:
	str.strip()函数原型

	声明：s为字符串，rm为要删除的字符序列

数字转换为字符串:
str(number)

全局/局部变量:
在python的函数中和全局同名的变量，如果你有修改变量的值就会变成局部变量，如果确定要引用全局变量，并且要对它修改，必须加上global关键字

定义函数:
def functionname( parameters ):
   "函数_文档字符串"
   function_suite
   return [expression]
----------------------------------------------------------------------------------------------------------
self加不加对对变量的影响:
①self只有在类的方法中才会有，独立的函数或方法是不必带有self的。self在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
②简单认为方法中加self的变量可以看成是类的属性，或者是特性。使用方法改变和调用属性，属性改变实例的状态。方法中不加self的变量可以看成一个局部变量，该变量不能被直接引用。

Python中参数:
和其他语言不一样，在Python中，一切皆对象， Python参数传递采用的都是“传对象引用”的方式。实际上，这种方式相当于传值和传引用的一种综合。如果函数收到的是一个可变对象（比如字典或者列表）的引用，就能修改对象的原始值，相当于通过“传引用”来传递对象。如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，就不能直接修改原始对象，相当于通过“传值’来传递对象，此时如果想改变这些变量的值，可以将这些变量申明为全局变量。


取消反斜杠的影响:
\\用两个

延时操作:
time.sleep(0.001)#秒

新建文件夹
import os

def mkdir(path):

	folder = os.path.exists(path)

	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print "---  new folder...  ---"
		print "---  OK  ---"

	else:
		print "---  There is this folder!  ---"

file = "G:\\xxoo\\test"
mkdir(file)             #调用函数


判断字符串包含特定字串:
python 判断字符串是否包含子字符串
2017-03-09 11:09 by 丨o聽乄雨o丨, 75791 阅读, 1 评论, 收藏, 编辑
第一种方法：in

string = 'helloworld'

if 'world' in string:


第二种方法：find

string = 'helloworld'

if string.find(’world‘) == 5: #5的意思是world字符从那个序开始，因为w位于第六个，及序为5，所以判断5


第三种方法：index，此方法与find作用类似，也是找到字符起始的序号

if string.index(’world‘) > -1: #因为-1的意思代表没有找到字符，所以判断>-1就代表能找到


'''