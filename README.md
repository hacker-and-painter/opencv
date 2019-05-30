# 物体检测与人脸检测

> 原文地址 [物体检测与人脸识别](https://github.com/TommyZihao/zihaoopencv)

# 本章内容

- 使用基于Haar特征的Cascade级联分类器进行人脸识别（听起来好高大上，但其实原理很简单）
- 用人脸识别同样的道理，扩展到人眼识别上
- 用opencv自带的Harr级联分类器进行人脸、人眼与微笑识别（附源代码，直接复制粘贴即可运行）

# 关于本文

[OpenCV-Python官方文档英文教程：Face Detection using Haar Cascades](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection)

[OpenCV-Python中文教程【子豪兄opencv-python教程】](https://github.com/TommyZihao/zihaoopencv)

译者：同济大学 子豪兄

2019-5-16

Bilibili视频教程：[同济子豪兄-子豪兄opencv-python教程](https://space.bilibili.com/1900783/#/)
知乎专栏：[人工智能小技巧](https://zhuanlan.zhihu.com/c_1032626015746502656)
简书专栏：[人工智能小技巧](https://www.jianshu.com/u/38cccf09b515)
Github：[TommyZihao](https://github.com/TommyZihao)


# 基础知识

## 啥是Harr特征

Haar特征包含三种：边缘特征、线性特征、中心特征和对角线特征。每种分类器都从图片中提取出对应的特征。

![Harr特征](https://upload-images.jianshu.io/upload_images/13714448-4f7cbda986ae8a33.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![搜索人脸上的指定特征](https://upload-images.jianshu.io/upload_images/13714448-f2b5b042d83a9635.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

比如上图中，横的黑道将人脸中较暗的双眼提取了出来，而竖的白道将人脸中较亮的鼻梁提取了出来。

> 这种分类器有很像卷积核，卷积核也是从图片中提取指定特征的筛选器。
>
> 我去年制作的这个视频生动形象展示了卷积神经网络提取图片特征的整个过程
>
> [大白话讲解卷积神经网络工作原理](<https://www.bilibili.com/video/av35087157>)

## 啥是Cascade级联分类器

基于Haar特征的cascade级联分类器是Paul Viola和 Michael Jone在2001年的论文”Rapid Object Detection using a Boosted Cascade of Simple Features”中提出的一种有效的物体检测方法。

### Cascade级联分类器的训练方法：Adaboost

级联分类器的函数是通过大量`带人脸`和`不带人脸`的图片通过机器学习得到的。对于人脸识别来说，需要几万个特征，通过机器学习找出人脸分类效果最好、错误率最小的特征。训练开始时，所有训练集中的图片具有相同的权重，对于被分类错误的图片，提升权重，重新计算出新的错误率和新的权重。直到错误率或迭代次数达到要求。这种方法叫做`Adaboost`。

在Opencv中可以直接调用级联分类器函数。

### 将弱分类器聚合成强分类器

最终的分类器是这些弱分类器的加权和。之所以称之为弱分类器是因为每个分类器不能单独分类图片，但是将他们聚集起来就形成了强分类器。论文表明，只需要200个特征的分类器在检测中的精确度达到了95%。最终的分类器大约有6000个特征。(将超过160000个特征减小到6000个，这是非常大的进步了） 。

### 级联的含义：需过五关斩六将才能被提取出来

事实上，一张图片绝大部分的区域都不是人脸。如果对一张图片的每个角落都提取6000个特征，将会浪费巨量的计算资源。

如果能找到一个简单的方法能够检测某个窗口是不是人脸区域，如果该窗口不是人脸区域，那么就只看一眼便直接跳过，也就不用进行后续处理了，这样就能集中精力判别那些可能是人脸的区域。 
为此，有人引入了Cascade 分类器。它不是将6000个特征都用在一个窗口，而是将特征分为不同的阶段，然后一个阶段一个阶段的应用这些特征(通常情况下，前几个阶段只有很少量的特征)。如果窗口在第一个阶段就检测失败了，那么就直接舍弃它，无需考虑剩下的特征。如果检测通过，则考虑第二阶段的特征并继续处理。如果所有阶段的都通过了，那么这个窗口就是人脸区域。 
作者的检测器将6000+的特征分为了38个阶段，前五个阶段分别有1，10，25，25，50个特征(前文图中提到的识别眼睛和鼻梁的两个特征实际上是Adaboost中得到的最好的两个特征)。根据作者所述，平均每个子窗口只需要使用6000+个特征中的10个左右。 

# OpenCV中的Haar-cascade检测

OpenCV 既可以作为检测器也可以进行机器学习训练。如果你打算训练自己的分类器识别任意的物品，比如车，飞机，咖啡杯等。你可以用OpenCV 创造一个。完整的细节在:[Cascade Classifier Training¶](https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html)中。 

下面给出调用OpenCV进行基于Haar特征的人脸和人眼Cascade级联分类器的源代码。

## 小试牛刀：十行代码完成人脸识别（附源代码）

```python
# 十行代码完成人脸识别
# 20190526
import cv2

img = cv2.imread('image1.jpg',1)

face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

faces = face_engine.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

看不太懂？莫慌，请看下面的注释版本~

```python
# 十行代码完成人脸识别-注释版
# 20190526

# 导入opencv-python
import cv2

# 读入一张图片，引号里为图片的路径，需要你自己手动设置
img = cv2.imread('image1.jpg',1)

# 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
faces = face_engine.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)

# 对每一张脸，进行如下操作
for (x,y,w,h) in faces:
    # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# 在"img2"窗口中展示效果图
cv2.imshow('img2',img)
# 监听键盘上任何按键，如有按键即退出并关闭窗口，并将图片保存为output.jpg
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg',img)
```

![人脸识别](https://upload-images.jianshu.io/upload_images/13714448-8b51c1b390b8eff8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 对单个图片进行人脸辨识和人眼检测

```python
# 单张图片人脸+眼睛识别
# 2019-5-26

#导入opencv
import cv2

# 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征，cv2.data.haarcascades即为存放所有级联分类器模型文件的目录
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# 导入人眼级联分类器引擎吗，'.xml'文件里包含训练出来的人眼特征
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

# 读入一张图片，引号里为图片的路径，需要你自己手动设置
img = cv2.imread('image3.png')

# 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
faces = face_cascade.detectMultiScale(img, 1.3, 5)

# 对每一张脸，进行如下操作
for (x,y,w,h) in faces:
    # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
    face_area = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_area)
    # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
    for (ex,ey,ew,eh) in eyes:
        #画出人眼框，绿色，画笔宽度为1
        cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
# 在"img2"窗口中展示效果图
cv2.imshow('img2',img)
# 监听键盘上任何按键，如有案件即退出并关闭窗口，并将图片保存为output.jpg
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg',img)
```

![人脸识别+人眼检测](https://upload-images.jianshu.io/upload_images/13714448-b0658f8312ec97ea.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![名画-人脸检测](https://upload-images.jianshu.io/upload_images/13714448-8e24eef3b4c1f100.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 调用电脑摄像头进行实时人脸辨识和人眼识别

```python
# 调用电脑摄像头进行实时人脸+眼睛识别，可直接复制粘贴运行
# 2019-5-26
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
# 调用摄像头摄像头
cap = cv2.VideoCapture(0)

while(True):
    # 获取摄像头拍摄到的画面
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    img = frame
    for (x,y,w,h) in faces:
    	# 画出人脸框，蓝色，画笔宽度微
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    	# 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_area)
    	# 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        for (ex,ey,ew,eh) in eyes:
            #画出人眼框，绿色，画笔宽度为1
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
	# 实时展示效果画面
    cv2.imshow('frame2',img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 最后，关闭所有窗口
cap.release()
cv2.destroyAllWindows()
```

![](https://raw.githubusercontent.com/gaohanghang/images/master/img20190528200745.png)

## 增加摄像头实时微笑识别功能

```python
# 调用电脑摄像头进行实时人脸+眼睛+微笑识别，可直接复制粘贴运行
# 2019-5-26
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
# 调用摄像头摄像头
cap = cv2.VideoCapture(0)

while(True):
    # 获取摄像头拍摄到的画面
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    for (x,y,w,h) in faces:
    	# 画出人脸框，蓝色，画笔宽度微
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    	# 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y+h, x:x+w]
        
        ## 人眼检测
        # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        eyes = eye_cascade.detectMultiScale(face_area,1.3,10)
        for (ex,ey,ew,eh) in eyes:
            #画出人眼框，绿色，画笔宽度为1
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
        ## 微笑检测
        # 用微笑级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        smiles = smile_cascade.detectMultiScale(face_area,scaleFactor= 1.16,minNeighbors=65,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in smiles:
            #画出微笑框，红色（BGR色彩体系），画笔宽度为1
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        
	# 实时展示效果画面
    cv2.imshow('frame2',img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 最后，关闭所有窗口
cap.release()
cv2.destroyAllWindows()
```

![](https://raw.githubusercontent.com/gaohanghang/images/master/img20190528200957.png)

![微笑检测](https://upload-images.jianshu.io/upload_images/13714448-b9f928a4e4bc6a5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![微笑检测](https://upload-images.jianshu.io/upload_images/5201633-62406a163322f429.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)



# 局限性

- 仅为人脸检测，非人脸“辩识”，即只能框出人脸的位置，看不出人脸是谁。
- 仅能标出静态图片和视频帧上的人脸、人眼和微笑，不能进行“活体识别”，即不能看出这张脸是真人还是手机上的照片，如果用于人脸打卡签到、人脸支付的话会带来潜在的安全风险。
- 仅为普通的机器学习方法，没有用到深度学习和深层神经网络。



# 参考文章与扩展阅读

[Face Detection using Haar Cascades](<https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection>)<br>

[使用Haar Cascade 进行人脸识别](<https://blog.csdn.net/wutao1530663/article/details/78294349>)<br>



