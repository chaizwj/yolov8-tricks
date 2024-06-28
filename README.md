<div align="">
  <p>
    <a href="https://yolovision.ultralytics.com/" target="_blank">
      </a>
  </p>


### ⭐简介
本人硕士的研究方向是目标检测，采用模型是yolov8，数据集是VisDrone2019，适用于小目标航拍图像检测。结合了<a href="https://docs.ultralytics.com/">官方使用文档</a>和<a href="https://www.bilibili.com/video/BV1QC4y1R74t/?spm_id_from=333.788.top_right_bar_window_custom_collection.content.click&vd_source=5fe50b1b35a25689fb0988c454fec5e0">b站岩学长</a>的视频讲解，教了如何更改模型，添加新模块到yolov8中，以及如何修改yaml配置文件。于是自己总结了 yolov8的训练仓库，后期大家可以随意添加不同的改进yaml配置文件进行模型的训练，觉得好的话可以点个star~


### ⭐具体详解


<details open>
<summary>安装</summary>


  
首先就是，当git clone 命令克隆本仓库到自己的本地电脑上


```bash
git clone git@github.com:chaizwj/yolov8-tricks.git
```
然后使用pip命令在一个[**Python>=3.8**](https://www.python.org/)环境中安装`ultralytics`包，此环境还需包含[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。这也会安装所有必要的[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。




```bash
pip install ultralytics
```
然后还要先要执行以下命令，根据 requirements.txt文件中需要的第三方库进行安装
```bash
pip install -r requirements.txt
```


</details>

<details open>
<summary>使用</summary>


#### 模型训练

在根目录下找到mytrain.py文件，运行下面这行代码

```python
from ultralytics import YOLO


# 加载 yolov8 模型，根据yaml配置文件。每次改进 yolov8 模型,这里更换对应 配置 yaml 就行 
model = YOLO('ultralytics/cfg/models/v8/yolov8-biformer.yaml')

# 选择预训练权重，这里默认导入了 yolov8s.pt 和 yolov8n.pt
model = YOLO('yolov8s.pt')

# 训练 yolov8 模型
results = model.train(data='VisDrone.yaml')
```

如果你想要使用其他版本的 预训练权重，可以去[预训练权重](https://github.com/ultralytics/assets/releases)中下载。

#### 模型预测

在根目录下找到mypredict.py文件，运行下面这行代码

```python
from ultralytics import YOLO


# 加载 yolov8 模型，根据yaml配置文件。每次改进 yolov8 模型,这里更换对应 配置 yaml 就行 
model = YOLO('ultralytics/cfg/models/v8/yolov8-biformer.yaml')

# 选择预训练权重，这里默认导入了 yolov8s.pt 和 yolov8n.pt
model = YOLO('yolov8s.pt')

# 训练 yolov8 模型
results = model.train(data='VisDrone.yaml')
```
### ⭐数据集

数据集已经提前下载好了，在datasets文件夹下有一个VisDrone，里面划分了训练集，测试集，还有验证集



### ⭐一些新增的地方
#### 热力图
在 Hot-Pic文件夹下的hotPic.py代码文件中，可以根据自己的喜好，选择一种生成方式，有 GradCAM, XGradCAM, EigenCAM, HiResCAM 等方式。下面是生成的热力图，仅供参考

<div align="center">
  

![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/5ad97a66-cd79-4665-a295-938637bf3f61)


              
![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/f81eab4c-de25-4660-8d23-259e731dd5b6)



</div>



</details>

#### 自定义的实验结果图

在 Experiment-Pic文件夹下有两个py代码文件，可以根据自己的喜好，选择一种生成方式。下面是生成的结果图，仅供参考

<div align="center">
  

![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/74d2aa1f-f8c5-4bbf-b38b-428276935a5c)
![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/641d063a-1c17-4544-8343-083f43d1e79b)




</div>


### ⭐持续更新中...
</details>
