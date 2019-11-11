# 中文意图识别（BERT）

>### 1、*数据准备与预处理*  
>### 2、*模型训练*  
>### 3、*模型预测*  
>### 4、*NLP下游任务* 

>#### 需要tensorflow>1.4(我是在CPU上跑的，如果你有GPU或者想利用google-colab的GPU或者TPU训练模型，那么参考bert源码)  

## 1、数据准备与预处理  
1.1、下图是训练数据格式  
>
>这部分不再细说  

## 2、模型训练  
2.1、bert预训练模型下载,[chinese_L-12_H-768_A-12](https://github.com/google-research/bert)  
>其他模型也可以，比如[roberta_zh](https://github.com/brightmart/roberta_zh)  。但是如果你想用[albert_zh](https://github.com/brightmart/albert_zh)，就需要对代码有一些改动。

2.2、并下载[bert源代码](https://github.com/google-research/bert)，存放在路径下bert文件夹中  
2.3、参考run_classifier.py文件，创建自己的训练文件bert-intention-train-pd.py  
2.4、python bert-intention-train-pd.py运行即可开始训练  








