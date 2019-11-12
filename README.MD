# 中文意图识别（BERT）

>### 关联项目：[中文命名实体识别（BERT）](https://github.com/471417367/ner)  
>### 1、*数据准备与预处理*  
>### 2、*模型训练*  
>### 3、*模型预测*  
>### 4、*NLP下游任务* 

>#### 需要tensorflow>1.4(我是在CPU上跑的，如果你有GPU或者想利用google-colab的GPU或者TPU训练模型，那么参考bert源码)  

## 1、数据准备与预处理  
1.1、下图是训练数据格式  
>![训练数据格式](https://github.com/471417367/imag/blob/master/imag_intention/%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.jpg)  
>这部分不再细说  

## 2、模型训练  
2.1、bert预训练模型下载,[chinese_L-12_H-768_A-12](https://github.com/google-research/bert)  
>其他模型也可以，比如[roberta_zh](https://github.com/brightmart/roberta_zh)  。但是如果你想用[albert_zh](https://github.com/brightmart/albert_zh)，就需要对代码有一些改动。

2.2、并下载[bert源代码](https://github.com/google-research/bert)，存放在路径下bert文件夹中  
2.3、参考run_classifier.py文件，创建自己的训练文件bert-intention-train-pd.py  
2.4、在项目目录下运行即可开始训练  
>### python bert-intention-train-pd.py  
>根据自己项目调整参数!  

2.5、训练结果(如果训练结果不理想，需要调整参数，甚至从数据标注与预处理开始重新审视项目)  
>

## 3、模型预测  
3.1、模型训练好后运行测试文件。  
>### python bert_intention_predict_pd.py  
>运行前，导入模型的文件参数需要修改  
>  
  
## 4、NLP下游任务
4.1、结合NER等NLU可以进行智能问答
4.2、情感分析，句子分类都分类任务都是通用的


## 参考资料：
1、[Bert 模型的使用](https://work.padeoe.com/notes/bert.html)





