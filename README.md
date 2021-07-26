#  一、预测分析·商品评论情感预测
竞赛地址： [https://www.heywhale.com/home/competition/609cc718ca31cd0017835fdc/content/1](https://www.heywhale.com/home/competition/609cc718ca31cd0017835fdc/content/1)
希望大家用PaddleNLP把它打趴下，地下爬。。。。。。
## 1.背景
众所周知，大数据是企业的基本生产资料，数据信息是企业 宝贵的资产。不同于其他资产，数据资产主要在企业运营过程中 产生，较易获取，但要持续积累、沉淀和做好管理却并不容易， 这是一项长期且系统性的工程。未经“雕琢”的数据是一组无序、 混乱的数字，并不能给企业带来何种价值，从庞杂晦涩的数据中 挖掘出“宝藏”充满着挑战，这需要将业务、技术与管理三者相 互融合起来进行创新。

随着网上购物越来越流行，人们对于网上购物的需求变得越来越高，这让京东，淘宝等电商平台得到了很大的发展机遇。但是，这种需求也推动了更多的电商平台的发展，引发了激烈的竞争。在这种电商平台激烈竞争的大背景下，除了提高商品质量，压低商品价格外，了解更多的消费者心声对于电商平台来说也越来越重要。其中非常重要的一种方式就是针对消费者的购物行为数据和文本评论数据进行内在信息的数据挖掘分析。而得到这些信息，也有利于对应商品的生产自身竞争力的提高，以及为用户提供高质量感兴趣的商品。

## 2.数据简介
* 本数据集包括52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据
* 本次练习赛所使用数据集基于JD的电商数据，来自WWW的JD.com E-Commerce Data，并且针对部分字段做出了一定的调整，所有的字段信息请以本练习赛提供的字段信息为准
* 评分为[1,5] 之间的整数

# 二、数据初步处理


```python
!pip install -U paddlenlp
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already up-to-date: paddlenlp in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (2.0.6)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.20.3)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.2.3)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.5)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.4.2)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl->paddlenlp) (56.2.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (7.2.0)


## 1.解压数据


```python
# !tar -xvf data/data96333/商品评论情感预测.gz
```

## 2.查看数据


```python
!head 训练集.csv
```

    数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容,评分
    TRAIN_0,300212.0,PRODUCT_60357,1282579200,刚到!!!!!!!!,"刚刚收到,2天我晕,一般快递最快到我们这要3天呢,赞个!!!   包装台简单了,说明书看不懂。 瓶子半透明。   问了官方,说卓越也是他们的合作伙伴,正品放心。",4.0
    TRAIN_1,213838.0,PRODUCT_354315,1305561600,很好的一本书,不过这本书没有赠送什么代金券。体现不出以前的正版图书送网站学习代金券的特点。,5.0
    TRAIN_2,1045492.0,PRODUCT_192005,1357747200,二手手机,"很负责任的说一句,亚马逊给我发过来的手机绝对是二手的!!",1.0
    TRAIN_3,587784.0,PRODUCT_1531,1305129600,送的光盘不行,"这本书内容很好,就是送的光盘不行。这次重新订购了一套,期望发过来的光盘能用",4.0
    TRAIN_4,1244067.0,PRODUCT_324528,1285689600,很实用,"很实用的一本书,非常喜欢!",5.0
    TRAIN_5,3361.0,PRODUCT_4163,1346256000,关于书籍的包装,"书籍本身没有问题,货物的包装实在不敢恭维。不知出于何种考虑,先前的纸盒包装现在换成了塑料袋,拍下的两本精装书拿到手居然卷了边,超级郁闷。以此种方式来降低成本,实在不足取。省下的只是仨瓜俩枣,失去的却是人们的信任。",4.0
    TRAIN_6,83841.0,PRODUCT_114046,1341849600,挺好的,"包装很好,内容也不错",4.0
    TRAIN_7,174475.0,PRODUCT_100236,1226505600,便宜点最好了,希望能尽快便宜一些!,4.0
    TRAIN_8,395880.0,PRODUCT_184161,1340812800,物流 包装 一如既往,对于自主游玩川渝还是很有帮助的,5.0



```python
!head 测试集.csv
```

    数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容
    TEST_0,1013654.0,PRODUCT_176056,1382025600,东西不错,"大三元之一 东西看上去不错,包装也都很好,关键是价格比京东便宜很多。 还没试过,回去试一下。 不足是不能开增票。比较遗憾"
    TEST_1,99935.0,PRODUCT_130680,1296144000,这么丰富的经历没写出来,"这么丰富的经历没写出来,对于我们以后上哪玩挺有帮助,作为游记一般吧。"
    TEST_2,307768.0,PRODUCT_323370,1303142400,很喜欢 支持离歌 支持饶雪漫~~,很喜欢 支持离歌 支持饶雪漫~~
    TEST_3,152011.0,PRODUCT_383545,1313510400,"内容空洞,不值得买","内容很空洞,有炫富意味,其它的倒还真没看出什么所以然来。很后悔买了这本书。完全想废纸一样。"
    TEST_4,1070630.0,PRODUCT_346185,1272556800,爱自己多一点,"这个书的内容总的来说不错的,书名有点夸张,但看了内容后,发现真的很实实在在的,一点也不夸大。本人特别喜欢后面部分关于鼓舞的内容。一个女人天生长得美人见人爱,而长得不好看的有很多人都自卑,于是总想方设法运用各种化妆品来装饰自己,以此来让别人喜欢自己。看了这个书的内容,很感动,并不是说她的观点如何的好,而是这样的观点出在减肥书上,不漂亮没关系,对自己自信一点,对周围的人更关心一点,你也可以由内而外变得越来越美丽,每天给自己一个小小的肯定,对自己说OK。"
    TEST_5,1133263.0,PRODUCT_247806,1336060800,"易懂,好用",程博士写的书易懂好用!
    TEST_6,42055.0,PRODUCT_82381,1324742400,火机油,"收到时外包装没问题,但奇怪的是里面瓶身上角有些挤变形了,还好没破,没有泄漏。除去包装外,满意。"
    TEST_7,1433.0,PRODUCT_457115,1338134400,不错的书,"不错的书,价格合适,质量还行"
    TEST_8,650346.0,PRODUCT_348453,1337097600,翻译它最大,"很喜欢里面的翻译讲解,用四步定位来解决每一个翻译题,屡试屡爽!"



```python
!head submission.csv
```

    id,score
    TEST_0,5
    TEST_1,5
    TEST_2,5
    TEST_3,1
    TEST_4,5
    TEST_5,4
    TEST_6,3
    TEST_7,5
    TEST_8,5


## 3.重写read方法读取自定义数据集


```python
from paddlenlp.datasets import load_dataset
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset
import re


# 数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容,评分
def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as in_f:
        next(in_f)
        for line in in_f:
            line = line.strip('\n')
            split_array = [i.start() for i in re.finditer(',', line)]
            id = line[:split_array[0]]
            comment_title = line[split_array[3] + 1:split_array[4]]
            comment = line[split_array[4] + 2:split_array[-2]]
            label = line[split_array[-1] + 1:]
            yield {'text': comment_title  +' '+ comment, 'label': str(int(label.split('.')[0])-1), 'qid': id}

# 数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容,评分
def read_test(data_path):
    with open(data_path, 'r', encoding='utf-8') as in_f:
        next(in_f)
        for line in in_f:
            line = line.strip('\n')
            split_array = [i.start() for i in re.finditer(',', line)]
            id = line[:split_array[0]]
            id=id.split('_')[-1]
            comment_title = line[split_array[3] + 1:split_array[4]]
            comment = line[split_array[4] + 2:split_array[-2]]
            label= '1'
            yield {'text': comment_title  +' '+ comment, 'label': label, 'qid': id}
```

## 4.训练集载入


```python
# data_path为read()方法的参数
dataset_ds = load_dataset(read, data_path='训练集.csv',lazy=False)
# 在这进行划分
train_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 15 != 1])
dev_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 15 == 1])

test_ds =  load_dataset(read_test, data_path='测试集.csv',lazy=False)
```


```python
for i in range(5):
    print(test_ds[i])
```

    {'text': '东西不错 大三元之一 东西看上去不错,包装也都很好', 'label': '1', 'qid': '0'}
    {'text': '这么丰富的经历没写出来 这么丰富的经历没写出来', 'label': '1', 'qid': '1'}
    {'text': '很喜欢 支持离歌 支持饶雪漫~~ ', 'label': '1', 'qid': '2'}
    {'text': '"内容空洞 值得买","内容很空洞', 'label': '1', 'qid': '3'}
    {'text': '爱自己多一点 这个书的内容总的来说不错的,书名有点夸张,但看了内容后,发现真的很实实在在的,一点也不夸大。本人特别喜欢后面部分关于鼓舞的内容。一个女人天生长得美人见人爱,而长得不好看的有很多人都自卑,于是总想方设法运用各种化妆品来装饰自己,以此来让别人喜欢自己。看了这个书的内容,很感动,并不是说她的观点如何的好,而是这样的观点出在减肥书上,不漂亮没关系,对自己自信一点,对周围的人更关心一点,你也可以由内而外变得越来越美丽', 'label': '1', 'qid': '4'}



```python
# 在转换为MapDataset类型
train_ds = MapDataset(train_ds)
dev_ds = MapDataset(dev_ds)
test_ds = MapDataset(test_ds)
print(len(train_ds))
print(len(dev_ds))
print(len(test_ds))
```

#  三、SKEP模型加载
![](https://ai-studio-static-online.cdn.bcebos.com/fc21e1201154451a80f32e0daa5fa84386c1b12e4b3244e387ae0b177c1dc963)


```python
# 指定模型名称一键加载模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

model = SkepForSequenceClassification.from_pretrained(
    'skep_ernie_1.0_large_ch', num_classes=  5)
# 指定模型名称一键加载tokenizer
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')
```

    [2021-07-24 00:17:29,500] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt


# 四、数据NLP特征处理


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
from utils import create_dataloader
# 处理的最大文本序列长度
max_seq_length=200
# 批量数据大小
batch_size=20

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

# 五、模型训练

## 1.训练准备


```python
import time

from utils import evaluate

# 训练轮次
epochs = 30
# 训练过程中保存模型参数的文件夹
ckpt_dir = "skep_ckpt"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```

## 2.开始训练


```python
# 开启训练

# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
best_val_acc=0
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            # 评估当前训练的模型
            eval_loss, eval_accu = evaluate(model, criterion, metric, dev_data_loader)
            print("eval  on dev  loss: {:.8}, accu: {:.8}".format(eval_loss, eval_accu))
            # 加入eval日志显示
            writer.add_scalar(tag="eval/loss", step=global_step, value=eval_loss)
            writer.add_scalar(tag="eval/acc", step=global_step, value=eval_accu)
            # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            save_dir = "best_checkpoint"
            # 加入保存       
            if eval_accu>best_val_acc:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                best_val_acc=eval_accu
                print(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                save_param_path = os.path.join(save_dir, 'best_model.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                fh = open('best_checkpoint/best_model.txt', 'w', encoding='utf-8')
                fh.write(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                fh.close()
```

    global step 10, epoch: 1, batch: 10, loss: 0.87521, accu: 0.46500, speed: 2.45 step/s
    global step 20, epoch: 1, batch: 20, loss: 0.93821, accu: 0.50500, speed: 2.26 step/s
    global step 30, epoch: 1, batch: 30, loss: 1.13836, accu: 0.52000, speed: 2.15 step/s
    global step 40, epoch: 1, batch: 40, loss: 1.27132, accu: 0.51000, speed: 1.71 step/s
    global step 50, epoch: 1, batch: 50, loss: 1.26422, accu: 0.51000, speed: 1.88 step/s
    global step 60, epoch: 1, batch: 60, loss: 1.08832, accu: 0.52583, speed: 2.10 step/s
    global step 70, epoch: 1, batch: 70, loss: 0.84578, accu: 0.54214, speed: 2.01 step/s
    global step 80, epoch: 1, batch: 80, loss: 0.68294, accu: 0.55063, speed: 1.86 step/s
    global step 90, epoch: 1, batch: 90, loss: 0.89902, accu: 0.55500, speed: 1.80 step/s
    global step 100, epoch: 1, batch: 100, loss: 1.53175, accu: 0.55650, speed: 1.87 step/s
    eval  on dev  loss: 0.95411694, accu: 0.61217056
    模型保存在 100 步， 最佳eval准确度为0.61217056！
    global step 110, epoch: 1, batch: 110, loss: 0.80307, accu: 0.66500, speed: 0.16 step/s
    global step 120, epoch: 1, batch: 120, loss: 0.90480, accu: 0.64000, speed: 1.85 step/s
    global step 130, epoch: 1, batch: 130, loss: 1.13419, accu: 0.62000, speed: 2.04 step/s
    global step 140, epoch: 1, batch: 140, loss: 0.81122, accu: 0.62750, speed: 2.16 step/s
    global step 150, epoch: 1, batch: 150, loss: 0.88405, accu: 0.63200, speed: 1.64 step/s
    global step 160, epoch: 1, batch: 160, loss: 1.05735, accu: 0.63250, speed: 1.84 step/s
    global step 170, epoch: 1, batch: 170, loss: 0.75583, accu: 0.63214, speed: 1.66 step/s
    global step 180, epoch: 1, batch: 180, loss: 0.92368, accu: 0.64000, speed: 1.85 step/s
    global step 190, epoch: 1, batch: 190, loss: 0.97310, accu: 0.64167, speed: 1.91 step/s
    global step 200, epoch: 1, batch: 200, loss: 1.02647, accu: 0.63850, speed: 1.89 step/s
    eval  on dev  loss: 0.85550189, accu: 0.64109706
    模型保存在 200 步， 最佳eval准确度为0.64109706！
    global step 210, epoch: 1, batch: 210, loss: 0.58538, accu: 0.64500, speed: 0.16 step/s
    global step 220, epoch: 1, batch: 220, loss: 0.99408, accu: 0.64000, speed: 2.33 step/s
    global step 230, epoch: 1, batch: 230, loss: 0.57425, accu: 0.65500, speed: 1.65 step/s
    global step 240, epoch: 1, batch: 240, loss: 0.55677, accu: 0.66500, speed: 1.82 step/s
    global step 250, epoch: 1, batch: 250, loss: 0.92098, accu: 0.65900, speed: 1.76 step/s
    global step 260, epoch: 1, batch: 260, loss: 1.10602, accu: 0.65667, speed: 2.11 step/s
    global step 270, epoch: 1, batch: 270, loss: 0.68762, accu: 0.65357, speed: 2.45 step/s
    global step 280, epoch: 1, batch: 280, loss: 1.29365, accu: 0.65125, speed: 1.76 step/s
    global step 290, epoch: 1, batch: 290, loss: 0.96734, accu: 0.65000, speed: 1.68 step/s
    global step 300, epoch: 1, batch: 300, loss: 0.71548, accu: 0.64950, speed: 1.60 step/s
    eval  on dev  loss: 0.86210579, accu: 0.6430255
    模型保存在 300 步， 最佳eval准确度为0.64302550！
    global step 310, epoch: 1, batch: 310, loss: 0.83134, accu: 0.65500, speed: 0.16 step/s
    global step 320, epoch: 1, batch: 320, loss: 1.02752, accu: 0.64250, speed: 1.83 step/s
    global step 330, epoch: 1, batch: 330, loss: 0.87035, accu: 0.64333, speed: 1.86 step/s
    global step 340, epoch: 1, batch: 340, loss: 0.82827, accu: 0.62875, speed: 1.84 step/s
    global step 350, epoch: 1, batch: 350, loss: 1.12230, accu: 0.63600, speed: 2.00 step/s
    global step 360, epoch: 1, batch: 360, loss: 0.75210, accu: 0.63833, speed: 1.81 step/s
    global step 370, epoch: 1, batch: 370, loss: 0.97518, accu: 0.63429, speed: 1.77 step/s
    global step 380, epoch: 1, batch: 380, loss: 0.50355, accu: 0.64000, speed: 1.99 step/s
    global step 390, epoch: 1, batch: 390, loss: 0.88269, accu: 0.64111, speed: 1.67 step/s
    global step 400, epoch: 1, batch: 400, loss: 0.96283, accu: 0.64000, speed: 1.68 step/s
    eval  on dev  loss: 0.84313977, accu: 0.63338333
    global step 410, epoch: 1, batch: 410, loss: 0.90406, accu: 0.65500, speed: 0.21 step/s
    global step 420, epoch: 1, batch: 420, loss: 0.72563, accu: 0.66000, speed: 2.11 step/s
    global step 430, epoch: 1, batch: 430, loss: 0.63781, accu: 0.65000, speed: 1.80 step/s
    global step 440, epoch: 1, batch: 440, loss: 1.07242, accu: 0.64125, speed: 1.62 step/s
    global step 450, epoch: 1, batch: 450, loss: 0.75142, accu: 0.64800, speed: 1.79 step/s
    global step 460, epoch: 1, batch: 460, loss: 0.92646, accu: 0.64250, speed: 1.75 step/s
    global step 470, epoch: 1, batch: 470, loss: 0.91629, accu: 0.63857, speed: 1.98 step/s
    global step 480, epoch: 1, batch: 480, loss: 0.89441, accu: 0.63250, speed: 1.52 step/s
    global step 490, epoch: 1, batch: 490, loss: 0.85962, accu: 0.63722, speed: 2.16 step/s
    global step 500, epoch: 1, batch: 500, loss: 1.20524, accu: 0.63700, speed: 1.84 step/s
    eval  on dev  loss: 0.81807065, accu: 0.65738162
    模型保存在 500 步， 最佳eval准确度为0.65738162！
    global step 510, epoch: 1, batch: 510, loss: 0.86491, accu: 0.66000, speed: 0.16 step/s
    global step 520, epoch: 1, batch: 520, loss: 0.90773, accu: 0.63000, speed: 1.73 step/s
    global step 530, epoch: 1, batch: 530, loss: 0.83013, accu: 0.62500, speed: 2.23 step/s
    global step 540, epoch: 1, batch: 540, loss: 0.82620, accu: 0.63625, speed: 1.74 step/s
    global step 550, epoch: 1, batch: 550, loss: 0.79216, accu: 0.64700, speed: 1.64 step/s
    global step 560, epoch: 1, batch: 560, loss: 0.75423, accu: 0.63917, speed: 2.48 step/s
    global step 570, epoch: 1, batch: 570, loss: 0.61977, accu: 0.63714, speed: 1.78 step/s
    global step 580, epoch: 1, batch: 580, loss: 0.97008, accu: 0.62875, speed: 1.53 step/s
    global step 590, epoch: 1, batch: 590, loss: 0.86932, accu: 0.63500, speed: 2.13 step/s
    global step 600, epoch: 1, batch: 600, loss: 0.66027, accu: 0.64050, speed: 2.04 step/s
    eval  on dev  loss: 0.81638068, accu: 0.64966788
    global step 610, epoch: 1, batch: 610, loss: 0.93282, accu: 0.57500, speed: 0.21 step/s
    global step 620, epoch: 1, batch: 620, loss: 0.73737, accu: 0.62250, speed: 1.96 step/s
    global step 630, epoch: 1, batch: 630, loss: 1.16130, accu: 0.62167, speed: 1.52 step/s
    global step 640, epoch: 1, batch: 640, loss: 1.19148, accu: 0.62750, speed: 1.55 step/s
    global step 650, epoch: 1, batch: 650, loss: 0.72973, accu: 0.63200, speed: 2.01 step/s
    global step 660, epoch: 1, batch: 660, loss: 0.77357, accu: 0.61833, speed: 1.84 step/s
    global step 670, epoch: 1, batch: 670, loss: 1.12195, accu: 0.62214, speed: 1.84 step/s
    global step 680, epoch: 1, batch: 680, loss: 0.86879, accu: 0.62750, speed: 1.91 step/s
    global step 690, epoch: 1, batch: 690, loss: 0.62406, accu: 0.62944, speed: 1.83 step/s
    global step 700, epoch: 1, batch: 700, loss: 0.80427, accu: 0.63200, speed: 1.74 step/s
    eval  on dev  loss: 0.82166165, accu: 0.65288194
    global step 710, epoch: 1, batch: 710, loss: 0.72915, accu: 0.67000, speed: 0.21 step/s
    global step 720, epoch: 1, batch: 720, loss: 0.84717, accu: 0.66750, speed: 1.61 step/s
    global step 730, epoch: 1, batch: 730, loss: 0.90545, accu: 0.66333, speed: 2.13 step/s
    global step 740, epoch: 1, batch: 740, loss: 1.05475, accu: 0.65125, speed: 1.78 step/s
    global step 750, epoch: 1, batch: 750, loss: 0.54910, accu: 0.65900, speed: 1.62 step/s
    global step 760, epoch: 1, batch: 760, loss: 0.71939, accu: 0.66333, speed: 2.10 step/s
    global step 770, epoch: 1, batch: 770, loss: 1.06941, accu: 0.66000, speed: 1.65 step/s
    global step 780, epoch: 1, batch: 780, loss: 0.75575, accu: 0.65000, speed: 1.91 step/s
    global step 790, epoch: 1, batch: 790, loss: 0.99837, accu: 0.65111, speed: 1.72 step/s
    global step 800, epoch: 1, batch: 800, loss: 0.35832, accu: 0.65600, speed: 1.73 step/s
    eval  on dev  loss: 0.85374933, accu: 0.63809728
    global step 810, epoch: 1, batch: 810, loss: 0.58213, accu: 0.68500, speed: 0.21 step/s
    global step 820, epoch: 1, batch: 820, loss: 0.84349, accu: 0.67000, speed: 2.04 step/s
    global step 830, epoch: 1, batch: 830, loss: 0.90423, accu: 0.66500, speed: 1.69 step/s
    global step 840, epoch: 1, batch: 840, loss: 0.91128, accu: 0.65375, speed: 2.09 step/s
    global step 850, epoch: 1, batch: 850, loss: 0.82607, accu: 0.66400, speed: 1.89 step/s
    global step 860, epoch: 1, batch: 860, loss: 1.15204, accu: 0.64917, speed: 2.15 step/s
    global step 870, epoch: 1, batch: 870, loss: 1.07332, accu: 0.65429, speed: 2.04 step/s
    global step 880, epoch: 1, batch: 880, loss: 0.63656, accu: 0.65187, speed: 1.61 step/s
    global step 890, epoch: 1, batch: 890, loss: 0.60962, accu: 0.65389, speed: 2.04 step/s
    global step 900, epoch: 1, batch: 900, loss: 0.99886, accu: 0.65000, speed: 2.02 step/s
    eval  on dev  loss: 0.81486952, accu: 0.64259696
    global step 910, epoch: 1, batch: 910, loss: 0.62347, accu: 0.59500, speed: 0.21 step/s
    global step 920, epoch: 1, batch: 920, loss: 0.79623, accu: 0.63250, speed: 2.35 step/s
    global step 930, epoch: 1, batch: 930, loss: 0.86971, accu: 0.63500, speed: 1.65 step/s
    global step 940, epoch: 1, batch: 940, loss: 0.87052, accu: 0.63000, speed: 1.92 step/s
    global step 950, epoch: 1, batch: 950, loss: 0.63715, accu: 0.64700, speed: 1.78 step/s
    global step 960, epoch: 1, batch: 960, loss: 0.89492, accu: 0.63833, speed: 1.74 step/s
    global step 970, epoch: 1, batch: 970, loss: 0.75345, accu: 0.63857, speed: 1.60 step/s
    global step 980, epoch: 1, batch: 980, loss: 1.00206, accu: 0.63875, speed: 1.57 step/s
    global step 990, epoch: 1, batch: 990, loss: 0.79009, accu: 0.64222, speed: 1.85 step/s
    global step 1000, epoch: 1, batch: 1000, loss: 0.78684, accu: 0.64850, speed: 1.99 step/s
    eval  on dev  loss: 0.8300128, accu: 0.64559674
    global step 1010, epoch: 1, batch: 1010, loss: 0.38939, accu: 0.70000, speed: 0.21 step/s
    global step 1020, epoch: 1, batch: 1020, loss: 0.81717, accu: 0.66750, speed: 1.80 step/s
    global step 1030, epoch: 1, batch: 1030, loss: 0.72697, accu: 0.67000, speed: 1.95 step/s
    global step 1040, epoch: 1, batch: 1040, loss: 0.80852, accu: 0.66875, speed: 1.57 step/s
    global step 1050, epoch: 1, batch: 1050, loss: 0.91033, accu: 0.67400, speed: 1.72 step/s
    global step 1060, epoch: 1, batch: 1060, loss: 0.78317, accu: 0.66333, speed: 1.71 step/s
    global step 1070, epoch: 1, batch: 1070, loss: 0.79349, accu: 0.65857, speed: 1.65 step/s
    global step 1080, epoch: 1, batch: 1080, loss: 0.73527, accu: 0.66563, speed: 2.08 step/s
    global step 1090, epoch: 1, batch: 1090, loss: 0.87832, accu: 0.66333, speed: 1.88 step/s
    global step 1100, epoch: 1, batch: 1100, loss: 0.79848, accu: 0.66100, speed: 1.87 step/s
    eval  on dev  loss: 0.81178689, accu: 0.64709664
    global step 1110, epoch: 1, batch: 1110, loss: 0.69097, accu: 0.66500, speed: 0.21 step/s
    global step 1120, epoch: 1, batch: 1120, loss: 0.82214, accu: 0.63500, speed: 2.05 step/s
    global step 1130, epoch: 1, batch: 1130, loss: 0.68457, accu: 0.64500, speed: 1.87 step/s
    global step 1140, epoch: 1, batch: 1140, loss: 0.83022, accu: 0.64875, speed: 1.68 step/s
    global step 1150, epoch: 1, batch: 1150, loss: 0.84339, accu: 0.65800, speed: 2.15 step/s
    global step 1160, epoch: 1, batch: 1160, loss: 0.94957, accu: 0.65833, speed: 1.99 step/s
    global step 1170, epoch: 1, batch: 1170, loss: 0.93026, accu: 0.65929, speed: 1.53 step/s
    global step 1180, epoch: 1, batch: 1180, loss: 0.90312, accu: 0.65375, speed: 2.11 step/s
    global step 1190, epoch: 1, batch: 1190, loss: 0.93971, accu: 0.65778, speed: 2.31 step/s
    global step 1200, epoch: 1, batch: 1200, loss: 0.97124, accu: 0.64950, speed: 2.18 step/s
    eval  on dev  loss: 0.82768482, accu: 0.6357403
    global step 1210, epoch: 1, batch: 1210, loss: 0.79707, accu: 0.67500, speed: 0.21 step/s
    global step 1220, epoch: 1, batch: 1220, loss: 0.98206, accu: 0.64000, speed: 2.19 step/s
    global step 1230, epoch: 1, batch: 1230, loss: 0.71661, accu: 0.65667, speed: 2.05 step/s
    global step 1240, epoch: 1, batch: 1240, loss: 0.79179, accu: 0.66750, speed: 2.06 step/s
    global step 1250, epoch: 1, batch: 1250, loss: 0.80008, accu: 0.65300, speed: 1.73 step/s
    global step 1260, epoch: 1, batch: 1260, loss: 0.68724, accu: 0.64667, speed: 2.04 step/s
    global step 1270, epoch: 1, batch: 1270, loss: 1.03846, accu: 0.65000, speed: 2.21 step/s
    global step 1280, epoch: 1, batch: 1280, loss: 0.77074, accu: 0.64250, speed: 1.95 step/s
    global step 1290, epoch: 1, batch: 1290, loss: 0.61175, accu: 0.64333, speed: 1.76 step/s
    global step 1300, epoch: 1, batch: 1300, loss: 0.74646, accu: 0.64450, speed: 1.88 step/s
    eval  on dev  loss: 0.79956192, accu: 0.65909578
    模型保存在 1300 步， 最佳eval准确度为0.65909578！
    global step 1310, epoch: 1, batch: 1310, loss: 0.99709, accu: 0.69500, speed: 0.16 step/s
    global step 1320, epoch: 1, batch: 1320, loss: 0.99347, accu: 0.67250, speed: 1.62 step/s
    global step 1330, epoch: 1, batch: 1330, loss: 0.81354, accu: 0.66667, speed: 1.75 step/s
    global step 1340, epoch: 1, batch: 1340, loss: 0.71296, accu: 0.66625, speed: 2.23 step/s
    global step 1350, epoch: 1, batch: 1350, loss: 0.75059, accu: 0.65600, speed: 1.86 step/s
    global step 1360, epoch: 1, batch: 1360, loss: 0.76990, accu: 0.66167, speed: 1.81 step/s
    global step 1370, epoch: 1, batch: 1370, loss: 1.03436, accu: 0.65286, speed: 1.86 step/s
    global step 1380, epoch: 1, batch: 1380, loss: 0.95691, accu: 0.65500, speed: 1.99 step/s
    global step 1390, epoch: 1, batch: 1390, loss: 0.69669, accu: 0.65667, speed: 1.70 step/s
    global step 1400, epoch: 1, batch: 1400, loss: 0.64169, accu: 0.65550, speed: 1.88 step/s
    eval  on dev  loss: 0.81665468, accu: 0.65438183
    global step 1410, epoch: 1, batch: 1410, loss: 0.85338, accu: 0.62500, speed: 0.21 step/s
    global step 1420, epoch: 1, batch: 1420, loss: 0.72271, accu: 0.64500, speed: 1.80 step/s
    global step 1430, epoch: 1, batch: 1430, loss: 0.66704, accu: 0.65167, speed: 2.21 step/s
    global step 1440, epoch: 1, batch: 1440, loss: 0.70549, accu: 0.66000, speed: 2.20 step/s
    global step 1450, epoch: 1, batch: 1450, loss: 1.05780, accu: 0.66000, speed: 2.03 step/s
    global step 1460, epoch: 1, batch: 1460, loss: 0.82373, accu: 0.65583, speed: 1.91 step/s
    global step 1470, epoch: 1, batch: 1470, loss: 0.90143, accu: 0.65357, speed: 1.85 step/s
    global step 1480, epoch: 1, batch: 1480, loss: 0.99474, accu: 0.64750, speed: 1.73 step/s
    global step 1490, epoch: 1, batch: 1490, loss: 0.87314, accu: 0.64889, speed: 1.64 step/s
    global step 1500, epoch: 1, batch: 1500, loss: 1.26072, accu: 0.64750, speed: 1.67 step/s
    eval  on dev  loss: 0.82370257, accu: 0.64923934
    global step 1510, epoch: 1, batch: 1510, loss: 0.99938, accu: 0.63000, speed: 0.21 step/s
    global step 1520, epoch: 1, batch: 1520, loss: 0.74188, accu: 0.65750, speed: 1.73 step/s
    global step 1530, epoch: 1, batch: 1530, loss: 0.98210, accu: 0.65500, speed: 2.04 step/s
    global step 1540, epoch: 1, batch: 1540, loss: 0.80816, accu: 0.65000, speed: 1.76 step/s
    global step 1550, epoch: 1, batch: 1550, loss: 1.03547, accu: 0.65500, speed: 1.92 step/s
    global step 1560, epoch: 1, batch: 1560, loss: 0.68978, accu: 0.67250, speed: 2.01 step/s
    global step 1570, epoch: 1, batch: 1570, loss: 0.81843, accu: 0.66714, speed: 1.68 step/s
    global step 1580, epoch: 1, batch: 1580, loss: 0.65557, accu: 0.66312, speed: 1.70 step/s
    global step 1590, epoch: 1, batch: 1590, loss: 0.78489, accu: 0.65778, speed: 2.23 step/s
    global step 1600, epoch: 1, batch: 1600, loss: 0.88561, accu: 0.66200, speed: 2.13 step/s
    eval  on dev  loss: 0.80971032, accu: 0.65545318
    global step 1610, epoch: 1, batch: 1610, loss: 1.07748, accu: 0.61000, speed: 0.21 step/s
    global step 1620, epoch: 1, batch: 1620, loss: 0.55523, accu: 0.64750, speed: 1.82 step/s
    global step 1630, epoch: 1, batch: 1630, loss: 0.70347, accu: 0.64500, speed: 1.79 step/s
    global step 1640, epoch: 1, batch: 1640, loss: 0.70438, accu: 0.64375, speed: 1.92 step/s
    global step 1650, epoch: 1, batch: 1650, loss: 0.74670, accu: 0.66300, speed: 1.82 step/s
    global step 1660, epoch: 1, batch: 1660, loss: 0.79041, accu: 0.67167, speed: 2.32 step/s
    global step 1670, epoch: 1, batch: 1670, loss: 0.95332, accu: 0.67571, speed: 1.90 step/s
    global step 1680, epoch: 1, batch: 1680, loss: 1.14410, accu: 0.66438, speed: 1.85 step/s
    global step 1690, epoch: 1, batch: 1690, loss: 0.71444, accu: 0.66556, speed: 1.78 step/s
    global step 1700, epoch: 1, batch: 1700, loss: 0.99640, accu: 0.66500, speed: 2.08 step/s
    eval  on dev  loss: 0.83359122, accu: 0.6582387
    global step 1710, epoch: 1, batch: 1710, loss: 0.70072, accu: 0.62500, speed: 0.21 step/s
    global step 1720, epoch: 1, batch: 1720, loss: 0.88859, accu: 0.65000, speed: 1.87 step/s
    global step 1730, epoch: 1, batch: 1730, loss: 0.68052, accu: 0.65667, speed: 1.81 step/s
    global step 1740, epoch: 1, batch: 1740, loss: 0.84132, accu: 0.65750, speed: 2.11 step/s
    global step 1750, epoch: 1, batch: 1750, loss: 0.94299, accu: 0.66900, speed: 1.92 step/s
    global step 1760, epoch: 1, batch: 1760, loss: 0.67155, accu: 0.67583, speed: 2.22 step/s
    global step 1770, epoch: 1, batch: 1770, loss: 0.90371, accu: 0.67357, speed: 1.86 step/s
    global step 1780, epoch: 1, batch: 1780, loss: 0.67240, accu: 0.67812, speed: 2.14 step/s
    global step 1790, epoch: 1, batch: 1790, loss: 0.56876, accu: 0.67611, speed: 1.66 step/s
    global step 1800, epoch: 1, batch: 1800, loss: 0.80860, accu: 0.67050, speed: 2.23 step/s
    eval  on dev  loss: 0.82318652, accu: 0.66102421
    模型保存在 1800 步， 最佳eval准确度为0.66102421！
    global step 1810, epoch: 1, batch: 1810, loss: 0.78073, accu: 0.69000, speed: 0.16 step/s
    global step 1820, epoch: 1, batch: 1820, loss: 0.71605, accu: 0.69250, speed: 1.75 step/s
    global step 1830, epoch: 1, batch: 1830, loss: 0.96416, accu: 0.68000, speed: 1.66 step/s
    global step 1840, epoch: 1, batch: 1840, loss: 1.00974, accu: 0.68125, speed: 1.86 step/s
    global step 1850, epoch: 1, batch: 1850, loss: 0.90829, accu: 0.67800, speed: 1.78 step/s
    global step 1860, epoch: 1, batch: 1860, loss: 0.76859, accu: 0.68417, speed: 1.76 step/s
    global step 1870, epoch: 1, batch: 1870, loss: 1.36621, accu: 0.68500, speed: 2.03 step/s
    global step 1880, epoch: 1, batch: 1880, loss: 0.73000, accu: 0.68312, speed: 1.99 step/s
    global step 1890, epoch: 1, batch: 1890, loss: 0.50641, accu: 0.68056, speed: 2.09 step/s
    global step 1900, epoch: 1, batch: 1900, loss: 0.99746, accu: 0.67550, speed: 1.79 step/s
    eval  on dev  loss: 0.80472779, accu: 0.65802443
    global step 1910, epoch: 1, batch: 1910, loss: 0.90758, accu: 0.60000, speed: 0.20 step/s
    global step 1920, epoch: 1, batch: 1920, loss: 0.81180, accu: 0.63500, speed: 1.65 step/s
    global step 1930, epoch: 1, batch: 1930, loss: 0.49471, accu: 0.66000, speed: 1.82 step/s
    global step 1940, epoch: 1, batch: 1940, loss: 0.93183, accu: 0.65125, speed: 1.76 step/s
    global step 1950, epoch: 1, batch: 1950, loss: 1.12643, accu: 0.64000, speed: 1.83 step/s
    global step 1960, epoch: 1, batch: 1960, loss: 1.27281, accu: 0.63167, speed: 1.76 step/s
    global step 1970, epoch: 1, batch: 1970, loss: 0.94313, accu: 0.63429, speed: 1.97 step/s
    global step 1980, epoch: 1, batch: 1980, loss: 0.84834, accu: 0.63125, speed: 1.66 step/s
    global step 1990, epoch: 1, batch: 1990, loss: 0.63157, accu: 0.63333, speed: 2.08 step/s
    global step 2000, epoch: 1, batch: 2000, loss: 0.65762, accu: 0.64200, speed: 1.70 step/s
    eval  on dev  loss: 0.83355778, accu: 0.64173988
    global step 2010, epoch: 1, batch: 2010, loss: 0.69389, accu: 0.63000, speed: 0.21 step/s
    global step 2020, epoch: 1, batch: 2020, loss: 0.62111, accu: 0.64750, speed: 1.74 step/s
    global step 2030, epoch: 1, batch: 2030, loss: 1.21887, accu: 0.64333, speed: 1.83 step/s
    global step 2040, epoch: 1, batch: 2040, loss: 1.01706, accu: 0.63500, speed: 1.76 step/s
    global step 2050, epoch: 1, batch: 2050, loss: 0.61565, accu: 0.64900, speed: 2.51 step/s
    global step 2060, epoch: 1, batch: 2060, loss: 0.71592, accu: 0.65000, speed: 1.85 step/s
    global step 2070, epoch: 1, batch: 2070, loss: 0.62547, accu: 0.66000, speed: 1.97 step/s
    global step 2080, epoch: 1, batch: 2080, loss: 1.09253, accu: 0.65250, speed: 2.00 step/s
    global step 2090, epoch: 1, batch: 2090, loss: 0.78285, accu: 0.65056, speed: 1.64 step/s
    global step 2100, epoch: 1, batch: 2100, loss: 0.80501, accu: 0.64350, speed: 1.96 step/s
    eval  on dev  loss: 0.79901415, accu: 0.65995286
    global step 2110, epoch: 1, batch: 2110, loss: 0.91953, accu: 0.60500, speed: 0.21 step/s
    global step 2120, epoch: 1, batch: 2120, loss: 0.71146, accu: 0.64000, speed: 1.66 step/s
    global step 2130, epoch: 1, batch: 2130, loss: 1.01302, accu: 0.62500, speed: 1.70 step/s
    global step 2140, epoch: 1, batch: 2140, loss: 0.74937, accu: 0.63625, speed: 1.74 step/s
    global step 2150, epoch: 1, batch: 2150, loss: 0.70096, accu: 0.64300, speed: 1.80 step/s
    global step 2160, epoch: 1, batch: 2160, loss: 0.51146, accu: 0.64333, speed: 1.75 step/s
    global step 2170, epoch: 1, batch: 2170, loss: 0.73789, accu: 0.64286, speed: 1.75 step/s
    global step 2180, epoch: 1, batch: 2180, loss: 1.00692, accu: 0.64438, speed: 1.86 step/s
    global step 2190, epoch: 1, batch: 2190, loss: 0.69140, accu: 0.64444, speed: 1.75 step/s
    global step 2200, epoch: 1, batch: 2200, loss: 0.86115, accu: 0.64900, speed: 1.86 step/s
    eval  on dev  loss: 0.81734127, accu: 0.6582387
    global step 2210, epoch: 1, batch: 2210, loss: 0.57194, accu: 0.63000, speed: 0.21 step/s
    global step 2220, epoch: 1, batch: 2220, loss: 0.93253, accu: 0.63500, speed: 2.41 step/s
    global step 2230, epoch: 1, batch: 2230, loss: 0.72161, accu: 0.63167, speed: 1.56 step/s
    global step 2240, epoch: 1, batch: 2240, loss: 0.92104, accu: 0.63500, speed: 1.93 step/s
    global step 2250, epoch: 1, batch: 2250, loss: 0.76234, accu: 0.64100, speed: 1.67 step/s
    global step 2260, epoch: 1, batch: 2260, loss: 0.84171, accu: 0.63500, speed: 1.97 step/s
    global step 2270, epoch: 1, batch: 2270, loss: 0.94007, accu: 0.63714, speed: 1.82 step/s
    global step 2280, epoch: 1, batch: 2280, loss: 0.94271, accu: 0.62938, speed: 1.85 step/s
    global step 2290, epoch: 1, batch: 2290, loss: 0.51407, accu: 0.63444, speed: 1.70 step/s
    global step 2300, epoch: 1, batch: 2300, loss: 0.60164, accu: 0.63550, speed: 1.87 step/s
    eval  on dev  loss: 0.80099446, accu: 0.65609599
    global step 2310, epoch: 1, batch: 2310, loss: 1.27848, accu: 0.61500, speed: 0.21 step/s
    global step 2320, epoch: 1, batch: 2320, loss: 0.66667, accu: 0.65250, speed: 1.74 step/s
    global step 2330, epoch: 1, batch: 2330, loss: 0.84800, accu: 0.66333, speed: 1.92 step/s
    global step 2340, epoch: 1, batch: 2340, loss: 1.05899, accu: 0.65250, speed: 2.29 step/s
    global step 2350, epoch: 1, batch: 2350, loss: 0.74581, accu: 0.64900, speed: 1.84 step/s
    global step 2360, epoch: 1, batch: 2360, loss: 1.00269, accu: 0.65750, speed: 1.81 step/s
    global step 2370, epoch: 1, batch: 2370, loss: 0.81223, accu: 0.66500, speed: 1.79 step/s
    global step 2380, epoch: 1, batch: 2380, loss: 0.70566, accu: 0.66625, speed: 1.58 step/s
    global step 2390, epoch: 1, batch: 2390, loss: 1.11628, accu: 0.66278, speed: 2.10 step/s
    global step 2400, epoch: 1, batch: 2400, loss: 0.93860, accu: 0.65900, speed: 1.87 step/s
    eval  on dev  loss: 0.80878526, accu: 0.65331048
    global step 2410, epoch: 1, batch: 2410, loss: 0.52924, accu: 0.63000, speed: 0.21 step/s
    global step 2420, epoch: 1, batch: 2420, loss: 0.70537, accu: 0.66250, speed: 1.67 step/s
    global step 2430, epoch: 1, batch: 2430, loss: 0.74585, accu: 0.66167, speed: 1.97 step/s
    global step 2440, epoch: 1, batch: 2440, loss: 0.98796, accu: 0.65750, speed: 1.75 step/s
    global step 2450, epoch: 1, batch: 2450, loss: 0.60435, accu: 0.66000, speed: 2.04 step/s
    global step 2460, epoch: 1, batch: 2460, loss: 0.86620, accu: 0.64917, speed: 2.04 step/s
    global step 2470, epoch: 1, batch: 2470, loss: 0.96104, accu: 0.65214, speed: 1.91 step/s
    global step 2480, epoch: 1, batch: 2480, loss: 0.53108, accu: 0.66250, speed: 2.33 step/s
    global step 2490, epoch: 1, batch: 2490, loss: 0.86858, accu: 0.66778, speed: 1.70 step/s
    global step 2500, epoch: 1, batch: 2500, loss: 0.83978, accu: 0.67050, speed: 1.98 step/s
    eval  on dev  loss: 0.8077386, accu: 0.65866724
    global step 2510, epoch: 1, batch: 2510, loss: 0.93295, accu: 0.68000, speed: 0.21 step/s
    global step 2520, epoch: 1, batch: 2520, loss: 0.85883, accu: 0.68500, speed: 1.70 step/s
    global step 2530, epoch: 1, batch: 2530, loss: 0.86212, accu: 0.70667, speed: 1.90 step/s
    global step 2540, epoch: 1, batch: 2540, loss: 1.07175, accu: 0.68125, speed: 1.98 step/s
    global step 2550, epoch: 1, batch: 2550, loss: 0.76824, accu: 0.67300, speed: 1.91 step/s
    global step 2560, epoch: 1, batch: 2560, loss: 0.90117, accu: 0.66167, speed: 2.05 step/s
    global step 2570, epoch: 1, batch: 2570, loss: 0.92606, accu: 0.66000, speed: 1.99 step/s
    global step 2580, epoch: 1, batch: 2580, loss: 0.99706, accu: 0.65625, speed: 1.88 step/s
    global step 2590, epoch: 1, batch: 2590, loss: 0.87677, accu: 0.65556, speed: 1.88 step/s
    global step 2600, epoch: 1, batch: 2600, loss: 1.38558, accu: 0.65250, speed: 1.83 step/s
    eval  on dev  loss: 0.7940383, accu: 0.65995286
    global step 2610, epoch: 1, batch: 2610, loss: 0.73948, accu: 0.62000, speed: 0.21 step/s
    global step 2620, epoch: 1, batch: 2620, loss: 0.91110, accu: 0.62250, speed: 1.84 step/s
    global step 2630, epoch: 1, batch: 2630, loss: 0.88494, accu: 0.63667, speed: 2.11 step/s
    global step 2640, epoch: 1, batch: 2640, loss: 0.63859, accu: 0.64000, speed: 1.57 step/s
    global step 2650, epoch: 1, batch: 2650, loss: 0.92009, accu: 0.64200, speed: 1.66 step/s
    global step 2660, epoch: 1, batch: 2660, loss: 0.68426, accu: 0.64667, speed: 2.07 step/s
    global step 2670, epoch: 1, batch: 2670, loss: 0.65169, accu: 0.64143, speed: 1.84 step/s
    global step 2680, epoch: 1, batch: 2680, loss: 0.93406, accu: 0.64312, speed: 2.18 step/s
    global step 2690, epoch: 1, batch: 2690, loss: 0.59322, accu: 0.64889, speed: 1.95 step/s
    global step 2700, epoch: 1, batch: 2700, loss: 0.94244, accu: 0.64700, speed: 2.00 step/s
    eval  on dev  loss: 0.80144584, accu: 0.65288194
    global step 2710, epoch: 1, batch: 2710, loss: 0.48502, accu: 0.69500, speed: 0.21 step/s
    global step 2720, epoch: 1, batch: 2720, loss: 0.85845, accu: 0.67000, speed: 1.90 step/s
    global step 2730, epoch: 1, batch: 2730, loss: 0.88215, accu: 0.65167, speed: 2.15 step/s
    global step 2740, epoch: 1, batch: 2740, loss: 0.80119, accu: 0.66375, speed: 1.92 step/s
    global step 2750, epoch: 1, batch: 2750, loss: 0.82193, accu: 0.66600, speed: 2.02 step/s
    global step 2760, epoch: 1, batch: 2760, loss: 0.43262, accu: 0.66667, speed: 1.99 step/s
    global step 2770, epoch: 1, batch: 2770, loss: 0.60010, accu: 0.67071, speed: 1.82 step/s
    global step 2780, epoch: 1, batch: 2780, loss: 0.88807, accu: 0.66750, speed: 2.12 step/s
    global step 2790, epoch: 1, batch: 2790, loss: 0.67042, accu: 0.66833, speed: 1.78 step/s
    global step 2800, epoch: 1, batch: 2800, loss: 0.68029, accu: 0.66850, speed: 1.74 step/s
    eval  on dev  loss: 0.82614052, accu: 0.65716735
    global step 2810, epoch: 1, batch: 2810, loss: 0.67573, accu: 0.66000, speed: 0.21 step/s
    global step 2820, epoch: 1, batch: 2820, loss: 1.30984, accu: 0.64750, speed: 2.21 step/s
    global step 2830, epoch: 1, batch: 2830, loss: 0.86723, accu: 0.65000, speed: 2.17 step/s
    global step 2840, epoch: 1, batch: 2840, loss: 0.79090, accu: 0.64125, speed: 1.55 step/s
    global step 2850, epoch: 1, batch: 2850, loss: 0.81577, accu: 0.64300, speed: 1.88 step/s
    global step 2860, epoch: 1, batch: 2860, loss: 0.77237, accu: 0.65000, speed: 1.69 step/s
    global step 2870, epoch: 1, batch: 2870, loss: 0.91332, accu: 0.65286, speed: 1.83 step/s
    global step 2880, epoch: 1, batch: 2880, loss: 0.44478, accu: 0.65063, speed: 1.86 step/s
    global step 2890, epoch: 1, batch: 2890, loss: 0.57355, accu: 0.65167, speed: 2.02 step/s
    global step 2900, epoch: 1, batch: 2900, loss: 0.89453, accu: 0.64900, speed: 1.82 step/s
    eval  on dev  loss: 0.80004764, accu: 0.65438183
    global step 2910, epoch: 1, batch: 2910, loss: 0.74248, accu: 0.66500, speed: 0.21 step/s
    global step 2920, epoch: 1, batch: 2920, loss: 0.74176, accu: 0.66500, speed: 1.69 step/s
    global step 2930, epoch: 1, batch: 2930, loss: 0.78287, accu: 0.65500, speed: 1.97 step/s
    global step 2940, epoch: 1, batch: 2940, loss: 0.75038, accu: 0.65375, speed: 2.03 step/s
    global step 2950, epoch: 1, batch: 2950, loss: 0.80057, accu: 0.66400, speed: 1.83 step/s
    global step 2960, epoch: 1, batch: 2960, loss: 1.14346, accu: 0.66500, speed: 1.87 step/s
    global step 2970, epoch: 1, batch: 2970, loss: 0.53647, accu: 0.67214, speed: 1.75 step/s
    global step 2980, epoch: 1, batch: 2980, loss: 0.64077, accu: 0.68063, speed: 1.75 step/s
    global step 2990, epoch: 1, batch: 2990, loss: 0.66081, accu: 0.68000, speed: 1.96 step/s
    global step 3000, epoch: 1, batch: 3000, loss: 1.32352, accu: 0.67750, speed: 2.00 step/s
    eval  on dev  loss: 0.79510653, accu: 0.65973859
    global step 3010, epoch: 1, batch: 3010, loss: 0.95762, accu: 0.64500, speed: 0.21 step/s
    global step 3020, epoch: 1, batch: 3020, loss: 0.79981, accu: 0.66000, speed: 1.99 step/s
    global step 3030, epoch: 1, batch: 3030, loss: 0.68160, accu: 0.65833, speed: 1.72 step/s
    global step 3040, epoch: 1, batch: 3040, loss: 0.81162, accu: 0.65375, speed: 1.87 step/s
    global step 3050, epoch: 1, batch: 3050, loss: 1.08320, accu: 0.66200, speed: 2.02 step/s
    global step 3060, epoch: 1, batch: 3060, loss: 0.74102, accu: 0.66417, speed: 1.95 step/s
    global step 3070, epoch: 1, batch: 3070, loss: 0.72124, accu: 0.65714, speed: 2.32 step/s
    global step 3080, epoch: 1, batch: 3080, loss: 0.51492, accu: 0.66500, speed: 1.69 step/s
    global step 3090, epoch: 1, batch: 3090, loss: 0.70120, accu: 0.66389, speed: 1.86 step/s
    global step 3100, epoch: 1, batch: 3100, loss: 0.83688, accu: 0.65950, speed: 1.81 step/s
    eval  on dev  loss: 0.80034053, accu: 0.66338119
    模型保存在 3100 步， 最佳eval准确度为0.66338119！
    global step 3110, epoch: 1, batch: 3110, loss: 0.62895, accu: 0.64000, speed: 0.16 step/s
    global step 3120, epoch: 1, batch: 3120, loss: 0.69512, accu: 0.61000, speed: 1.92 step/s
    global step 3130, epoch: 1, batch: 3130, loss: 0.65150, accu: 0.62667, speed: 2.03 step/s
    global step 3140, epoch: 1, batch: 3140, loss: 0.63753, accu: 0.64125, speed: 2.11 step/s
    global step 3150, epoch: 1, batch: 3150, loss: 1.06662, accu: 0.65000, speed: 1.79 step/s
    global step 3160, epoch: 1, batch: 3160, loss: 0.97003, accu: 0.65000, speed: 1.76 step/s
    global step 3170, epoch: 1, batch: 3170, loss: 0.54216, accu: 0.64929, speed: 1.98 step/s
    global step 3180, epoch: 1, batch: 3180, loss: 0.76906, accu: 0.65000, speed: 2.15 step/s
    global step 3190, epoch: 1, batch: 3190, loss: 0.91980, accu: 0.64944, speed: 2.03 step/s
    global step 3200, epoch: 1, batch: 3200, loss: 1.03564, accu: 0.65000, speed: 2.00 step/s
    eval  on dev  loss: 0.78995717, accu: 0.66316692
    global step 3210, epoch: 1, batch: 3210, loss: 0.83859, accu: 0.66500, speed: 0.21 step/s
    global step 3220, epoch: 1, batch: 3220, loss: 0.76186, accu: 0.67250, speed: 1.82 step/s
    global step 3230, epoch: 1, batch: 3230, loss: 0.84858, accu: 0.66167, speed: 1.63 step/s
    global step 3240, epoch: 1, batch: 3240, loss: 0.74887, accu: 0.65000, speed: 1.94 step/s
    global step 3250, epoch: 1, batch: 3250, loss: 0.70967, accu: 0.65000, speed: 1.90 step/s
    global step 3260, epoch: 1, batch: 3260, loss: 0.86670, accu: 0.64917, speed: 2.15 step/s
    global step 3270, epoch: 2, batch: 3, loss: 0.75220, accu: 0.65398, speed: 2.26 step/s
    global step 3280, epoch: 2, batch: 13, loss: 1.10453, accu: 0.65976, speed: 2.09 step/s
    global step 3290, epoch: 2, batch: 23, loss: 0.81256, accu: 0.66258, speed: 1.98 step/s
    global step 3300, epoch: 2, batch: 33, loss: 0.75562, accu: 0.66533, speed: 2.02 step/s
    eval  on dev  loss: 0.80205983, accu: 0.66466681
    模型保存在 3300 步， 最佳eval准确度为0.66466681！
    global step 3310, epoch: 2, batch: 43, loss: 0.62548, accu: 0.74000, speed: 0.16 step/s
    global step 3320, epoch: 2, batch: 53, loss: 1.12059, accu: 0.72250, speed: 1.90 step/s
    global step 3330, epoch: 2, batch: 63, loss: 0.91133, accu: 0.71167, speed: 1.82 step/s
    global step 3340, epoch: 2, batch: 73, loss: 0.77600, accu: 0.72000, speed: 2.17 step/s
    global step 3350, epoch: 2, batch: 83, loss: 0.70518, accu: 0.71300, speed: 1.96 step/s
    global step 3360, epoch: 2, batch: 93, loss: 0.66494, accu: 0.71333, speed: 1.70 step/s
    global step 3370, epoch: 2, batch: 103, loss: 0.78981, accu: 0.71143, speed: 2.45 step/s
    global step 3380, epoch: 2, batch: 113, loss: 1.04309, accu: 0.70688, speed: 1.73 step/s
    global step 3390, epoch: 2, batch: 123, loss: 0.55101, accu: 0.70778, speed: 2.17 step/s
    global step 3400, epoch: 2, batch: 133, loss: 0.71418, accu: 0.70600, speed: 1.82 step/s
    eval  on dev  loss: 0.79841977, accu: 0.66016713
    global step 3410, epoch: 2, batch: 143, loss: 1.00799, accu: 0.66500, speed: 0.21 step/s
    global step 3420, epoch: 2, batch: 153, loss: 0.58485, accu: 0.67750, speed: 2.32 step/s
    global step 3430, epoch: 2, batch: 163, loss: 0.60318, accu: 0.67500, speed: 2.05 step/s
    global step 3440, epoch: 2, batch: 173, loss: 0.65162, accu: 0.67375, speed: 1.72 step/s
    global step 3450, epoch: 2, batch: 183, loss: 0.57642, accu: 0.67600, speed: 1.84 step/s
    global step 3460, epoch: 2, batch: 193, loss: 0.63799, accu: 0.68500, speed: 1.58 step/s
    global step 3470, epoch: 2, batch: 203, loss: 0.66872, accu: 0.68143, speed: 2.07 step/s
    global step 3480, epoch: 2, batch: 213, loss: 0.90030, accu: 0.68812, speed: 2.15 step/s
    global step 3490, epoch: 2, batch: 223, loss: 0.75798, accu: 0.68889, speed: 1.88 step/s
    global step 3500, epoch: 2, batch: 233, loss: 0.69780, accu: 0.68300, speed: 1.86 step/s
    eval  on dev  loss: 0.80273438, accu: 0.66273838
    global step 3510, epoch: 2, batch: 243, loss: 0.52024, accu: 0.69000, speed: 0.21 step/s
    global step 3520, epoch: 2, batch: 253, loss: 0.79787, accu: 0.68000, speed: 1.80 step/s
    global step 3530, epoch: 2, batch: 263, loss: 0.77509, accu: 0.68000, speed: 1.87 step/s
    global step 3540, epoch: 2, batch: 273, loss: 0.70919, accu: 0.68750, speed: 1.85 step/s
    global step 3550, epoch: 2, batch: 283, loss: 0.90692, accu: 0.67500, speed: 2.05 step/s
    global step 3560, epoch: 2, batch: 293, loss: 0.72004, accu: 0.66750, speed: 1.79 step/s
    global step 3570, epoch: 2, batch: 303, loss: 0.58490, accu: 0.66357, speed: 1.81 step/s
    global step 3580, epoch: 2, batch: 313, loss: 0.95947, accu: 0.65687, speed: 2.07 step/s
    global step 3590, epoch: 2, batch: 323, loss: 0.45463, accu: 0.65389, speed: 1.92 step/s
    global step 3600, epoch: 2, batch: 333, loss: 0.83351, accu: 0.65200, speed: 1.76 step/s
    eval  on dev  loss: 0.80765027, accu: 0.65781016
    global step 3610, epoch: 2, batch: 343, loss: 0.51790, accu: 0.78500, speed: 0.21 step/s
    global step 3620, epoch: 2, batch: 353, loss: 0.64419, accu: 0.72250, speed: 1.83 step/s
    global step 3630, epoch: 2, batch: 363, loss: 0.79201, accu: 0.71000, speed: 2.04 step/s
    global step 3640, epoch: 2, batch: 373, loss: 0.63908, accu: 0.70375, speed: 1.50 step/s
    global step 3650, epoch: 2, batch: 383, loss: 0.63323, accu: 0.69600, speed: 1.67 step/s
    global step 3660, epoch: 2, batch: 393, loss: 0.57323, accu: 0.68833, speed: 1.58 step/s
    global step 3670, epoch: 2, batch: 403, loss: 0.72571, accu: 0.68714, speed: 1.70 step/s
    global step 3680, epoch: 2, batch: 413, loss: 0.53700, accu: 0.68625, speed: 1.60 step/s
    global step 3690, epoch: 2, batch: 423, loss: 0.85894, accu: 0.68500, speed: 1.84 step/s
    global step 3700, epoch: 2, batch: 433, loss: 0.73665, accu: 0.68200, speed: 1.83 step/s
    eval  on dev  loss: 0.80183113, accu: 0.66552389
    模型保存在 3700 步， 最佳eval准确度为0.66552389！
    global step 3710, epoch: 2, batch: 443, loss: 0.83825, accu: 0.70500, speed: 0.16 step/s
    global step 3720, epoch: 2, batch: 453, loss: 0.84138, accu: 0.69750, speed: 2.16 step/s
    global step 3730, epoch: 2, batch: 463, loss: 0.50845, accu: 0.68167, speed: 1.74 step/s
    global step 3740, epoch: 2, batch: 473, loss: 0.75225, accu: 0.67750, speed: 1.80 step/s
    global step 3750, epoch: 2, batch: 483, loss: 0.82532, accu: 0.67200, speed: 1.60 step/s
    global step 3760, epoch: 2, batch: 493, loss: 0.67880, accu: 0.67833, speed: 2.07 step/s
    global step 3770, epoch: 2, batch: 503, loss: 0.55985, accu: 0.68357, speed: 1.82 step/s
    global step 3780, epoch: 2, batch: 513, loss: 0.65104, accu: 0.69000, speed: 1.78 step/s
    global step 3790, epoch: 2, batch: 523, loss: 0.80040, accu: 0.68889, speed: 1.83 step/s
    global step 3800, epoch: 2, batch: 533, loss: 1.34018, accu: 0.68350, speed: 1.85 step/s
    eval  on dev  loss: 0.81632793, accu: 0.66016713
    global step 3810, epoch: 2, batch: 543, loss: 0.77622, accu: 0.63500, speed: 0.21 step/s
    global step 3820, epoch: 2, batch: 553, loss: 0.97100, accu: 0.64500, speed: 2.08 step/s
    global step 3830, epoch: 2, batch: 563, loss: 0.82548, accu: 0.66333, speed: 1.86 step/s
    global step 3840, epoch: 2, batch: 573, loss: 0.68312, accu: 0.67375, speed: 2.13 step/s
    global step 3850, epoch: 2, batch: 583, loss: 0.66755, accu: 0.67100, speed: 1.74 step/s
    global step 3860, epoch: 2, batch: 593, loss: 0.83802, accu: 0.67917, speed: 1.83 step/s
    global step 3870, epoch: 2, batch: 603, loss: 0.63794, accu: 0.68000, speed: 2.16 step/s
    global step 3880, epoch: 2, batch: 613, loss: 0.66122, accu: 0.68125, speed: 1.77 step/s
    global step 3890, epoch: 2, batch: 623, loss: 0.61903, accu: 0.68333, speed: 1.71 step/s
    global step 3900, epoch: 2, batch: 633, loss: 0.86326, accu: 0.68350, speed: 1.76 step/s
    eval  on dev  loss: 0.80445302, accu: 0.66145275
    global step 3910, epoch: 2, batch: 643, loss: 0.86478, accu: 0.68500, speed: 0.21 step/s
    global step 3920, epoch: 2, batch: 653, loss: 0.94249, accu: 0.69250, speed: 2.58 step/s
    global step 3930, epoch: 2, batch: 663, loss: 1.11287, accu: 0.67667, speed: 1.65 step/s
    global step 3940, epoch: 2, batch: 673, loss: 0.92369, accu: 0.66000, speed: 1.77 step/s
    global step 3950, epoch: 2, batch: 683, loss: 0.61824, accu: 0.66400, speed: 1.85 step/s
    global step 3960, epoch: 2, batch: 693, loss: 1.01485, accu: 0.65667, speed: 1.95 step/s
    global step 3970, epoch: 2, batch: 703, loss: 0.64442, accu: 0.66429, speed: 1.81 step/s
    global step 3980, epoch: 2, batch: 713, loss: 0.72841, accu: 0.66312, speed: 1.79 step/s
    global step 3990, epoch: 2, batch: 723, loss: 0.86091, accu: 0.66944, speed: 1.57 step/s
    global step 4000, epoch: 2, batch: 733, loss: 0.42414, accu: 0.67650, speed: 2.10 step/s
    eval  on dev  loss: 0.81257719, accu: 0.65931005
    global step 4010, epoch: 2, batch: 743, loss: 0.60243, accu: 0.70500, speed: 0.21 step/s
    global step 4020, epoch: 2, batch: 753, loss: 0.72468, accu: 0.69250, speed: 2.10 step/s
    global step 4030, epoch: 2, batch: 763, loss: 0.82465, accu: 0.66833, speed: 1.85 step/s
    global step 4040, epoch: 2, batch: 773, loss: 0.94732, accu: 0.66250, speed: 1.98 step/s
    global step 4050, epoch: 2, batch: 783, loss: 0.80727, accu: 0.67100, speed: 2.08 step/s
    global step 4060, epoch: 2, batch: 793, loss: 0.87535, accu: 0.66583, speed: 1.65 step/s
    global step 4070, epoch: 2, batch: 803, loss: 0.96470, accu: 0.66429, speed: 1.80 step/s
    global step 4080, epoch: 2, batch: 813, loss: 0.47750, accu: 0.66500, speed: 1.92 step/s
    global step 4090, epoch: 2, batch: 823, loss: 0.73601, accu: 0.66833, speed: 1.88 step/s
    global step 4100, epoch: 2, batch: 833, loss: 0.72477, accu: 0.67500, speed: 1.94 step/s
    eval  on dev  loss: 0.81891519, accu: 0.66059567
    global step 4110, epoch: 2, batch: 843, loss: 0.82771, accu: 0.65500, speed: 0.20 step/s
    global step 4120, epoch: 2, batch: 853, loss: 0.38340, accu: 0.67250, speed: 2.09 step/s
    global step 4130, epoch: 2, batch: 863, loss: 0.68871, accu: 0.69667, speed: 2.08 step/s
    global step 4140, epoch: 2, batch: 873, loss: 0.69376, accu: 0.69375, speed: 1.97 step/s
    global step 4150, epoch: 2, batch: 883, loss: 0.67848, accu: 0.68900, speed: 1.73 step/s
    global step 4160, epoch: 2, batch: 893, loss: 0.90088, accu: 0.70250, speed: 1.79 step/s
    global step 4170, epoch: 2, batch: 903, loss: 0.66440, accu: 0.69786, speed: 1.81 step/s
    global step 4180, epoch: 2, batch: 913, loss: 0.79882, accu: 0.69312, speed: 1.93 step/s
    global step 4190, epoch: 2, batch: 923, loss: 1.20396, accu: 0.69833, speed: 2.48 step/s
    global step 4200, epoch: 2, batch: 933, loss: 0.67364, accu: 0.69300, speed: 1.86 step/s
    eval  on dev  loss: 0.79905552, accu: 0.66273838
    global step 4210, epoch: 2, batch: 943, loss: 0.82527, accu: 0.71000, speed: 0.20 step/s
    global step 4220, epoch: 2, batch: 953, loss: 0.80942, accu: 0.69500, speed: 1.68 step/s
    global step 4230, epoch: 2, batch: 963, loss: 0.78035, accu: 0.67667, speed: 1.76 step/s
    global step 4240, epoch: 2, batch: 973, loss: 0.74138, accu: 0.67125, speed: 2.04 step/s
    global step 4250, epoch: 2, batch: 983, loss: 0.83043, accu: 0.67300, speed: 1.64 step/s
    global step 4260, epoch: 2, batch: 993, loss: 0.61312, accu: 0.68500, speed: 1.79 step/s
    global step 4270, epoch: 2, batch: 1003, loss: 0.51545, accu: 0.68571, speed: 1.67 step/s
    global step 4280, epoch: 2, batch: 1013, loss: 0.69003, accu: 0.68563, speed: 2.06 step/s
    global step 4290, epoch: 2, batch: 1023, loss: 0.95496, accu: 0.68222, speed: 2.20 step/s
    global step 4300, epoch: 2, batch: 1033, loss: 0.76795, accu: 0.68050, speed: 1.66 step/s
    eval  on dev  loss: 0.81479049, accu: 0.64623955
    global step 4310, epoch: 2, batch: 1043, loss: 0.69333, accu: 0.66500, speed: 0.21 step/s
    global step 4320, epoch: 2, batch: 1053, loss: 0.74593, accu: 0.67750, speed: 2.15 step/s
    global step 4330, epoch: 2, batch: 1063, loss: 0.86294, accu: 0.66333, speed: 1.56 step/s
    global step 4340, epoch: 2, batch: 1073, loss: 0.80201, accu: 0.66875, speed: 1.92 step/s
    global step 4350, epoch: 2, batch: 1083, loss: 0.56257, accu: 0.67300, speed: 1.93 step/s
    global step 4360, epoch: 2, batch: 1093, loss: 0.46705, accu: 0.67500, speed: 1.71 step/s
    global step 4370, epoch: 2, batch: 1103, loss: 0.60798, accu: 0.67643, speed: 1.72 step/s
    global step 4380, epoch: 2, batch: 1113, loss: 0.87586, accu: 0.67000, speed: 2.16 step/s
    global step 4390, epoch: 2, batch: 1123, loss: 0.70918, accu: 0.67111, speed: 1.69 step/s
    global step 4400, epoch: 2, batch: 1133, loss: 0.75910, accu: 0.67700, speed: 1.52 step/s
    eval  on dev  loss: 0.81914282, accu: 0.66595243
    模型保存在 4400 步， 最佳eval准确度为0.66595243！
    global step 4410, epoch: 2, batch: 1143, loss: 0.99554, accu: 0.58000, speed: 0.16 step/s
    global step 4420, epoch: 2, batch: 1153, loss: 0.98512, accu: 0.62000, speed: 1.79 step/s
    global step 4430, epoch: 2, batch: 1163, loss: 0.56977, accu: 0.64500, speed: 2.02 step/s
    global step 4440, epoch: 2, batch: 1173, loss: 0.59268, accu: 0.65750, speed: 1.93 step/s
    global step 4450, epoch: 2, batch: 1183, loss: 0.58146, accu: 0.66400, speed: 2.03 step/s
    global step 4460, epoch: 2, batch: 1193, loss: 1.07594, accu: 0.66167, speed: 1.94 step/s
    global step 4470, epoch: 2, batch: 1203, loss: 0.60523, accu: 0.66929, speed: 1.63 step/s
    global step 4480, epoch: 2, batch: 1213, loss: 0.54619, accu: 0.67188, speed: 1.87 step/s
    global step 4490, epoch: 2, batch: 1223, loss: 0.86720, accu: 0.67111, speed: 2.02 step/s
    global step 4500, epoch: 2, batch: 1233, loss: 0.54052, accu: 0.67450, speed: 1.93 step/s
    eval  on dev  loss: 0.82425016, accu: 0.65588172
    global step 4510, epoch: 2, batch: 1243, loss: 0.77839, accu: 0.61500, speed: 0.21 step/s
    global step 4520, epoch: 2, batch: 1253, loss: 0.50854, accu: 0.65250, speed: 2.06 step/s
    global step 4530, epoch: 2, batch: 1263, loss: 0.76805, accu: 0.67000, speed: 1.80 step/s
    global step 4540, epoch: 2, batch: 1273, loss: 0.46902, accu: 0.67500, speed: 1.89 step/s
    global step 4550, epoch: 2, batch: 1283, loss: 0.67757, accu: 0.68200, speed: 1.78 step/s
    global step 4560, epoch: 2, batch: 1293, loss: 0.69832, accu: 0.68583, speed: 1.77 step/s
    global step 4570, epoch: 2, batch: 1303, loss: 0.69800, accu: 0.68571, speed: 2.16 step/s
    global step 4580, epoch: 2, batch: 1313, loss: 0.67550, accu: 0.67875, speed: 1.79 step/s
    global step 4590, epoch: 2, batch: 1323, loss: 0.75163, accu: 0.68167, speed: 1.88 step/s
    global step 4600, epoch: 2, batch: 1333, loss: 0.53255, accu: 0.68850, speed: 1.70 step/s
    eval  on dev  loss: 0.82795066, accu: 0.66123848
    global step 4610, epoch: 2, batch: 1343, loss: 0.53214, accu: 0.68500, speed: 0.21 step/s
    global step 4620, epoch: 2, batch: 1353, loss: 1.34168, accu: 0.67250, speed: 1.64 step/s
    global step 4630, epoch: 2, batch: 1363, loss: 0.71245, accu: 0.67500, speed: 2.07 step/s
    global step 4640, epoch: 2, batch: 1373, loss: 0.70767, accu: 0.67875, speed: 2.08 step/s
    global step 4650, epoch: 2, batch: 1383, loss: 0.88002, accu: 0.67600, speed: 2.04 step/s
    global step 4660, epoch: 2, batch: 1393, loss: 0.83579, accu: 0.69000, speed: 2.02 step/s
    global step 4670, epoch: 2, batch: 1403, loss: 0.85867, accu: 0.68500, speed: 1.85 step/s
    global step 4680, epoch: 2, batch: 1413, loss: 0.92358, accu: 0.68937, speed: 1.75 step/s
    global step 4690, epoch: 2, batch: 1423, loss: 0.65299, accu: 0.68889, speed: 1.83 step/s
    global step 4700, epoch: 2, batch: 1433, loss: 0.68043, accu: 0.68700, speed: 2.09 step/s
    eval  on dev  loss: 0.82563418, accu: 0.65438183
    global step 4710, epoch: 2, batch: 1443, loss: 0.77077, accu: 0.64500, speed: 0.21 step/s
    global step 4720, epoch: 2, batch: 1453, loss: 1.06732, accu: 0.64250, speed: 1.93 step/s
    global step 4730, epoch: 2, batch: 1463, loss: 0.82109, accu: 0.65000, speed: 1.85 step/s
    global step 4740, epoch: 2, batch: 1473, loss: 0.43879, accu: 0.65625, speed: 1.78 step/s
    global step 4750, epoch: 2, batch: 1483, loss: 0.62388, accu: 0.67100, speed: 1.94 step/s
    global step 4760, epoch: 2, batch: 1493, loss: 0.59731, accu: 0.67917, speed: 1.80 step/s
    global step 4770, epoch: 2, batch: 1503, loss: 0.78872, accu: 0.68071, speed: 1.71 step/s
    global step 4780, epoch: 2, batch: 1513, loss: 0.57818, accu: 0.68563, speed: 1.59 step/s
    global step 4790, epoch: 2, batch: 1523, loss: 0.86459, accu: 0.68889, speed: 2.00 step/s
    global step 4800, epoch: 2, batch: 1533, loss: 0.85720, accu: 0.68800, speed: 2.08 step/s
    eval  on dev  loss: 0.80853212, accu: 0.66316692
    global step 4810, epoch: 2, batch: 1543, loss: 0.75075, accu: 0.71500, speed: 0.21 step/s
    global step 4820, epoch: 2, batch: 1553, loss: 0.63892, accu: 0.72250, speed: 2.07 step/s
    global step 4830, epoch: 2, batch: 1563, loss: 1.25591, accu: 0.70167, speed: 1.98 step/s
    global step 4840, epoch: 2, batch: 1573, loss: 0.64098, accu: 0.70875, speed: 1.77 step/s
    global step 4850, epoch: 2, batch: 1583, loss: 0.45689, accu: 0.70600, speed: 2.13 step/s
    global step 4860, epoch: 2, batch: 1593, loss: 0.64455, accu: 0.70667, speed: 1.81 step/s
    global step 4870, epoch: 2, batch: 1603, loss: 1.01118, accu: 0.70000, speed: 1.79 step/s
    global step 4880, epoch: 2, batch: 1613, loss: 0.67485, accu: 0.69812, speed: 1.97 step/s
    global step 4890, epoch: 2, batch: 1623, loss: 0.68302, accu: 0.69944, speed: 1.71 step/s
    global step 4900, epoch: 2, batch: 1633, loss: 0.73176, accu: 0.70000, speed: 2.05 step/s
    eval  on dev  loss: 0.8363229, accu: 0.65566745
    global step 4910, epoch: 2, batch: 1643, loss: 0.70723, accu: 0.68500, speed: 0.21 step/s
    global step 4920, epoch: 2, batch: 1653, loss: 0.85267, accu: 0.69000, speed: 1.79 step/s
    global step 4930, epoch: 2, batch: 1663, loss: 0.86563, accu: 0.68000, speed: 1.65 step/s
    global step 4940, epoch: 2, batch: 1673, loss: 0.89467, accu: 0.66125, speed: 2.28 step/s
    global step 4950, epoch: 2, batch: 1683, loss: 0.76068, accu: 0.67500, speed: 1.63 step/s
    global step 4960, epoch: 2, batch: 1693, loss: 0.65857, accu: 0.67667, speed: 1.81 step/s
    global step 4970, epoch: 2, batch: 1703, loss: 0.60407, accu: 0.68286, speed: 2.13 step/s
    global step 4980, epoch: 2, batch: 1713, loss: 0.89349, accu: 0.67688, speed: 2.06 step/s
    global step 4990, epoch: 2, batch: 1723, loss: 0.79171, accu: 0.67833, speed: 2.41 step/s
    global step 5000, epoch: 2, batch: 1733, loss: 0.53474, accu: 0.68100, speed: 1.85 step/s
    eval  on dev  loss: 0.79984087, accu: 0.6567388
    global step 5010, epoch: 2, batch: 1743, loss: 0.63196, accu: 0.69500, speed: 0.21 step/s
    global step 5020, epoch: 2, batch: 1753, loss: 0.82068, accu: 0.67250, speed: 1.83 step/s
    global step 5030, epoch: 2, batch: 1763, loss: 0.72670, accu: 0.65000, speed: 2.22 step/s
    global step 5040, epoch: 2, batch: 1773, loss: 0.56122, accu: 0.64250, speed: 2.17 step/s
    global step 5050, epoch: 2, batch: 1783, loss: 0.46578, accu: 0.65700, speed: 1.82 step/s
    global step 5060, epoch: 2, batch: 1793, loss: 0.64105, accu: 0.66917, speed: 1.79 step/s
    global step 5070, epoch: 2, batch: 1803, loss: 0.60947, accu: 0.65929, speed: 2.06 step/s
    global step 5080, epoch: 2, batch: 1813, loss: 0.60859, accu: 0.65563, speed: 1.75 step/s
    global step 5090, epoch: 2, batch: 1823, loss: 0.89270, accu: 0.65500, speed: 2.22 step/s
    global step 5100, epoch: 2, batch: 1833, loss: 0.97202, accu: 0.65450, speed: 1.74 step/s
    eval  on dev  loss: 0.79701173, accu: 0.66209556
    global step 5110, epoch: 2, batch: 1843, loss: 0.76578, accu: 0.67500, speed: 0.21 step/s
    global step 5120, epoch: 2, batch: 1853, loss: 0.90802, accu: 0.69500, speed: 2.08 step/s
    global step 5130, epoch: 2, batch: 1863, loss: 0.74731, accu: 0.68833, speed: 1.77 step/s
    global step 5140, epoch: 2, batch: 1873, loss: 0.80501, accu: 0.68750, speed: 1.99 step/s
    global step 5150, epoch: 2, batch: 1883, loss: 0.91142, accu: 0.69300, speed: 1.70 step/s
    global step 5160, epoch: 2, batch: 1893, loss: 0.56924, accu: 0.69333, speed: 1.93 step/s
    global step 5170, epoch: 2, batch: 1903, loss: 0.48958, accu: 0.69357, speed: 1.86 step/s
    global step 5180, epoch: 2, batch: 1913, loss: 0.72334, accu: 0.68750, speed: 2.20 step/s
    global step 5190, epoch: 2, batch: 1923, loss: 0.60574, accu: 0.68667, speed: 2.02 step/s
    global step 5200, epoch: 2, batch: 1933, loss: 0.75697, accu: 0.68550, speed: 1.96 step/s
    eval  on dev  loss: 0.81202143, accu: 0.66016713
    global step 5210, epoch: 2, batch: 1943, loss: 0.81726, accu: 0.70000, speed: 0.20 step/s
    global step 5220, epoch: 2, batch: 1953, loss: 1.10298, accu: 0.68000, speed: 1.85 step/s
    global step 5230, epoch: 2, batch: 1963, loss: 1.03148, accu: 0.67667, speed: 1.62 step/s
    global step 5240, epoch: 2, batch: 1973, loss: 0.58473, accu: 0.68500, speed: 2.38 step/s
    global step 5250, epoch: 2, batch: 1983, loss: 0.91917, accu: 0.69700, speed: 1.73 step/s
    global step 5260, epoch: 2, batch: 1993, loss: 0.61960, accu: 0.70167, speed: 1.57 step/s
    global step 5270, epoch: 2, batch: 2003, loss: 0.50324, accu: 0.70214, speed: 1.63 step/s
    global step 5280, epoch: 2, batch: 2013, loss: 0.68795, accu: 0.70875, speed: 2.21 step/s
    global step 5290, epoch: 2, batch: 2023, loss: 1.07804, accu: 0.70278, speed: 2.06 step/s
    global step 5300, epoch: 2, batch: 2033, loss: 0.75577, accu: 0.69850, speed: 2.22 step/s
    eval  on dev  loss: 0.80125755, accu: 0.65716735
    global step 5310, epoch: 2, batch: 2043, loss: 0.94171, accu: 0.64000, speed: 0.21 step/s
    global step 5320, epoch: 2, batch: 2053, loss: 0.54871, accu: 0.67250, speed: 1.91 step/s
    global step 5330, epoch: 2, batch: 2063, loss: 0.62728, accu: 0.67500, speed: 1.99 step/s
    global step 5340, epoch: 2, batch: 2073, loss: 1.05463, accu: 0.66500, speed: 2.24 step/s
    global step 5350, epoch: 2, batch: 2083, loss: 0.55826, accu: 0.66500, speed: 2.02 step/s
    global step 5360, epoch: 2, batch: 2093, loss: 0.64697, accu: 0.66917, speed: 1.82 step/s
    global step 5370, epoch: 2, batch: 2103, loss: 0.77227, accu: 0.65786, speed: 1.89 step/s
    global step 5380, epoch: 2, batch: 2113, loss: 0.88052, accu: 0.65438, speed: 1.80 step/s
    global step 5390, epoch: 2, batch: 2123, loss: 0.77476, accu: 0.66056, speed: 1.94 step/s
    global step 5400, epoch: 2, batch: 2133, loss: 0.47553, accu: 0.67050, speed: 1.86 step/s
    eval  on dev  loss: 0.8251524, accu: 0.6582387
    global step 5410, epoch: 2, batch: 2143, loss: 0.65684, accu: 0.65500, speed: 0.21 step/s
    global step 5420, epoch: 2, batch: 2153, loss: 0.69941, accu: 0.69000, speed: 2.00 step/s
    global step 5430, epoch: 2, batch: 2163, loss: 0.58475, accu: 0.67833, speed: 1.90 step/s
    global step 5440, epoch: 2, batch: 2173, loss: 0.65651, accu: 0.67625, speed: 1.90 step/s
    global step 5450, epoch: 2, batch: 2183, loss: 0.51087, accu: 0.68100, speed: 1.79 step/s
    global step 5460, epoch: 2, batch: 2193, loss: 0.52662, accu: 0.67500, speed: 2.05 step/s
    global step 5470, epoch: 2, batch: 2203, loss: 0.85005, accu: 0.68000, speed: 2.06 step/s
    global step 5480, epoch: 2, batch: 2213, loss: 0.57512, accu: 0.68188, speed: 2.31 step/s
    global step 5490, epoch: 2, batch: 2223, loss: 0.69191, accu: 0.68111, speed: 2.11 step/s
    global step 5500, epoch: 2, batch: 2233, loss: 0.66528, accu: 0.68900, speed: 1.90 step/s
    eval  on dev  loss: 0.80941546, accu: 0.66380973
    global step 5510, epoch: 2, batch: 2243, loss: 0.86500, accu: 0.62500, speed: 0.20 step/s
    global step 5520, epoch: 2, batch: 2253, loss: 0.78940, accu: 0.63000, speed: 1.75 step/s
    global step 5530, epoch: 2, batch: 2263, loss: 0.81701, accu: 0.64333, speed: 2.20 step/s
    global step 5540, epoch: 2, batch: 2273, loss: 0.90769, accu: 0.64625, speed: 1.87 step/s
    global step 5550, epoch: 2, batch: 2283, loss: 0.68321, accu: 0.65600, speed: 1.93 step/s
    global step 5560, epoch: 2, batch: 2293, loss: 0.83902, accu: 0.65583, speed: 1.92 step/s
    global step 5570, epoch: 2, batch: 2303, loss: 1.17842, accu: 0.65500, speed: 1.90 step/s
    global step 5580, epoch: 2, batch: 2313, loss: 0.54290, accu: 0.64812, speed: 1.73 step/s
    global step 5590, epoch: 2, batch: 2323, loss: 0.74514, accu: 0.65167, speed: 1.85 step/s
    global step 5600, epoch: 2, batch: 2333, loss: 1.00223, accu: 0.64750, speed: 1.76 step/s
    eval  on dev  loss: 0.79813039, accu: 0.66188129
    global step 5610, epoch: 2, batch: 2343, loss: 0.65761, accu: 0.72000, speed: 0.21 step/s
    global step 5620, epoch: 2, batch: 2353, loss: 0.91459, accu: 0.71750, speed: 1.88 step/s
    global step 5630, epoch: 2, batch: 2363, loss: 0.62412, accu: 0.70167, speed: 1.83 step/s
    global step 5640, epoch: 2, batch: 2373, loss: 1.11975, accu: 0.69875, speed: 2.21 step/s
    global step 5650, epoch: 2, batch: 2383, loss: 0.95319, accu: 0.69900, speed: 1.93 step/s
    global step 5660, epoch: 2, batch: 2393, loss: 0.76180, accu: 0.69333, speed: 1.80 step/s
    global step 5670, epoch: 2, batch: 2403, loss: 0.69035, accu: 0.68786, speed: 1.79 step/s
    global step 5680, epoch: 2, batch: 2413, loss: 0.34862, accu: 0.68937, speed: 2.18 step/s
    global step 5690, epoch: 2, batch: 2423, loss: 0.80374, accu: 0.68722, speed: 2.10 step/s
    global step 5700, epoch: 2, batch: 2433, loss: 0.67627, accu: 0.68850, speed: 1.73 step/s
    eval  on dev  loss: 0.80125964, accu: 0.66552389
    global step 5710, epoch: 2, batch: 2443, loss: 0.55802, accu: 0.66000, speed: 0.21 step/s
    global step 5720, epoch: 2, batch: 2453, loss: 0.57294, accu: 0.68750, speed: 1.74 step/s
    global step 5730, epoch: 2, batch: 2463, loss: 0.92981, accu: 0.67500, speed: 1.74 step/s
    global step 5740, epoch: 2, batch: 2473, loss: 0.70119, accu: 0.67125, speed: 1.85 step/s
    global step 5750, epoch: 2, batch: 2483, loss: 0.53571, accu: 0.67400, speed: 2.18 step/s
    global step 5760, epoch: 2, batch: 2493, loss: 0.87481, accu: 0.67333, speed: 1.94 step/s
    global step 5770, epoch: 2, batch: 2503, loss: 0.57328, accu: 0.67000, speed: 2.19 step/s
    global step 5780, epoch: 2, batch: 2513, loss: 0.73697, accu: 0.67188, speed: 1.84 step/s
    global step 5790, epoch: 2, batch: 2523, loss: 1.01587, accu: 0.67889, speed: 1.75 step/s
    global step 5800, epoch: 2, batch: 2533, loss: 0.58864, accu: 0.67800, speed: 1.89 step/s
    eval  on dev  loss: 0.80294585, accu: 0.67173773
    模型保存在 5800 步， 最佳eval准确度为0.67173773！
    global step 5810, epoch: 2, batch: 2543, loss: 1.10729, accu: 0.64500, speed: 0.16 step/s
    global step 5820, epoch: 2, batch: 2553, loss: 0.62812, accu: 0.65500, speed: 1.87 step/s
    global step 5830, epoch: 2, batch: 2563, loss: 0.66229, accu: 0.65333, speed: 1.66 step/s
    global step 5840, epoch: 2, batch: 2573, loss: 0.53786, accu: 0.66250, speed: 1.59 step/s
    global step 5850, epoch: 2, batch: 2583, loss: 0.65355, accu: 0.66500, speed: 1.96 step/s
    global step 5860, epoch: 2, batch: 2593, loss: 1.19095, accu: 0.66750, speed: 1.84 step/s
    global step 5870, epoch: 2, batch: 2603, loss: 1.17971, accu: 0.66643, speed: 2.20 step/s
    global step 5880, epoch: 2, batch: 2613, loss: 1.11371, accu: 0.66938, speed: 1.70 step/s
    global step 5890, epoch: 2, batch: 2623, loss: 0.53809, accu: 0.67278, speed: 2.04 step/s
    global step 5900, epoch: 2, batch: 2633, loss: 0.78161, accu: 0.67000, speed: 1.83 step/s
    eval  on dev  loss: 0.80315292, accu: 0.66166702
    global step 5910, epoch: 2, batch: 2643, loss: 0.96686, accu: 0.65000, speed: 0.21 step/s
    global step 5920, epoch: 2, batch: 2653, loss: 0.64106, accu: 0.67750, speed: 1.98 step/s
    global step 5930, epoch: 2, batch: 2663, loss: 0.80597, accu: 0.68333, speed: 1.78 step/s
    global step 5940, epoch: 2, batch: 2673, loss: 0.71285, accu: 0.67625, speed: 1.99 step/s
    global step 5950, epoch: 2, batch: 2683, loss: 0.75426, accu: 0.66300, speed: 1.95 step/s
    global step 5960, epoch: 2, batch: 2693, loss: 0.60351, accu: 0.66750, speed: 1.78 step/s
    global step 5970, epoch: 2, batch: 2703, loss: 0.43200, accu: 0.67714, speed: 1.78 step/s
    global step 5980, epoch: 2, batch: 2713, loss: 0.72820, accu: 0.67937, speed: 1.64 step/s
    global step 5990, epoch: 2, batch: 2723, loss: 0.83258, accu: 0.67500, speed: 1.88 step/s
    global step 6000, epoch: 2, batch: 2733, loss: 0.80078, accu: 0.67700, speed: 2.14 step/s
    eval  on dev  loss: 0.79674768, accu: 0.66059567
    global step 6010, epoch: 2, batch: 2743, loss: 0.76626, accu: 0.64000, speed: 0.21 step/s
    global step 6020, epoch: 2, batch: 2753, loss: 0.91190, accu: 0.65500, speed: 1.91 step/s
    global step 6030, epoch: 2, batch: 2763, loss: 0.79623, accu: 0.67500, speed: 2.04 step/s
    global step 6040, epoch: 2, batch: 2773, loss: 0.56865, accu: 0.67125, speed: 1.94 step/s
    global step 6050, epoch: 2, batch: 2783, loss: 0.74231, accu: 0.66900, speed: 1.84 step/s
    global step 6060, epoch: 2, batch: 2793, loss: 0.91888, accu: 0.66167, speed: 2.07 step/s
    global step 6070, epoch: 2, batch: 2803, loss: 0.78122, accu: 0.66643, speed: 1.79 step/s
    global step 6080, epoch: 2, batch: 2813, loss: 0.59712, accu: 0.66812, speed: 2.10 step/s
    global step 6090, epoch: 2, batch: 2823, loss: 0.76330, accu: 0.66278, speed: 1.76 step/s
    global step 6100, epoch: 2, batch: 2833, loss: 0.44343, accu: 0.66650, speed: 1.84 step/s
    eval  on dev  loss: 0.80275184, accu: 0.65588172
    global step 6110, epoch: 2, batch: 2843, loss: 0.79137, accu: 0.66000, speed: 0.21 step/s
    global step 6120, epoch: 2, batch: 2853, loss: 1.02464, accu: 0.67000, speed: 1.93 step/s
    global step 6130, epoch: 2, batch: 2863, loss: 0.76288, accu: 0.68167, speed: 2.26 step/s
    global step 6140, epoch: 2, batch: 2873, loss: 0.82862, accu: 0.67750, speed: 2.05 step/s
    global step 6150, epoch: 2, batch: 2883, loss: 0.78840, accu: 0.68400, speed: 2.02 step/s
    global step 6160, epoch: 2, batch: 2893, loss: 0.80317, accu: 0.67750, speed: 1.79 step/s
    global step 6170, epoch: 2, batch: 2903, loss: 0.81730, accu: 0.67786, speed: 1.87 step/s
    global step 6180, epoch: 2, batch: 2913, loss: 1.01026, accu: 0.67875, speed: 1.77 step/s
    global step 6190, epoch: 2, batch: 2923, loss: 0.57841, accu: 0.68167, speed: 1.87 step/s
    global step 6200, epoch: 2, batch: 2933, loss: 0.64243, accu: 0.68200, speed: 2.27 step/s
    eval  on dev  loss: 0.81299204, accu: 0.65781016
    global step 6210, epoch: 2, batch: 2943, loss: 0.86922, accu: 0.70500, speed: 0.21 step/s
    global step 6220, epoch: 2, batch: 2953, loss: 0.73485, accu: 0.69500, speed: 2.29 step/s
    global step 6230, epoch: 2, batch: 2963, loss: 0.89662, accu: 0.67167, speed: 1.83 step/s
    global step 6240, epoch: 2, batch: 2973, loss: 0.84184, accu: 0.67750, speed: 1.94 step/s
    global step 6250, epoch: 2, batch: 2983, loss: 0.63574, accu: 0.67700, speed: 1.84 step/s
    global step 6260, epoch: 2, batch: 2993, loss: 0.68008, accu: 0.67250, speed: 1.78 step/s
    global step 6270, epoch: 2, batch: 3003, loss: 0.80584, accu: 0.66786, speed: 1.86 step/s
    global step 6280, epoch: 2, batch: 3013, loss: 0.49548, accu: 0.67125, speed: 2.02 step/s
    global step 6290, epoch: 2, batch: 3023, loss: 0.98110, accu: 0.66944, speed: 1.89 step/s
    global step 6300, epoch: 2, batch: 3033, loss: 0.83601, accu: 0.67100, speed: 1.68 step/s
    eval  on dev  loss: 0.80394763, accu: 0.65952432
    global step 6310, epoch: 2, batch: 3043, loss: 0.68580, accu: 0.66500, speed: 0.21 step/s
    global step 6320, epoch: 2, batch: 3053, loss: 0.87772, accu: 0.67500, speed: 1.89 step/s
    global step 6330, epoch: 2, batch: 3063, loss: 0.96471, accu: 0.68167, speed: 1.92 step/s
    global step 6340, epoch: 2, batch: 3073, loss: 0.91486, accu: 0.68875, speed: 1.90 step/s
    global step 6350, epoch: 2, batch: 3083, loss: 0.77587, accu: 0.70300, speed: 2.02 step/s
    global step 6360, epoch: 2, batch: 3093, loss: 0.87745, accu: 0.69500, speed: 1.61 step/s
    global step 6370, epoch: 2, batch: 3103, loss: 0.82805, accu: 0.69643, speed: 1.62 step/s
    global step 6380, epoch: 2, batch: 3113, loss: 0.74532, accu: 0.70375, speed: 1.70 step/s
    global step 6390, epoch: 2, batch: 3123, loss: 0.57850, accu: 0.70389, speed: 2.06 step/s
    global step 6400, epoch: 2, batch: 3133, loss: 1.11012, accu: 0.70050, speed: 1.83 step/s
    eval  on dev  loss: 0.8044191, accu: 0.66488108
    global step 6410, epoch: 2, batch: 3143, loss: 1.09634, accu: 0.66000, speed: 0.21 step/s
    global step 6420, epoch: 2, batch: 3153, loss: 0.76334, accu: 0.65250, speed: 1.90 step/s
    global step 6430, epoch: 2, batch: 3163, loss: 0.60661, accu: 0.67667, speed: 2.09 step/s
    global step 6440, epoch: 2, batch: 3173, loss: 0.98321, accu: 0.68750, speed: 1.72 step/s
    global step 6450, epoch: 2, batch: 3183, loss: 0.65365, accu: 0.68600, speed: 1.87 step/s
    global step 6460, epoch: 2, batch: 3193, loss: 0.83521, accu: 0.67667, speed: 1.57 step/s
    global step 6470, epoch: 2, batch: 3203, loss: 0.61826, accu: 0.68000, speed: 1.65 step/s
    global step 6480, epoch: 2, batch: 3213, loss: 0.64479, accu: 0.68437, speed: 1.87 step/s
    global step 6490, epoch: 2, batch: 3223, loss: 0.86745, accu: 0.68611, speed: 1.58 step/s
    global step 6500, epoch: 2, batch: 3233, loss: 0.88337, accu: 0.68900, speed: 1.85 step/s
    eval  on dev  loss: 0.79665518, accu: 0.6661667
    global step 6510, epoch: 2, batch: 3243, loss: 0.59023, accu: 0.69500, speed: 0.21 step/s
    global step 6520, epoch: 2, batch: 3253, loss: 0.56851, accu: 0.69250, speed: 1.65 step/s
    global step 6530, epoch: 2, batch: 3263, loss: 0.77030, accu: 0.70500, speed: 1.74 step/s
    global step 6540, epoch: 3, batch: 6, loss: 0.56204, accu: 0.71122, speed: 2.17 step/s
    global step 6550, epoch: 3, batch: 16, loss: 1.04900, accu: 0.71601, speed: 1.72 step/s
    global step 6560, epoch: 3, batch: 26, loss: 0.32134, accu: 0.72506, speed: 1.69 step/s
    global step 6570, epoch: 3, batch: 36, loss: 0.79475, accu: 0.72290, speed: 2.07 step/s
    global step 6580, epoch: 3, batch: 46, loss: 0.70526, accu: 0.72379, speed: 2.18 step/s
    global step 6590, epoch: 3, batch: 56, loss: 0.79080, accu: 0.71891, speed: 1.67 step/s
    global step 6600, epoch: 3, batch: 66, loss: 0.42509, accu: 0.72052, speed: 1.82 step/s
    eval  on dev  loss: 0.8246482, accu: 0.66295265
    global step 6610, epoch: 3, batch: 76, loss: 0.61388, accu: 0.75000, speed: 0.21 step/s
    global step 6620, epoch: 3, batch: 86, loss: 0.56371, accu: 0.74000, speed: 1.64 step/s
    global step 6630, epoch: 3, batch: 96, loss: 0.52937, accu: 0.76000, speed: 1.74 step/s
    global step 6640, epoch: 3, batch: 106, loss: 0.59531, accu: 0.75500, speed: 1.98 step/s
    global step 6650, epoch: 3, batch: 116, loss: 1.00811, accu: 0.75300, speed: 2.15 step/s
    global step 6660, epoch: 3, batch: 126, loss: 0.65118, accu: 0.74333, speed: 1.59 step/s
    global step 6670, epoch: 3, batch: 136, loss: 0.51896, accu: 0.74357, speed: 2.05 step/s
    global step 6680, epoch: 3, batch: 146, loss: 0.60242, accu: 0.73687, speed: 2.01 step/s
    global step 6690, epoch: 3, batch: 156, loss: 0.49864, accu: 0.73667, speed: 2.04 step/s
    global step 6700, epoch: 3, batch: 166, loss: 0.37496, accu: 0.73950, speed: 2.21 step/s
    eval  on dev  loss: 0.8496905, accu: 0.65888151
    global step 6710, epoch: 3, batch: 176, loss: 0.75812, accu: 0.73500, speed: 0.21 step/s
    global step 6720, epoch: 3, batch: 186, loss: 0.65392, accu: 0.74500, speed: 1.58 step/s
    global step 6730, epoch: 3, batch: 196, loss: 0.49644, accu: 0.74167, speed: 1.85 step/s
    global step 6740, epoch: 3, batch: 206, loss: 0.82362, accu: 0.74125, speed: 1.61 step/s
    global step 6750, epoch: 3, batch: 216, loss: 0.68475, accu: 0.74400, speed: 2.18 step/s
    global step 6760, epoch: 3, batch: 226, loss: 0.80333, accu: 0.74333, speed: 1.64 step/s
    global step 6770, epoch: 3, batch: 236, loss: 0.37657, accu: 0.73571, speed: 1.86 step/s
    global step 6780, epoch: 3, batch: 246, loss: 0.80611, accu: 0.73813, speed: 1.69 step/s
    global step 6790, epoch: 3, batch: 256, loss: 0.87477, accu: 0.73389, speed: 1.88 step/s
    global step 6800, epoch: 3, batch: 266, loss: 0.83531, accu: 0.72650, speed: 1.78 step/s
    eval  on dev  loss: 0.81505877, accu: 0.65995286
    global step 6810, epoch: 3, batch: 276, loss: 0.61799, accu: 0.71500, speed: 0.21 step/s
    global step 6820, epoch: 3, batch: 286, loss: 0.86069, accu: 0.71750, speed: 1.80 step/s
    global step 6830, epoch: 3, batch: 296, loss: 0.80143, accu: 0.70833, speed: 2.06 step/s
    global step 6840, epoch: 3, batch: 306, loss: 0.89584, accu: 0.69375, speed: 1.95 step/s
    global step 6850, epoch: 3, batch: 316, loss: 0.45910, accu: 0.70400, speed: 2.33 step/s
    global step 6860, epoch: 3, batch: 326, loss: 0.78524, accu: 0.70250, speed: 1.70 step/s
    global step 6870, epoch: 3, batch: 336, loss: 0.44477, accu: 0.71143, speed: 2.04 step/s
    global step 6880, epoch: 3, batch: 346, loss: 0.86213, accu: 0.70937, speed: 1.85 step/s



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-14-66a2d518ca10> in <module>
         12         input_ids, token_type_ids, labels = batch
         13         # 喂数据给model
    ---> 14         logits = model(input_ids, token_type_ids)
         15         # 计算损失函数值
         16         loss = criterion(logits, labels)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlenlp/transformers/skep/modeling.py in forward(self, input_ids, token_type_ids, position_ids, attention_mask)
        407             token_type_ids=token_type_ids,
        408             position_ids=position_ids,
    --> 409             attention_mask=attention_mask)
        410 
        411         pooled_output = self.dropout(pooled_output)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlenlp/transformers/skep/modeling.py in forward(self, input_ids, token_type_ids, position_ids, attention_mask)
        326             position_ids=position_ids,
        327             token_type_ids=token_type_ids)
    --> 328         encoder_outputs = self.encoder(embedding_output, attention_mask)
        329         sequence_output = encoder_outputs
        330         pooled_output = self.pooler(sequence_output)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/transformer.py in forward(self, src, src_mask, cache)
        681         for i, mod in enumerate(self.layers):
        682             if cache is None:
    --> 683                 output = mod(output, src_mask=src_mask)
        684             else:
        685                 output, new_cache = mod(output,


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/transformer.py in forward(self, src, src_mask, cache)
        565         # Add cache for encoder for the usage like UniLM
        566         if cache is None:
    --> 567             src = self.self_attn(src, src, src, src_mask)
        568         else:
        569             src, incremental_cache = self.self_attn(src, src, src, src_mask,


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/transformer.py in forward(self, query, key, value, attn_mask, cache)
        392         # compute q ,k ,v
        393         if cache is None:
    --> 394             q, k, v = self._prepare_qkv(query, key, value, cache)
        395         else:
        396             q, k, v, cache = self._prepare_qkv(query, key, value, cache)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/transformer.py in _prepare_qkv(self, query, key, value, cache)
        226             k, v = cache.k, cache.v
        227         else:
    --> 228             k, v = self.compute_kv(key, value)
        229 
        230         if isinstance(cache, self.Cache):


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/transformer.py in compute_kv(self, key, value)
        260                 and their data types are same as inputs.
        261         """
    --> 262         k = self.k_proj(key)
        263         v = self.v_proj(value)
        264         k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/common.py in forward(self, input)
        127     def forward(self, input):
        128         out = F.linear(
    --> 129             x=input, weight=self.weight, bias=self.bias, name=self.name)
        130         return out
        131 


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/functional/common.py in linear(x, weight, bias, name)
       1449         pre_bias = _varbase_creator(dtype=x.dtype)
       1450         core.ops.matmul(x, weight, pre_bias, 'transpose_X', False,
    -> 1451                         'transpose_Y', False, "alpha", 1)
       1452         return dygraph_utils._append_bias_in_dygraph(
       1453             pre_bias, bias, axis=len(x.shape) - 1)


    KeyboardInterrupt: 


## 3.训练日志
visual dl 就不放了

```
模型保存在 2000 步， 最佳eval准确度为0.66757143！
global step 2010, epoch: 2, batch: 210, loss: 0.49394, accu: 0.66286, speed: 0.09 step/s
global step 2020, epoch: 2, batch: 220, loss: 0.66231, accu: 0.65429, speed: 0.90 step/s
global step 2030, epoch: 2, batch: 230, loss: 0.67677, accu: 0.66286, speed: 0.86 step/s
global step 2040, epoch: 2, batch: 240, loss: 0.75220, accu: 0.67143, speed: 0.90 step/s
global step 2050, epoch: 2, batch: 250, loss: 0.66303, accu: 0.67600, speed: 1.17 step/s
global step 2060, epoch: 2, batch: 260, loss: 0.67201, accu: 0.67857, speed: 0.79 step/s
global step 2070, epoch: 2, batch: 270, loss: 1.00059, accu: 0.67224, speed: 0.76 step/s
global step 2080, epoch: 2, batch: 280, loss: 0.74657, accu: 0.67786, speed: 0.93 step/s
global step 2090, epoch: 2, batch: 290, loss: 0.70754, accu: 0.67778, speed: 0.92 step/s
global step 2100, epoch: 2, batch: 300, loss: 0.74980, accu: 0.68257, speed: 0.84 step/s
```

# 六、预测提交结果

## 1.测试数据集处理


```python
test_ds =  load_dataset(read_test, data_path='测试集.csv',lazy=False)
# 在转换为MapDataset类型
test_ds = MapDataset(test_ds)
print(len(test_ds))
```

    30000



```python
import numpy as np
import paddle

# 处理测试集数据
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack() # qid
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 2.加载预测模型


```python
# 根据实际运行情况，更换加载的参数路径
params_path = 'best_checkpoint/best_model.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```

    Loaded parameters from best_checkpoint/best_model.pdparams


## 3.开始预测


```python
# 处理测试集数据
label_map = {0: '1', 1:'2', 2:'3', 3:'4',4:'5'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))
```

## 4.保存结果
根据官网要求写入文件（注意：此处数据集给的submission.csv并不对，严格按照官网来）


```python
# 写入预测结果
with open( "submission.csv", 'w', encoding="utf-8") as f:
    # f.write("数据ID,评分\n")
    f.write("id,score\n")

    for (idx, label) in results:
        f.write('TEST_'+str(idx[0])+","+label+"\n")
```

## 5.检查结果


```python
!tail 测试集.csv
```


```python
!head submission.csv
```


```python
!tail submission.csv
```

## 6.提交

利用所学知识，取得第六的成绩，如下图：

![](https://ai-studio-static-online.cdn.bcebos.com/cf5dc3fca8bc46f9af2ad960a5d68fbd5aca7bc93fc3400494b4f85c054520d0)


