# Crosstalk-Generation
![demo](README.asset/gy_wc.png)

Crosstalk Generator for *Natural Language Processing* 2019 Fall @ Fudan University, by  [**Zuobai Zhang**](<https://oxer11.github.io/>)

* [x/](./src)
  * [/code/main.py](./code/main.py)
    * [main.py](./code/main.py)
    * [train.sh参数解释](./ptuning/train.sh) 
  * [/data/pre.py](/data/pre.py)
* [/data/pre.py](./examples)
  * [x.md](./examples/ads_generation.md)
* [README.md](./README.md)

## Tree sructure
```
├── Crosstalk-Generation
│   ├── README.asset
│   │   └── gy_wc.png
│   ├── README.md
│   ├── code
│   │   ├── config.py
│   │   ├── data
│   │   │   └── crosstalk_peng.txt
│   │   ├── evaluate.py
│   │   ├── load.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── train.py
│   ├── data
│   │   ├── analyzer.ipynb
│   │   ├── crosstalk
│   │   │   ├── 郭德纲 李菁《论梦》相声文本.txt
│   │   │   ├── 郭德纲 张文顺《赌论》相声文本.txt
│   │   │   ├── 郭德纲 李菁《卖五器》相声文本.txt
│   │   │   ├── ...
│   │   ├── crosstalk_dirty
│   │   │   ├── 1.txt
│   │   │   ├── 10.txt
│   │   │   ├── 11.txt
│   │   │   ├...
│   │   └── pre.py
│   └── doc
│       ├── PJ.bib
│       ├── figure
│       │   ├── attention.png
│       │   ├── conv1.png
│       │   ├── gdg.png
│       │   ├── gdg1.png
│       │   ├── gdg_wc.png
│       │   ├── gy_wc.png
│       │   ├── loc_wc.png
│       │   ├── luke1.png
│       │   ├── luke2.png
│       │   ├── luke_wc.png
│       │   ├── luke_wc1.png
│       │   ├── name_wc.png
│       │   ├── seg.png
│       │   ├── seq2seq.png
│       │   ├── yq1.png
│       │   ├── yq_wc.png
│       │   └── yq_z.png
│       ├── report.pdf
│       └── report.tex
├── GuoDegang-Generation
│   ├── README.asset
│   │   └── gy_wc.png
│   ├── README.md
│   ├── code
│   │   ├── config.py
│   │   ├── data
│   │   │   └── crosstalk_peng.txt
│   │   ├── evaluate.py
│   │   ├── load.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── train.py
│   ├── data
│   │   ├── analyzer.ipynb
│   │   ├── crosstalk
│   │   │   ├── 郭德纲 李菁《论梦》相声文本.txt
│   │   │   ├── 郭德纲 张文顺《赌论》相声文本.txt
│   │   │   ├── 郭德纲 李菁《卖五器》相声文本.txt
│   │   │   ├── ...
│   │   ├── crosstalk_dirty
│   │   │   ├── 1.txt
│   │   │   ├── 10.txt
│   │   │   ├── 11.txt
│   │   │   ├── 12.txt
│   │   │   ...
│   │   └── pre.py
│   └── doc
│       ├── PJ.bib
│       ├── figure
│       │   ├── attention.png
│       │   ├── conv1.png
│       │   ├── gdg.png
│       │   ├── gdg1.png
│       │   ├── gdg_wc.png
│       │   ├── gy_wc.png
│       │   ├── loc_wc.png
│       │   ├── luke1.png
│       │   ├── luke2.png
│       │   ├── luke_wc.png
│       │   ├── luke_wc1.png
│       │   ├── name_wc.png
│       │   ├── seg.png
│       │   ├── seq2seq.png
│       │   ├── yq1.png
│       │   ├── yq_wc.png
│       │   └── yq_z.png
│       ├── report.pdf
│       └── report.tex
└── directory_structure.md
```
## What's this

*Crosstalk(相声)* is a traditional Chinese comedic performance in the form of a dialogue. The language, rich in puns and allusions, is used in a rapid, bantering style. Crosstalk is one of China's foremost performing arts. Usually, *the main joker(逗哏)* make the audience laugh with words or expressions and *the assistant joker(捧哏)* plays the supporting role of a crosstalk. In this project, we trained a dialogue model to imitate a well-know crosstalk comedian, Qian Yu's speaking style. The code is adapted from [this repo](<https://github.com/ywk991112/pytorch-chatbot>).

## How to use

### Prerequisites

```
Python 3.7
Pytorch 1.0.1
```

### Usage

To train the model, you should first go into the /code repository and then run this command.

```bash
python main.py -tr ./data/crosstalk_peng.txt -la 2 -hi 256 -lr 0.0001 -it 10000 -b 64 -p 500 -s 1000
```

To generate your own crosstalk or chat with Qian Yu, you can type this command.

```bash
python main.py -te ./save/model/crosstalk_peng/2-2_256/10000_backup_bidir_model.tar -c ./data/crosstalk_peng -i
```

## Examples

```bash
# 我和于谦大爷的相声
逗： 今天给大家说段相声。
捧： 怎么样？您给说说啦？怎么样？ 
逗： 这位我是儿子，于谦。
捧： 你是我爸爸怎么说？我知道你。谁说话是   # 叫儿子竟然知道回复爸爸了！
逗： 非常高兴跟于老师同台演出。
捧： 哎呀，行了行了行了。 
逗： 捧您是应该的。
捧： 上家去，代言也不能脸上这个？！
# 捧哏技巧没学着，暴脾气倒是全学来了...
```

```bash
# 给《大碗宽面》捧个哏
逗： 你看这碗又大又圆
捧： 嗨，呵（长套子）啊~吧~吧~
逗： 你看这面又长又宽
捧： 嘿，好嘛   又 多 大 鼓唱 的 干 嘛 呀 ？ 
逗： 有时生活特别累
捧： 这 不 是 什 么 意思 呢 。 你 捧 我 送 这 一 句 
逗： 这碗大！
捧： 好 吗 ， 拿 一 敲 这个 ， 好 吗 ， 又 多 一百 炸
逗： 这大碗宽面也很贵
捧： 阿 ？ 好 嘛 ， 哪 年代 的 事儿 了 ？ ？ ！ 您 这
# 怕不是吃错了药...
```
