The package contains the code for the paper: "Improved Transformer with Multi-head Dense Collaboration". It contains the implementation of our Collformer model. The implementation of Collformer is in the file: src/model.py. Our implementation is based on the framework of tensor2tensor(https://github.com/tensorflow/tensor2tensor). As the prerequisite, please configure the environment of tensor2tensor (version: 1.13).

Descriptions of the source files:
* src/model.py：The implementation of Collformer, and some baseline models, e.g., CollMHA、RealFormer、 MHADR、ConcreteGate、MixedMA、TalkingHA
* datagen.sh: script for the data preprocessing.
* decode.sh: script for decoding.
* get_bleu.sh: script for computing the BLUE score.
* avg_checkpoints.py: it is used to compute the average of the model weights for the checkpoints.
* run.sh：script for training and testing.

For the user, please download the following datasets to the folder: /data
IWSLT 2015 English-Vietnamese (En-Vi) and WMT 2014 English-German (En-De)：  https://nlp.stanford.edu/projects/nmt/ 
WMT 2014 English-French (En-Fr)： https://www.statmt.org/wmt14/translation-task.html

For the data preprocessing, run the following command: sh datagen.sh
Then, the preprocessed data would be in the folder: /t2t_data

For the model training and testing, please run the command: sh run.sh. The settings including dataset, model, parameters can be configured via this file. 
The training log is saved in the folder /log, the training and testing results are saved in the folder /t2t_train, such as:
t2t_train/translate_enfr_wmt32k
t2t_train/translate_ende_wmt32k
t2t_train/translate_envi_iwslt32k



**************************************************

该代码是对本论文"Improved Transformer with Multi-head Dense Collaboration". 这里给出了基于MHDC的collformer的代码实现。

CollFormer结构实现见src/model.py。 我们的代码框架是基于tensor2tensor(https://github.com/tensorflow/tensor2tensor)实现，
首先参考该代码安装说明进行环境配置。

文件说明：
src/model.py： 模型实现，包括文本方法(CollFormer)和基准方法(CollMHA、RealFormer、 MHADR、ConcreteGate、MixedMA、TalkingHA)
datagen.sh: 数据预处里文件
decode.sh：解码脚本
get_bleu.sh: 计算bleu值
avg_checkpoints.py:用于对训练中保存的checkpoints中权重取平均
run.sh：模型训练和测试脚本

首先下载数据集， 并将数据下载,并放置至data文件夹中：
IWSLT 2015 English-Vietnamese (En-Vi)和WMT 2014 English-German (En-De)：  https://nlp.stanford.edu/projects/nmt/
WMT 2014 English-French (En-Fr)： https://www.statmt.org/wmt14/translation-task.html

数据预处理，执行以下脚本： 
sh datagen.sh
处理完数据会放在t2t_data文件夹下


模型训练和测试，执行以下脚本：
sh run.sh
其中数据、方法、参数设置可相应修改。训练log文件会保存在log文件夹下， 模型和测试结果则保存在t2t_train文件夹下，如
t2t_train/translate_enfr_wmt32k
t2t_train/translate_ende_wmt32k
t2t_train/translate_envi_iwslt32k


