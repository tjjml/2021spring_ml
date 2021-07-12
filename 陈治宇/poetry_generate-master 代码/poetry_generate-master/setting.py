

batchSize = 16 ：批次样本数目
epochNum =30 ：训练轮次
hidden_units=128：一层LSTM网络模型的神经元数目，也是词向量的维度
layers=2：LSTM网络模型的层数num_steps= maxLength：LSTM 展开的步（step）数，相当于每个批次输入一个样本中数据的数目，这里为每batchSize首诗中最长的诗的字数。
learningRateBase = 0.001：初始学习率
learningRateDecayStep = 1000 每隔 1000 步学习率下降
learningRateDecayRate= 0.95 ：在过了每个learningRateDecayStep步之后，学习率的衰减率
init_scale=0.05：权重参数的初始取值跨度
dropout=0.2：在Dropout层的留存率，这里只对隐藏层进行Dropout处理
