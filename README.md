Attention Is All You Need
---
- 问题：之前facbook的研究人员用自家的cnn做翻译模型，做出来比G家的rnn又好又快，G家的研究人员于是弄了个又不是CNN又不是rnn的attention模型，就是这个，于是又做到了比facebook的模型更快更好；
- 核心：multihead_attention结构，这个是dot attention的一个扩展，使用八个attention并行运算
	- 比如要做attention的两个矩阵分别是query和keys；
	- 将query和keys映射到8*num_heads, 然后再分割成八份，分别做attention计算；
	- query乘以keys矩阵，再除以keys的维度的方根，再将keys填充部分的都变成0，取softmax作为attention
	- 之后attention再对query填充的部分变成0，再矩阵乘以keys作为输出
	- 注意到用六个multihead_attention结构叠加起来；
- multihead_attention后面用一个feedforward网络输出一个type和shape都相同的矩阵
- encoding的时候注意到用了位置矩阵，就是word embedding加一个position matrix合起来训练
- encoder和decoder分别对自己用了attention之后，记作encoder'和decoder',再将其输入multihead_attention网络得到最终输出, 输出的向量映射到一个宽度为vocabulary size的向量，取值最大的位置作为目标词汇；
- 评论：等我实验了再来评论

Rethinking the inception architecture for computer vision
---
- 问题：GoogLeNet在做图片分类的时候取得了很好的效果，其中inception结构的作用很大，作者介绍了inception的直觉和一些经验之谈；做NLP的时候很多场景也需要用到CNN,遂将这篇论文的理解记录于此；
- General Design Principles
	- 1: 避免representatinal bottlenecks；我的理解是，NN从输入到输出可以看作一个信息流动，每层做convolution或者其他变换都会将sample映射成新的presentation，比如一个句子先被你变成32*256的embeding表示，做了RNN之后又会变成32*rnn_output_dimesion的表示，作者的意思是每次变幻的时候都不要太激进，比如你不能一下子把一个256*256的表示变成2*2的表示，这样会造成大量的信息损失；
	- 2：高维度的representation在network里面更容易process；这个我理解为数据初步做convolution的时候怎么样搞都可以，经过几次conv和maxpooling之后就需要小心翼翼的做process了（我理解的可能不靠谱）；
	- 3：spatial aggregation在低维度的representation的时候可以几乎不损失信息；这点说的就是做3*3convolution或者其他空间信息压缩的时候，先降低representation的维度；
	- 4： 长度和宽度保持平衡；
- Factorizing Convolutions with Large Filters
	- 1: 用更多的size更小的convolution，比如将5*5的convolution替换成3*3 on top of 3*3；
	- 2：用对称的convolution结构，比如将n*n的 convolution替换成1*n on top of n*1；

NN model stacks
---
- 这个不是论文，是我自己工作里面实验的结果记录
- 对于多分类问题（1000类），能不能用一个模型进行粗分类缩小范围，再用另一个模型进行精分类？
- 对于问题匹配模型，我先用autoencoder模型去top 10的句子，再用RNN进行排序，得到了大概7%的提升，不知道有木有什么数学依据；
- 下面是比较精髓的一段代码：

			# 因为tf.gather得缺陷，最好的解决方案是reshape到一维向量再index
			_, top5idx = tf.nn.top_k(pred, tpk)
			bs = tf.expand_dims(self.batch_size, -1)
			top5idx = bs*self.num_anses + top5idx
			# simic是rnn得到的对每个类的结果
			# 主要是用reshape+index的方法解决tf.gather的缺陷；
			self.simicr = tf.reshape(simic, [-1])
			top5idx = tf.reshape(top5idx, [-1])

			simi2 = tf.gather(self.simicr, top5idx)
			simi2 = tf.reshape(simi2, [-1, tpk])
			simi2 = tf.concat(1, [simi2, simitrue])
			simi2 = tf.nn.softmax(simi2, -1)
			_ = tf.expand_dims([1.0]*tpk + [-1.0], -1)
- 如果有更好的方法，请指正；


Boosting and neural network(非论文，我自己的问题)
---
- 问题就是：我现在做机器人对话，有些反馈数据，如何利用这部分feedback数据，我联想到了boosting，但是基本上没有关于boosting在NN上的应用，为什么？
- 猜想：
	- 1：boosting里面都是弱的和不稳定的分类器；由弱的分类器构成了一个比较复杂的分类边界（decision boundaries），但是NN本身就几乎可以模拟任何复杂的函数，就像AMD八核胶水敌不过intel双核的区别一样；
	- 2：想一想神经网络和传统机器学习的训练方式不同，神经网络是一个batch一个batch训练，将数据过很多遍，而且使用dropout避免过拟合，iteration很多次；然而传统的比如gradient boosting是通过多次迭代，每次迭代最优化一个参数来使得最终的loss最小，从这个角度看，NN就是gradient boosting的"无限"迭代状态,其实不是无限，因为NN的参数还是有限的；
	- 3：本人尚未完全找到一个有效利用feedback的方法，如果有想法，请告知 jy2641@columbia.edu;

Curriculum Learning
---
- 解决的问题：这是bengio大神的文章，因为他发现有时候有些深度网络要取的比其他深度网络更好的结果，而且务监督的预训练有时候让模型在测试集上表现更好，他提出了这中curriculum learning的思想就是想通过一步步的训练让模型收敛更快，表现更好
- Curriculum详解：我刚开始以是“简历”的意思，然后发现“变形”更好。
- 这篇文章的结构：
	- 1，2: introduction之类的
	- 3.1: 提出有一种continuation的方法可以解决这个问题
	- 3.2: continuation的方法实际上在他用来就是reweight，就是先训练简单的数据，再训练复杂的数据，这个很神奇，比如说二分类，先主要让模型把注意力集中在那些离得比较远的点上，那些显然可以分开的点，再看那些分不太开的点；这样先训练出一个分类器，再reweight，逐渐提高那些模糊点的weight，最后达到样本开始时的分布；
	- 4: 刚提到，curriculum learning本质上是一种reweight，这部分作者提出了一些reweight的方法，然而作者也说了，具体问题，还是要具体分析。（我一直觉得深度学习是一种伪科学，黑猫白猫抓到老鼠就是好猫，这又印证了我的观点）
	- 5，6：作者分别用一个简单的图像识别和语言模型来说明怎么样reweigth，重点说说语言模型，这里的问题是给定前四个词预测下一个单词，然后训练数据用的是wikipedia的句子，作者首先用最常出现的五千单词做，也就是说训练数据的五个连续单词必须出现在这五千个单词里面，训练一段时间之后再用10000单词做，这样可以视为一种抽象的reweight；
- 评价：这篇文章结构清晰，由易入难，阐述了作者的思想，即sample reweight. 但是跟其他深度学习文章一样，作者喜欢搞一些听起来高达上的名字，reweight就reweight，搞个什么curriculum是几个意思，搞的神神秘秘的；

QA_LSTM_CNN_ATTENTION_IBM
---
- 解决的问题: answer selection. 即给定问题，计算多个答案的匹配率，选取最好的那个，适用于单轮对话. <br>
- 根据这篇文章我选用了bidirectional-lstm+hidden state maxpooling的方法，改进loss函数之后，比较有效的解决了问题到答案的映射问题。 这篇文章提出了多个模型，实际上都差不太多，作者是个比较有好的人，我给他发邮件询问模型的正负样本配比还回我了。
- 先来看看最基本的模型结构:
	- ![Alt text](/imgs/bilstm.png)
	- 就是用两个biLSTM分别应用于问题和答案；这里的答案如果是假答案，则label 0，反之则label 1；
	- biLSTM分别run过一个句子之后，会得到n个中间hidden state和一个最终的hidden state；
	- 可以直接用最终的hidden state做cosine距离计算，从而比较真假答案和原问题之间的距离；
	- 也可以利用所有的hidden state，上图提到了做mean，或者max pooling；
- biLSTM + cnn
	- ![Alt text](/imgs/bilstm_cnn.png)
	- 略
- biLSTM + cnn + attention
	- 注意到这里的attention是将问题向量应用到答案向量里面去；
	- 应用的方式是，先将问题的hidden state做mean，然后根据这个向量分别和所有的答案hidden state做计算出weigth，然后答案hidden state乘以做个weight作为attention based答案hidden state
	- 然后使用cnn分别对问题hidden states，和attention based答案hidden states做抽象；
- 评价：一篇简洁易懂又实用的论文，值得一看的；

<br><br>
Sequential Matching Network by MS:
---
- 原标题：Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots<br>
- 解决的问题: 多轮对话里面的答案选取，本质上是一个匹配模型，而不是生成式模型。<br>
模型结构：<br>
![Alt text](/imgs/sequential_matching.png)
- 结构讲解:<br>
	- First Layer: 说的是考虑了上下文，其实就是把历史对话逐个和带选取的答案进行比较，但是比较的方式比较特别。以前我们用的lstm都是直接拿问题和答案的最终hidden state进行consine距离计算或者全连接得到一个相似度，denoising autoencoder的话是直接问题答案的hidden state进行cosine相似度；但是在这里，它是先拿一个句子里面的所有词向量和另一个句子里面的所有词向量进行相似度计算之后得到一个相似度矩阵，即M1，然后拿GRU去run这两个句子，得到两个hidden states序列，再进行全部的相似度计算，得到M2，之后做卷积，即Covolution和pooling部分。<br>
	- Matching Accumulation: 卷积过后，每个历史问题和待选答案之间得到一个向量，这样得到一个序列的向量，再用第二个GRU去run这个向量，中间的hidden states用attention机制来做加权平均得到一个最终向量，（或者直接取最终向量拉倒）再经过全连阶层得到一个最终相似度。<br>
- 评价：<br>
	- 作者没分词，直接用单个的汉字来做的，第一层里面用词之间的相似度矩阵做convolution的方式比较特别，值得拿来研究。<br>
<br><br>

EFFICIENT VECTOR REPRESENTATION FOR DOCUMENTS THROUGH CORRUPTION
---
- 简介：就是用来把一句话转换成一个向量，但是不同于word2vec里面的方法，这个用了个dropout的方法训练document程度的vector
- 模型结构图：
![Alt text](/imgs/doc2vecc.png)
	- 不同于word2vec的地方：
	<img src="/imgs/doc2vecc_2.png" width="80%" height="80%">
	这里上面的那个式子我理解为每个doc vector是不断变化的(corruption)，但是初始为所有词向量的平均值；
	- 其他地方都很简单；
	- 具体效果有待实验，不过这个方法可以同时训练词向量；



<br><br>
Introduction to different gradient descend methods:
---
- 原文档连接：http://sebastianruder.com/optimizing-gradient-descent/index.html#stochasticgradientdescent <br>
- 目的：理解几个常用的gradient descend的机制和优劣
- Stochastic Gradient Descend:
![Alt text](/imgs/SGD.PNG)
	- SGD 是相当于batch gradient descend来说的，SGD对每一个训练样本都更新其参数，这样频繁的更新会使得目标函数波动幅度很大，同时收敛迅速；
	![Alt text](/imgs/sgd_fluctuation.png)
	- 当我们慢慢减少SGD的learning rate的时候，几乎可以保证损失函数能收敛到局部最小值(non-convex函数)和全局最小值(convex函数)
	- 优点就是收敛迅速，缺点个人认为是要不断调整learning rate.

- Adadelta
![Alt text](/imgs/adadelta.png)
![Alt text](/imgs/adadelta_decay.png)
	- Adadelta可以看作Adagrad变种而来，它们都假设gradients的方差遵循一个自相关的过程；作为一个自适应的gradient的方法，adadelta不需要设定learning rate；
	- 优点是训练初中期，加速效果不错，很快， 缺点是训练后期，反复在局部最小值附近抖动


- Adam
![Alt text](/imgs/adam.png)
	- adam结合gradient的一阶和二阶矩估计来调整learning rate；上图中mt和vt分别是梯度的一阶和二阶矩估计；
	- 适用于大多非凸优化 - 适用于大数据集和高维空间
