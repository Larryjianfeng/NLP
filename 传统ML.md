子曰温故而知新；

pLSA和LDA
---
先提几个问题：
1：LDA里面如何算topics之间的距离，进一步，如何根据topics之间的距离确定最优的topics的数量？
2：为什么EM算法可以用来估计pLSA的参数，但是一般LDA的参数都用gibbs sampling？进一步，这两种方法的本质区别是什么？
- 首先无论EM还是gibbs sampling，都是用来做含隐变量的MLE的参数估计，一般貌似含很多很复杂的隐变量的时候用gibbs sampling; 实际上通过EM的分解可以看出，EM适用于已经知道hidden state的情况下，很容易得到观察到的sample/hidden state的expectation的情况，而且可以通过这个expectation得到最有的参数值，显然对多变量的情况要解偏微分方程很麻烦，因此常用gibbs sampling；
- gibbs sampling可以看作是MC metropolis hasting算法的一种情况；
- MCMC本质上是一种sampling的方法，跟random uniform(0, 1)这种没有本质区别，只是名字比较叼罢了；
- metropolis hasting的思想是通过构造转移矩阵（函数），使得待估计的函数变成Markov chain的平稳分布，而构造转移函数是通过让其满足细致平稳条件得到的；
- 略
3：可以用variational inference EM来估算LDA；


Boosting
---
- Adaboost之前用的比较多，很有意思，adaboosting改变sample权值的时候是提高错误估算的样本的权值，但是DL里面的curriculum learning方法说的是先将简单易分的样本的权值提高；
- Gradient boosting的思想可能是残差神经网络的老祖；

Boosting和Random Forest
---
- Boosting用来降低bias，RF用来降低variance, 这里的bias和variance是假设given x，y的prediction是个分布，因此才会有bias和variance一说；
- 略；

HMM和CRF
--
- HMM和CRF可以对比naive bayes classifier和logistic regression，前者都是generative，就是x和y的联合分布，后者都是discrimitive，就是given x的时候y的分布；
- HMM和linear chain CRF的直观上的区别就是HMM的hidden state转移矩阵是静态不依赖x，但是CRF里面可以是x相关的；
- CRF的参数估计就是尽量使得feature函数对数据的期望接近对模型的期望；
- CRF还没真的用过，以后再补；