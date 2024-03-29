# 第二章 模型评估
- 只有选择与问题相匹配的评估方法，才能快速地发现模型选择或训练过程中出现的问题，迭代地对模型进行优化。
- 模型评估主要分为离线评估和在线评估两个阶段。
- 针对分类、排序、回归、序列预测等不同类型的机器学习问题，评估指标的选择也有所不同。
- 知道每种评估指标的精确定义、有针对性地选择合适的评估指标、根据评估指标的反馈进行模型调整，
这些都是机器学习在模型评估阶段的关键问题，也是一名合格的算法工程师应当具备的基本功。
----
## 01 评估指标的局限性
在模型评估过程中，分类问题、排序问题、回归问题往往需要使用不同的指标进行评估。
在诸多的评估指标中，大部分指标只能片面地反映模型的一部分性能。
如果不能合理地运用评估指标，不仅不能发现模型本身的问题，而且会得出错误的结论。

真正例TP = True Positive，预测值是1，真实值是1，被正确分类的正例样本<br>
假正例FP = False Positive，预测值是1，但真实值是0<br>
真反例TN = True Negative，预测值是0，真实值是0<br>
假反例FN = False Negative，预测值是0，但真实值是1<br>
准确率（Accuracy）<br>
精确率（Precision）<br>
召回率（Recall）<br>
均方根误差（MSE）
![](https://github.com/pchen12567/picture_store/blob/master/AI_For_NLP/validation.png?raw=true)

### 准确率的局限性?
- 准确率：指分类正确的样本占总样本个数的比例
$$ Accuracy = \frac{n_{correct}}{n_{total}} $$
- 准确率是分类问题中最简单也是最直观的评价指标，但存在明显的缺陷。
比如，当负样本占99%时，分类器把所有样本都预测为负样本也可以获得99%的准确率。
所以，当不同类别的样本比例非常不均衡时，占比大的类别往往成为影响准确率的最主要因素。
- 为了解决这个问题，可以使用更为有效的平均准确率（每个类别下的样本准确率的算术平均）作为模型评估的指标。
- 标准答案其实也不限于指标的选择，即使评估指标选择对了，
仍会存在模型过拟合或欠拟合、测试集和训练集划分不合理、线下评估与线上测试的样本分布存在差异等一系列问题，
但评估指标的选择是最容易被发现，也是最可能影响评估结果的因素。

### 精确率与召回率的权衡?
- 精确率：分类正确的正样本个数占分类器判定为正样本的样本个数的比例。
- 召回率：分类正确的正样本个数占真正的正样本个数的比例。
- Precision值和Recall值是既矛盾又统一的两个指标，为了提高Precision值，分类器需要尽量在“更有把握”时才把样本预测为正样本，
但此时往往会因为过于保守而漏掉很多“没有把握”的正样本，导致Recall值降低。
- 通常使用F1值，即精确率和召回率的调和平均值，来综合反映一个模型的性能。

**P-R曲线**<br>
P-R曲线的横轴是召回率，纵轴是精确率。
对于一个排序模型来说，其P-R曲线上的一个点代表着，在某一阈值下，模型将大于该阈值的结果判定为正样本，
小于该阈值的结果判定为负样本,此时返回结果对应的召回率和精确率。整条P-R曲线是通过将阈值从高到低移动而生成的。

如图所示P-R曲线样例图，其中实线代表模型A的P-R曲线，虚线代表模型B的P-R曲线。原点附近代表当阈值最大时模型的精确率和召回率。<br>
![](https://github.com/pchen12567/picture_store/blob/master/Interview/estimation_06.png?raw=true)

由图可见,当召回率接近于0时,模型A的精确率为0.9，模型B的精确率是1，这说明模型B得分前几位的样本全部是真正的正样本，
而模型A即使得分最高的几个样本也存在预测错误的情况。并且随着召回率的增加,精确率整体呈下降趋势。
但是,当召回率为1时模型A的精确率反而超过了模型B。这充分说明,只用某个点对应的精确率和召回率是不能全面地衡量模型的性能，
只有通过P-R曲线的整体表现,才能够对模型进行更为全面的评估。

## 02 ROC曲线
二值分类器（Binary Classifier）是机器学习领域中最常见也是应用最广泛的分类器。
评价二值分类器的指标很多，比如precision、recall、F1 score、P-R曲线等。
但也发现这些指标或多或少只能反映模型在某一方面的性能。
相比而言，ROC曲线则有很多优点，经常作为评估二值分类器最重要的指标之一。

### 什么是ROC曲线？
- ROC曲线是Receiver Operating Characteristic Curve的简称，中文名为“受试者工作特征曲线”。
- ROC曲线的横坐标为假阳性率（False Positive Rate，FPR）；纵坐标为真阳性率（True Positive Rate，TPR）。
$$ FPR = \frac{FP}{N} $$
$$ TPR = \frac{TP}{P} $$
P是真实的正样本的数量，N是真实的负样本的数量，TP是P个正样本中被分类器预测为正样本的个数，FP是N个负样本中被分类器预测为正样本的个数。

### 如何绘制ROC曲线?
事实上，ROC曲线是通过不断移动分类器的“截断点”来生成曲线上的一组关键点的。

在二值分类问题中，模型的输出一般都是预测样本为正例的概率。
假设测试集中有20个样本，样本按照预测概率从高到低排序。
在输出最终的正例、负例之前，我们需要指定一个阈值，预测概率大于该阈值的样本会被判为正例，小于该阈值的样本则会被判为负例。
比如，指定阈值为0.9，那么只有第一个样本会被预测为正例，其他全部都是负例。上面所说的“截断点”指的就是区分正负预测结果的阈值。

通过动态地调整截断点，从最高的得分开始（实际上是从正无穷开始，对应着ROC曲线的零点），逐渐调整到最低得分，
每一个截断点都会对应一个FPR和TPR，在ROC图上绘制出每个截断点对应的位置，再连接所有点就得到最终的ROC曲线。

就本例来说，当截断点选择为正无穷时，模型把全部样本预测为负例，那么FP和TP必然都为0，FPR和TPR也都为0，
因此曲线的第一个点的坐标就是（0,0）。当把截断点调整为0.9时，模型预测1号样本为正样本，并且该样本确实是正样本，因此，TP=1，
20个样本中，所有正例数量为P=10，故TPR=TP/P=1/10；这里没有预测错的正样本，即FP=0，负样本总数N=10，故FPR=FP/N=0/10=0，
对应ROC曲线上的点（0,0.1）。依次调整截断点，直到画出全部的关键点，再连接关键点即得到最终的ROC曲线。

还有一种更直观地绘制ROC曲线的方法。
首先，根据样本标签统计出正负样本的数量，假设正样本数量为P，负样本数量为N；
接下来，把横轴的刻度间隔设置为1/N，纵轴的刻度间隔设置为1/P；
再根据模型输出的预测概率对样本进行排序（从高到低）；
依次遍历样本，同时从零点开始绘制ROC曲线，每遇到一个正样本就沿纵轴方向绘制一个刻度间隔的曲线，
每遇到一个负样本就沿横轴方向绘制一个刻度间隔的曲线，直到遍历完所有样本，曲线最终停在（1,1）这个点，整个ROC曲线绘制完成。<br>
![](https://github.com/pchen12567/picture_store/blob/master/Interview/estimation_01.jpg?raw=true)

### 如何计算AUC？
AUC指的是ROC曲线下的面积大小，该值能够量化地反映基于ROC曲线衡量出的模型性能。
计算AUC值只需要沿着ROC横轴做积分就可以了。
由于ROC曲线一般都处于y=x这条直线的上方（如果不是的话，只要把模型预测的概率反转成1−p就可以得到一个更好的分类器），
所以AUC的取值一般在0.5～1之间。AUC越大，说明分类器越可能把真正的正样本排在前面，分类性能越好。

### ROC曲线相比P-R曲线有什么特点？
相比P-R曲线，ROC曲线有一个特点，当正负样本的分布发生变化时，ROC曲线的形状能够基本保持不变，而P-R曲线的形状一般会发生较剧烈的变化。<br>
![](https://github.com/pchen12567/picture_store/blob/master/Interview/estimation_02.jpg?raw=true)

可以看出，P-R曲线发生了明显的变化，而ROC曲线形状基本不变。
这个特点让ROC曲线能够尽量降低不同测试集带来的干扰，更加客观地衡量模型本身的性能。
在很多实际问题中，正负样本数量往往很不均衡。比如，计算广告领域经常涉及转化率模型，正样本的数量往往是负样本数量的1/1000甚至1/10000。
若选择不同的测试集，P-R曲线的变化就会非常大，而ROC曲线则能够更加稳定地反映模型本身的好坏。
所以，ROC曲线的适用场景更多，被广泛用于排序、推荐、广告等领域。但需要注意的是，选择P-R曲线还是ROC曲线是因实际问题而异的。

## 03 余弦距离的应用
> [余弦距离参考](https://github.com/pchen12567/AI_For_NLP/blob/master/Week_06_TFIDF/LectureCode_06.ipynb)
- 在机器学习问题中，通常将特征表示为向量的形式，所以在分析两个特征向量之间的相似性时，常使用余弦相似度来表示。
- 余弦相似度的取值范围是[−1,1]，相同的两个向量之间的相似度为1。
- 如果希望得到类似于距离的表示，将1减去余弦相似度即为余弦距离。$ dist(A, B) = 1 - cos\theta $
- 因此，余弦距离的取值范围为[0,2]，相同的两个向量余弦距离为0。
- 给定两个特征向量，以下方法可以计算这两个向量相似度
    - 曼哈顿距离
    - 欧式距离
    - 余弦夹角（余弦相似度）

### 结合学习和研究经历，探讨为什么在一些场景中要使用余弦相似度而不是欧式距离？
- 对于两个向量A和B，其余弦相似度定义为：
![](https://github.com/pchen12567/picture_store/blob/master/Interview/estimation_04.png?raw=true)<br>
分子为向量A与向量B的点乘，分母为二者各自的L2相乘，即将所有维度值的平方相加后开方。 
余弦相似度的取值为[-1,1]，值越大表示越相似。
- 两个向量夹角的余弦，关注的是向量之间的角度关系，并不关心它们的绝对大小，其取值范围是[−1,1]。
当一对文本相似度的长度差距很大、但内容相近时，如果使用词频或词向量作为特征，它们在特征空间中的的欧氏距离通常很大；
而如果使用余弦相似度的话，它们之间的夹角可能很小，因而相似度高。
- 此外，在文本、图像、视频等领域，研究的对象的特征维度往往很高，余弦相似度在高维情况下依然保持“相同时为1，正交时为0，
相反时为−1”的性质，而欧氏距离的数值则受维度的影响，范围不固定，并且含义也比较模糊。
- 总体来说，欧氏距离体现数值上的绝对差异，而余弦距离体现方向上的相对差异。
- 例如，统计两部剧的用户观看行为，用户A的观看向量为(0,1)，用户B为(1,0)；此时二者的余弦距离很大，而欧氏距离很小；
分析两个用户对于不同视频的偏好，更关注相对差异，显然应当使用余弦距离。
- 而当我们分析用户活跃度，以登陆次数(单位：次)和平均观看时长(单位：分钟)作为特征时，余弦距离会认为(1,10)、(10,100)两个用户距离很近；
但显然这两个用户活跃度是有着极大差异的，此时我们更关注数值绝对差异，应当使用欧氏距离。
![](https://github.com/pchen12567/picture_store/blob/master/AI_For_NLP/cosine_similarity.jpg?raw=true)

### 余弦距离是否是一个严格定义的距离？
- 距离的定义：在一个集合中，如果每一对元素均可唯一确定一个实数，使得三条距离公理（正定性，对称性，三角不等式）成立，
则该实数可称为这对元素之间的距离。
- 余弦距离满足正定性和对称性，但是不满足三角不等式，因此它并不是严格定义的距离。
- 在机器学习领域，被俗称为距离，却不满足三条距离公理的不仅仅有余弦距离，还有KL距离（Kullback-Leibler Divergence），
也叫作相对熵，它常用于计算两个分布之间的差异，但不满足对称性和三角不等式。

## 04 A/B测试的陷阱
在互联网公司中，A/B 测试是验证新模块、新功能、新产品是否有效，新算法、新模型的效果是否有提升，新设计是否受到用户欢迎，
新更改是否影响用户体验的主要测试方法。在机器学习领域中，A/B 测试是验证模型最终效果的主要手段。

### 在对模型进行过充分的离线评估之后，为什么还要进行在线A/B测试？
1. 离线评估无法完全消除模型过拟合的影响，因此，得出的离线评估结果无法完全替代线上评估结果。
2. 离线评估无法完全还原线上的工程环境。
一般来讲，离线评估往往不会考虑线上环境的延迟、数据丢失、标签数据缺失等情况。因此，离线评估的结果是理想工程环境下的结果。
3. 线上系统的某些商业指标在离线评估中无法计算。
离线评估一般是针对模型本身进行评估，而与模型相关的其他指标，特别是商业指标，往往无法直接获得。
比如，上线了新的推荐算法，离线评估往往关注的是ROC曲线、P-R曲线等的改进，
而线上评估可以全面了解该推荐算法带来的用户点击率、留存时长、PV访问量等的变化。这些都要由A/B测试来进行全面的评估。

### 如何进行线上A/B测试？
进行A/B测试的主要手段是进行用户分桶，即将用户分成实验组和对照组，对实验组的用户施以新模型，对对照组的用户施以旧模型。
在分桶的过程中，要注意样本的独立性和采样方式的无偏性，确保同一个用户每次只能分到同一个桶中，
在分桶过程中所选取的user_id需要是一个随机数，这样才能保证桶中的样本是无偏的。

### 如何划分实验组和对照组？
H公司的算法工程师们最近针对系统中的“美国用户”研发了一套全新的视频推荐模型A，而目前正在使用的针对全体用户的推荐模型是B。
在正式上线之前，工程师们希望通过A/B测试来验证新推荐模型的效果。下面有三种实验组和对照组的划分方法，请指出哪种划分方法是正确的？
（1）根据user_id（user_id完全随机生成）个位数的奇偶性将用户划分为实验组和对照组，对实验组施以推荐模型A，对照组施以推荐模型B；
（2）将user_id个位数为奇数且为美国用户的作为实验组，其余用户为对照组；
（3）将user_id个位数为奇数且为美国用户的作为实验组，user_id个位数为偶数的用户作为对照组。

上述3种A/B测试的划分方法都不正确。
正确的做法是将所有美国用户根据user_id个位数划分为试验组合对照组，分别施以模型A和B，才能够验证模型A的效果。<br>
![](https://github.com/pchen12567/picture_store/blob/master/Interview/estimation_03.png?raw=true)

## 05 模型评估的方法
### 在模型评估过程中，有哪些主要的验证方法，它们的优缺点是什么？
1. Holdout检验 <br>
Holdout检验是最简单也是最直接的验证方法，它将原始的样本集合随机划分成训练集和验证集两部分。
比方说，对于一个点击率预测模型，我们把样本按照 70%～30% 的比例分成两部分，70% 的样本用于模型训练；30% 的样本用于模型验证，
包括绘制ROC曲线、计算精确率和召回率等指标来评估模型性能。
Holdout 检验的缺点很明显，即在验证集上计算出来的最后评估指标与原始分组有很大关系。为了消除随机性，研究者们引入了“交叉检验”的思想。

2. 交叉验证 <br>
k-fold交叉验证：首先将全部样本划分成k个大小相等的样本子集；依次遍历这k个子集，每次把当前子集作为验证集，其余所有子集作为训练集，
进行模型的训练和评估；最后把k次评估指标的平均值作为最终的评估指标。在实际实验中，k经常取10。<br>
留一验证：每次留下1个样本作为验证集，其余所有样本作为测试集。样本总数为n，依次对n个样本进行遍历，进行n次验证，
再将评估指标求平均值得到最终的评估指标。在样本总数较多的情况下，留一验证法的时间开销极大。事实上，留一验证是留p验证的特例。
留p验证是每次留下p个样本作为验证集，它的时间开销更是远远高于留一验证，故而很少在实际工程中被应用。

3. 自助法 <br>
不管是Holdout检验还是交叉检验，都是基于划分训练集和测试集的方法进行模型评估的。
然而，当样本规模比较小时，将样本集进行划分会让训练集进一步减小，这可能会影响模型训练效果。
自助法可以比较好地解决这个问题。<br>
自助法是基于自助采样法的检验方法。对于总数为n的样本集合，进行n次有放回的随机抽样，得到大小为n的训练集。
n次采样过程中，有的样本会被重复采样，有的样本没有被抽出过，将这些没有被抽出的样本作为验证集，进行模型验证，这就是自助法的验证过程。

### 在自助法的采样过程中，对N个样本进行N此自助抽样，当N趋于无穷大时，最终有多少数据从未被选择过？
(To be continue...)

## 06 超参数调优
### 超参数有哪些调优方法？
为了进行超参数调优，我们一般会采用网格搜索、随机搜索、贝叶斯优化等算法。<br>
在具体介绍算法之前，需要明确超参数搜索算法一般包括哪几个要素。一是目标函数，即算法需要最大化/最小化的目标；
二是搜索范围，一般通过上限和下限来确定；三是算法的其他参数，如搜索步长。

- 网格搜索 <br>
网格搜索可能是最简单、应用最广泛的超参数搜索算法，它通过查找搜索范围内的所有的点来确定最优值。
如果采用较大的搜索范围以及较小的步长，网格搜索有很大概率找到全局最优值。
然而，这种搜索方案十分消耗计算资源和时间，特别是需要调优的超参数比较多的时候。
因此，在实际应用中，网格搜索法一般会先使用较广的搜索范围和较大的步长，来寻找全局最优值可能的位置；
然后会逐渐缩小搜索范围和步长，来寻找更精确的最优值。
这种操作方案可以降低所需的时间和计算量，但由于目标函数一般是非凸的，所以很可能会错过全局最优值。

- 随机搜索 <br>
随机搜索的思想与网格搜索比较相似，只是不再测试上界和下界之间的所有值，而是在搜索范围中随机选取样本点。
它的理论依据是，如果样本点集足够大，那么通过随机采样也能大概率地找到全局最优值，或其近似值。
随机搜索一般会比网格搜索要快一些，但是和网格搜索的快速版一样，它的结果也是没法保证的。

- 贝叶斯优化算法 <br>
贝叶斯优化算法在寻找最优最值参数时，采用了与网格搜索、随机搜索完全不同的方法。
网格搜索和随机搜索在测试一个新点时，会忽略前一个点的信息；而贝叶斯优化算法则充分利用了之前的信息。
贝叶斯优化算法通过对目标函数形状进行学习，找到使目标函数向全局最优值提升的参数。
具体来说，它学习目标函数形状的方法是，首先根据先验分布，假设一个搜集函数；
然后，每一次使用新的采样点来测试目标函数时，利用这个信息来更新目标函数的先验分布；
最后，算法测试由后验分布给出的全局最值最可能出现的位置的点。
对于贝叶斯优化算法，有一个需要注意的地方，一旦找到了一个局部最优值，它会在该区域不断采样，所以很容易陷入局部最优值。
为了弥补这个缺陷，贝叶斯优化算法会在探索和利用之间找到一个平衡点，“探索”就是在还未取样的区域获取采样点；
而“利用”则是根据后验分布在最可能出现全局最值的区域进行采样。

## 07 过拟合与欠拟合
### 在模型评估过程中，过拟合和欠拟合具体是指什么现象？
- 过拟合是指模型对于训练数据拟合过当的情况，反映到评估指标上，就是模型在训练集上的表现很好，但在测试集和新数据上的表现较差。
- 欠拟合指的是模型在训练和预测时表现都不好的情况。

### 能否说出几种降低过拟合和欠拟合风险的方法？
> [过拟合和欠拟合参考](https://github.com/pchen12567/AI_For_NLP/blob/master/Week_07_MachineLearning/Assignment_07.md)

- 过拟合原因
    - 数据存在噪声；
    - 训练数据不足，有限的训练数据；
    - 训练模型过度，模型复杂度太高。
    
- 降低过拟合风险的方法
    - 从数据入手，获得更多的训练数据。使用更多的训练数据是解决过拟合问题最有效的手段，因为更多的样本能够让模型学习到更多更有效的特征，
    减小噪声的影响。当然，直接增加实验数据一般是很困难的，但是可以通过一定的规则来扩充训练数据。
    比如，在图像分类的问题上，可以通过图像的平移、旋转、缩放等方式扩充数据；
    更进一步地，可以使用生成式对抗网络来合成大量的新训练数据。
    - 降低模型复杂度，特征选择，减少特征数或使用较少的特征组合，对于按区间离散化的特征，增大划分的区间。
    在数据较少时，模型过于复杂是产生过拟合的主要因素，适当降低模型复杂度可以避免模型拟合过多的采样噪声。
    例如，在神经网络模型中减少网络层数、神经元个数等；在决策树模型中降低树的深度、进行剪枝等。
    - 正则化方法，常用的有 L1、L2 正则，而且 L1正则还可以自动进行特征选择。
    给模型的参数加上一定的正则约束，比如将权值的大小加入到损失函数中。<br>
    以L2正则化为例：$ C = C_0 + \frac{\lambda}{2n} \cdot \sum_i w_i^2 $
    - 如果有正则项则可以考虑增大正则项参数；
    - 集成学习方法。集成学习是把多个模型集成在一起，来降低单一模型的过拟合风险，如Bagging方法。
    
- 欠拟合原因
    - 模型复杂度过低
    - 训练误差大
    
- 降低欠拟合风险的方法
    - 添加新特征。当特征不足或者现有特征与样本标签的相关性不强时，模型容易出现欠拟合。
    通过挖掘“上下文特征”“ID类特征”“组合特征”等新的特征，往往能够取得更好的效果。
    在深度学习潮流中，有很多模型可以帮助完成特征工程，如因子分解机、梯度提升决策树、Deep-crossing等都可以成为丰富特征的方法。
    - 增加模型复杂度。简单模型的学习能力较差，通过增加模型的复杂度可以使模型拥有更强的拟合能力。
    例如，在线性模型中添加高次项，在神经网络模型中增加网络层数或神经元个数等。
    - 减小正则化系数。正则化是用来防止过拟合的，但当模型出现欠拟合现象时，则需要有针对性地减小正则化系数。

## 08 生成模型和判别模型
### 生成模型和判别模型的区别是什么？
1. 结论
    - 公式上看
        - 生成模型： 学习时先得到$ P(x,y) $，继而得到$ P(y|x) $。预测时应用最大后验概率法（MAP）得到预测类别y。 
        - 判别模型： 直接学习得到$ P(y|x) $，利用MAP得到 y。或者直接学得一个映射函数$ y=f(x) $。
    - 直观上看
        - 生成模型： 关注数据是**如何生成**的。
        - 判别模型： 关注类别之间的**差别**。

2. 先直观理解
    - 生成模型：<br>
    **源头导向**。尝试去找到底这个数据是怎么产生的，然后再对一个信号进行分类。
    基于你学习到的生成假设，判断哪个类别最有可能产生这个信号，这个信号就属于那个类别。
    - 判别模型：<br> 
    **差别导向**。并不关心数据是怎么生成的，它只关心信号之间的差别，然后用差别来简单对给定的一个信号进行分类。

3. 举个例子<br>
    假如你的任务是识别一个语音属于哪种语言。例如对面一个人走过来，和你说了一句话，你需要识别出她说的到底是汉语、英语还是法语等。
    那么你可以有两种方法达到这个目的：
    
    学习每一种语言，你花了大量精力把汉语、英语和法语等都学会了，我指的学会是你知道什么样的语音对应什么样的语言。
    然后再有人过来对你说，你就可以知道他说的是什么语音。
    
    不去学习每一种语言，你只学习这些语言之间的差别，然后再判断（分类）。意思是指我学会了汉语和英语等语言的发音是有差别的，
    我学会这种差别就好了。
    
    那么第一种方法就是生成方法，第二种方法是判别方法。

4. 深入理解<br>
    监督学习的任务：学习一个模型，应用这一模型，对给定的输入预测相应的输出。
    这一模型的一般形式为一个决策函数或者条件概率分布：
    
    - 条件概率分布
    $$ P(y|x) $$
    预测时用最大后验概率(MAP) $ y = argmax_{y_i} P(y_i|x) $的方法决定输出类别 y。（例如贝叶斯分类器就属于这种）
    - 决策函数
    $$ y=f(x) $$
    直接得到输入 x 到 输出 y（某个类别）的映射函数。（例如神经网络和SVM等属于这种）
    
    因此，监督学习方法又可以分为生成方法(generative approach)和判别方法(discriminative approach)。
    所学到的模型分别为生成模型(generative model)和判别模型(discriminative model)。
    
5. 生成模型<br>
    生成式模型（Generative Model）会对x和y的联合分布p(x,y)建模，然后通过贝叶斯公式来求得p(yi|x)，然后选取使得p(yi|x)最大的yi
    
    先由数据学习联合概率分布$P(x,y)$和先验概率分布$P(x)$，然后求出条件概率分布$P(y|x)=P(x,y)/P(x)$作为预测的模型，
    即得到生成模型:
    $$ P(y|x) = \frac{P(x,y}{P(x)} $$
    
    **生成方法强调的是：通过得到 $P(x,y)$，继而得到 $P(y|x)$。**
    
    这样的方法之所以称为生成方法，是因为模型表示了给定输入x产生输出y的生成关系。这种方法一般建立在统计学和Bayes理论的基础之上。
    
    - 特点
        - 从统计的角度表示数据的分布情况，能够反映同类数据本身的相似度，但它不关心到底划分各类的那个分类边界在哪。
        - 生成方法能还原出联合概率分布，而判别方法不能。
        - 生成方法的学习收敛速度更快、即当样本容量增加的时候，学到的模型可以更快地收敛于真实模型。
        - 当存在隐变量时，仍可以用生成方法学习，此时判别方法不能用。
    - 典型的生成模型
        - 朴素贝叶斯分类器
        - 马尔科夫模型
        - 高斯混合模型
        - Mixtures of Multinomials
        - Mixtures of Gaussians
        - Mixtures of Experts
        - 隐马尔科夫模型(HMMs)
        - Sigmoidal Belief Networks
        - Latent Dirichlet Allocation
        - 马尔科夫随机场(Markov Random Fields)
        - 深度信念网络(DBN)

6. 判别模型<br>
    判别式模型（Discriminative Model）是直接对条件概率p(y|x;θ)建模。
    
    判别方法由数据直接学习决策函数$f(x)$或者条件概率分布$P(y|x)$作为预测的。判别模型利用正负例和分类标签，
    关注在判别模型的边缘分布。
    
    **判别方法强调的是：对给定的输入x，应该预测什么样的输出y。**
    
    - 特点
        - 判别方法寻找不同类别之间的最优分类面，反映的是异类数据之间的差异。
        - 判别方法利用了训练数据的类别标识信息，直接学习的是条件概率$P(Y|X)$或者决策函数$f(X)$，直接面对预测，往往学习的准确率更高；
        - 由于直接学习条件概率$P(Y|X)$或者决策函数$f(X)$，可以对数据进行各种程度上的抽象、定义特征并使用特征，因此可以简化学习问题。
        - 缺点是不能反映训练数据本身的特性
    - 典型的判别模型
        - k近邻法
        - 感知机
        - 决策树
        - logistic回归
        - 最大熵模型
        - SVM
        - boosting方法
        - 条件随机场(CRF)
        - 区分度训练
        - Linear Discriminant Analysis
        - 线性回归(Linear Regression)
        - Traditional Neural Networks
        - 神经网络(NN)
        - CART(Classification and Regression Tree)