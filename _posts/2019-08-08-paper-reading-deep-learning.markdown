---
layout: post
title:  "论文阅读"
date:   2019-08-08 10:08:00 +0800
date_update: 2019-08-09 01:02:00 +0800
categories: 学术科研
tags: [机器学习,深度学习]
author: 幽玄
use_mathjax: true
---

## 可变形模板 Deformable Template

<span>&#9733;&#9733;&#9733;</span> Learning Conditional Deformable Templates with Convolutional Networks. [arXiv 1908.02738](https://arxiv.org/abs/1908.02738)
- 关键词：条件化模板，形变场，极大似然估计，微分同胚变换 diffeomorphism transform
- 医疗图像处理，图像配准，对齐 alignment，对应关系 correspondence
- 模板可以看成是对数据变化模式的概括。

作者信息：
- [Adrian Vasile Dalca](http://www.mit.edu/~adalca/index.html)：研究方向主要是概率模型和医疗图像分析。博士毕业于MIT CSAIL的[医疗视觉研究组](http://groups.csail.mit.edu/vision/medical-vision/)（Medical Vision Group），导师是[Polina Golland](http://people.csail.mit.edu/polina/)。目前在MIT和哈佛医学院做博士后。
-  Marianne Rakic：来自ETH D-ITET和MIT CSAIL的研究人员，没有找到个人主页。
- [John Guttag](https://people.csail.mit.edu/guttag/)：比较老的一个教授，有自己的[Wikipedia页面](https://en.wikipedia.org/wiki/John_Guttag)。目前在MIT CSAIL带领[数据驱动推断研究组](https://www.csail.mit.edu/research/data-driven-inference-group)（Data-Driven Inference Group），他曾经担任MIT EECS的主任，是美国科学院（American Academy of Arts and Sciences）院士。主要研究方向是医疗影像处理，但是同时也涉及运动和财经数据分析相关的领域。
- [Mert Rory Sabuncu](http://sabuncu.engineering.cornell.edu/)：康奈尔大学的副教授，主要研究生物医学数据分析，尤其关注脑和神经科学。博士毕业于普林斯顿大学，曾跟随[Polina Golland](http://people.csail.mit.edu/polina/)在MIT CSAIL做博士后。

这个工作的任务其实是生成一些可变形模板，仅考虑怎么更好地生成，而没有太讨论用这个模板来做什么（实验中好像有简单涉及）。这里所谓的“更好”，主要包括两个方面，一是生成速度快，包括生成模板本身和图像到模板（或者模板到图像）的变换；二是能够以属性为条件来针对性地生成模板，并且学习这种条件化的模板时能够用到数据集中所有的数据，不需要对数据集进行实际划分来分别去学习模板。

**方法**：做法上其实比较简单，首先以属性\\(a_i\\)作为输入用一个网络$$g_{t, \theta_t}(a_i)$$预测模板\\(t\\)，然后以模板\\(t\\)和图像\\(x_i\\)作为输入用另一个网络\\(g_{v, \theta_v}(t, x_i)\\)预测速度场\\(v_i\\)（velocity field），再根据速度场通过积分（integration layer）计算出对应的形变场\\(\phi_{v_i}\\)（deformation field），之后通过形变场\\(\phi_{v_i}\\)将模板\\(t\\)变换为输入（对输入\\(x_i\\)进行重构）。

![image.png](https://i.loli.net/2019/08/08/tgx51o4IJrDbmBf.png)

其中对于模板\\(t\\)，如果不采用条件化的生成方式，例如没有属性信息的时候，那可以直接学习\\(t\\)，而不需要学习一个生成\\(t\\)的模型。

感觉这个方法的关键点主要有两个：采用神经网络作为预测器，采用随机梯度下降进行模型优化。方法中采用了两个神经网络来分别预测模板和速度场（进行图像校准），构建起一个端到端的模型，然后这样就可以采用随机梯度下降法来对整个模型进行优化，这样不管是训练还是测试都会比之前的方法高效。

其实在计算机视觉领域，这样的模型应该算是很普通和常见的了，将现有预测器等替换为神经网络或者用神经网络预测未知量从而构造**端到端**的**基于学习的**（learning based）模型是一个基本套路，而**条件化**的思想在各种生成模型中已经得到了广泛应用。不过这个工作侧重于医学图像的处理（虽然实验大部分是在MNIST数据集上做进行），可能在这个方向，这个模型还是具有一定的新意并且解决了现有方法的一些问题。

虽然在方法上我感觉不是很新鲜，但是可能只是从最终模型的结构上来看。这个工作实际上是设计了一个生成图像的概率模型，通俗的说这主要体现在损失函数上，即模板变换之后重构图像，**如何衡量重构结果和原始输入之间的差异，以及加入的约束**。

自己的一点想法：
- 生成模板的时候可以采用残差的方式，平均图像+残差，其中残差通过属性来预测，平均图像作为初始化，其也可以在学习的过程中更新。
- 如果校准的时候不针对每个图像进行优化，而是用一个学好的网络来预测速度场/形变场，这样也许误差会更大？换言之模型的泛化能力对校准质量非常关键。
- 不同属性的图像采用不同的模板，那么相互之间怎么比较形变场呢？似乎只能在同属性的图像之间比较了。不过这样看来不同属性带来的差别也许能方便地看出来，直接比较对应的模板就可以，学出来的可能比平均图像要更有代表性一点。
- 为什么不直接预测形变场和在形变场上加约束，而要先预测速度场再积分得到形变场？可能和微分同胚变换有关？

下面再具体看一下文章给出的**概率模型**。首先是问题的形式化，这里要求解的就是两个神经网络的参数，一个生成模板，一个进行配准，用最大后验估计（论文中说的是“似然”，但是我觉得这里应该是最大后验估计/贝叶斯估计）对两个任务进行优化（注意下面的\\(\mathcal{V}\\)就是对应于用来配准的神经网络）：
\\[ \begin{align}
\hat{\theta_t}, \hat{\mathcal{V}} & = \arg\max_{\theta_t, \mathcal{V}} \log p_{\theta_t}(\mathcal{V} \vert \mathcal{X}, \mathcal{A}) \newline
& = \arg\max_{\theta_t, \mathcal{V}} \log p_{\theta_t}(\mathcal{X} \vert \mathcal{V}; \mathcal{A}) \log p(\mathcal{V})
\end{align} \\]
这里包含两项：似然和先验，其中先验可以用来对形变施加约束，让得到的形变满足一些所希望的性质。

对于**先验**，作者引入了两方面的假设：光滑和无偏。

对于**似然**，作者采用高斯分布建模，以变换之后的模板（即对输入的重构）作为均值，加上一个小的噪声作为方差，换言之，将真实的输入看成是重构结果在邻域内进行一个小的扰动所得到的，其实这样建模就相当于是计算均方误差，只不过前面多了一个系数（跟高斯分布的方差有关，方差越小，系数越大，即要求重构误差越小）。关于均方误差和极大似然估计/高斯分布的关系，可以参考这个博文：[MSE as Maximum Likelihood](https://www.jessicayung.com/mse-as-maximum-likelihood/)。

论文公式(6)应该是写错了，第一项感觉就很奇怪，最小化负的均方误差，相当于在最大化重构的均方误差！？高斯密度函数的指数部分，系数是负的，这里是负对数似然，加了负号，因而要取反变成正的。后面先验的部分照抄了论文前面的公式(4)，也没有取反，符号也错了。

关于微分同胚变换，知乎上有人给了从苏大强变换到吴彦祖的图（[知乎 - 如何理解微分同胚的概念？](https://www.zhihu.com/question/27551225/answer/680178307)），虽然感觉很有意思，但是我还是没太具体明白这个变换的特性。。。

<img src="https://pic3.zhimg.com/50/v2-386fe9c5d5da9029e1f68127e9c001fd_hd.gif" alt="微分同胚变换" height="150">
