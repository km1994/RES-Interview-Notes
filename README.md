# 推荐系统 百面百搭

<img src="img/微信截图_20230918094559.png" width="50%" >

>  NLP 面无不过 面试交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

<img src="img/微信截图_20210301212242.png" width="50%" >

- [推荐系统 百面百搭](#推荐系统-百面百搭)
  - [一、推荐系统导论篇](#一推荐系统导论篇)
  - [二、推荐系统机器学习篇](#二推荐系统机器学习篇)
    - [2.1 【关于 协同过滤篇】那些你不知道的事](#21-关于-协同过滤篇那些你不知道的事)
    - [2.2【关于 矩阵分解篇】那些你不知道的事](#22关于-矩阵分解篇那些你不知道的事)
    - [2.3 【关于 逻辑回归篇】 那些你不知道的事](#23-关于-逻辑回归篇-那些你不知道的事)
    - [2.4 FM 算法篇](#24-fm-算法篇)
    - [2.5 FFM 算法篇](#25-ffm-算法篇)
    - [2.6 GBDT+LR 篇](#26-gbdtlr-篇)
  - [三、推荐系统 深度学习篇](#三推荐系统-深度学习篇)
    - [3.1 AutoRec 篇](#31-autorec-篇)
    - [3.2 NeuralCF模型 篇](#32-neuralcf模型-篇)
    - [3.3 Deep Crossing模型 篇](#33-deep-crossing模型-篇)
    - [3.4 Wide＆Deep模型 篇](#34-widedeep模型-篇)
    - [3.5 FM与深度学习模型的结合 篇](#35-fm与深度学习模型的结合-篇)
  - [四、推荐系统 落地篇](#四推荐系统-落地篇)
  - [五、多角度审视推荐系统篇](#五多角度审视推荐系统篇)
  - [六、推荐系统 评估方法篇](#六推荐系统-评估方法篇)
  - [七、推荐系统 工程落地篇](#七推荐系统-工程落地篇)

## [一、推荐系统导论篇](introduction/)

- 1.1 什么是推荐系统？
- 1.2 推荐系统的作用？
- 1.3 推荐系统的意义？
- 1.4 推荐系统要解决的问题？
- 1.5 常用的推荐系统的逻辑框架是怎么样的呢？
- 1.6 常用的推荐系统的技术架构是怎么样的呢？
- 1.7 推荐系统算法工程师日常解决问题？
- 1.8 推荐系统算法工程师 处理的数据部分有哪些，最后得到什么数据？
- 1.9 推荐系统算法工程师 处理的模型部分有哪些，最后得到什么数据？
- 1.10 模型训练的方式？
- 1.11 推荐系统 的 流程是什么？
- 1.12 推荐系统 的 流程是什么？
- 1.13 推荐系统 与 搜索、广告 的 异同？
- 1.14 推荐系统 整体架构？

> [点击查看答案](https://articles.zsxq.com/id_grz7880endsk.html)

## [二、推荐系统机器学习篇](traditional_recommendation_model/)

### [2.1 【关于 协同过滤篇】那些你不知道的事](https://articles.zsxq.com/id_lje4bgibeb4i.html)

- 一、基础篇
  - 1.1 什么是协同过滤？
  - 1.2 协同过滤的推荐流程是怎么样？
- 二、基于用户的协同过滤 （User-CF-Based）篇
  - 2.1 基于用户的协同过滤 （User-CF-Based） 是什么？
  - 2.2 基于用户的协同过滤 （User-CF-Based） 的思想是什么？
  - 2.3 基于用户的协同过滤 （User-CF-Based） 的特点是什么？
- 三、基于物品的协同过滤 （Item-CF-Based）篇
  - 3.1 基于物品的协同过滤 （Item-CF-Based） 是什么？
  - 3.2 基于物品的协同过滤 （Item-CF-Based） 的思想是什么？
  - 3.3 基于物品的协同过滤 （Item-CF-Based） 的特点是什么？
  - 3.4 基于物品的协同过滤 （Item-CF-Based） 的具体步骤是什么？
- 四、User-CF-Based 与 Item-CF-Based 对比篇
  - 4.1 User-CF-Based 与 Item-CF-Based 的应用场景的区别
  - 4.2 User-CF-Based 与 Item-CF-Based 的存在问题的区别
- 五、User-CF-Based 与 Item-CF-Based 问题篇

> [点击查看答案](https://articles.zsxq.com/id_lje4bgibeb4i.html)

### [2.2【关于 矩阵分解篇】那些你不知道的事](https://articles.zsxq.com/id_4hjo78at5lj8.html)

- 一、动机篇
  - 1.1 为什么 需要 矩阵分解？
- 二、隐语义模型 介绍篇
  - 2.1 什么是 隐语义模型？
  - 2.2 隐语义模型 存在什么问题？
- 三、矩阵分解 介绍篇
  - 3.1 如何 获取 ⽤户矩阵Q 和 音乐矩阵P？
  - 3.2 矩阵分解 思路 是什么？
  - 3.3 矩阵分解 原理 是什么？
  - 3.4 如何 利用 矩阵分解 计算 用户 u 对 物品 v 的 评分？
- 四、矩阵分解 优缺点篇
  - 4.1 矩阵分解 存在什么问题？

> [点击查看答案](https://articles.zsxq.com/id_4hjo78at5lj8.html)

### [2.3 【关于 逻辑回归篇】 那些你不知道的事](https://articles.zsxq.com/id_3kstrwlvfuw0.html)

- 一、动机篇
  - 1.1 为什么 需要 逻辑回归？
- 二、逻辑回归 介绍篇
  - 2.1 逻辑回归 如何解决 上述问题？
  - 2.2 什么是逻辑回归
- 三、逻辑回归 推导篇
  - 3.1 逻辑回归 如何推导？
  - 3.2 逻辑回归 如何求解优化？
- 四、逻辑回归 推荐流程篇
  - 4.1 逻辑回归 推荐流程？
- 五、逻辑回归 优缺点篇
  - 5.1 逻辑回归 有哪些优点？
  - 5.2 逻辑回归 有哪些缺点？

> [点击查看答案](https://articles.zsxq.com/id_3kstrwlvfuw0.html)

### [2.4 FM 算法篇](https://articles.zsxq.com/id_4zqld440t2lm.html)

- 一、为什么要使用 FM？
- 二、FM 的思路是什么？
- 三、FM 的优点？
- 四、FM 的缺点？
- 五、POLY2 vs FM？

> [点击查看答案](https://articles.zsxq.com/id_4zqld440t2lm.html)

### [2.5 FFM 算法篇](https://articles.zsxq.com/id_uz5p2ategto9.html)

- 一、为什么要使用 FFM？
- 二、FFM 的思路是什么？
- 三、FM vs FFM？

> [点击查看答案](https://articles.zsxq.com/id_3kstrwlvfuw0.html)

### [2.6 GBDT+LR 篇](https://articles.zsxq.com/id_0l5gdn0wjtsp.html)

- 一、动机篇
  - 1.1 为什么 需要 GBDT+LR？
- 二、GBDT 介绍篇
  - 2.1 GBDT 的基础结构是什么样的？
  - 2.2 GBDT 的学习方式？
  - 2.3 GBDT 的思路？
  - 2.4 GBDT 的特点是怎么样？
  - 2.5 GBDT 所用分类器是什么？
  - 2.6 GBDT 解决二分类和回归问题的方式？
  - 2.7 GBDT 损失函数 是什么？
  - 2.8 构建分类GBDT的步骤 是什么？
  - 2.9 GBDT 优缺点篇？
- 三、GBDT+LR 模型介绍篇
  - 3.1 GBDT+LR 模型 思路是什么样？
  - 3.2 GBDT+LR 模型 步骤是什么样？
  - 3.3 GBDT+LR 模型 关键点是什么样？
  - 3.4 GBDT+LR 模型 本质是什么样？
- 四、GBDT+LR 优缺点篇
  - 4.1 GBDT+LR 的优点是什么？
  - 4.2 GBDT+LR 的缺点是什么？
- 五、问题讨论
  - 5.1 为什么要使用集成的决策树模型，而不是单棵的决策树模型？
  - 5.2 为什么建树采用GBDT而非RF？
  - 5.3 Logistic Regression是一个线性分类器，也就是说会忽略掉特征与特征之间的关联信息，那么是否可以采用构建新的交叉特征这一特征组合方式从而提高模型的效果？
  - 5.4 GBDT很有可能构造出的新训练数据是高维的稀疏矩阵，而Logistic Regression使用高维稀疏矩阵进行训练，会直接导致计算量过大，特征权值更新缓慢的问题？
  - 5.5 FM 因为采用FM对本来已经是高维稀疏矩阵做完特征交叉后，新的特征维度会更加多，并且由于元素非0即1，新的特征数据可能也会更加稀疏，那么怎么办？
  - 5.6 为什么要将GBDT与LR融合？

> [点击查看答案](https://articles.zsxq.com/id_0l5gdn0wjtsp.html)


## 三、推荐系统 深度学习篇

### 3.1 AutoRec 篇

- 什么是自编码器?
- AutoRec 思路 是什么？
- AutoRec 基本原理是什么？
- AutoRec模型的结构 长什么样子？
- AutoRec模型的特点？
- AutoRec模型的存在问题？
  
> [点击查看答案](https://articles.zsxq.com/id_dntrd6igjk9i.html)

### 3.2 NeuralCF模型 篇

- 为什么需要NeuralCF模型？
- NeuralCF模型 的 普通结构？
- NeuralCF模型 的 混合结构？
- NeuralCF模型 主要思想？
- NeuralCF模型的优势和局限性？

> [点击查看答案](https://articles.zsxq.com/id_bjd8e1r6kow3.html)

### 3.3 Deep Crossing模型 篇

- 为什么需要 Deep Crossing？
- Deep Crossing 模型的所用特征 是什么？
- Deep Crossing 模型的模型结构？

> [点击查看答案](https://articles.zsxq.com/id_wl94fgqn0r5t.html)

### 3.4 Wide＆Deep模型 篇

- 模型的记忆能力与泛化能力
- Wide＆Deep模型 模型结构？
- Wide＆Deep模型 模型 Trick？
- Wide＆Deep模型 优点是什么？
- Wide＆Deep模型的影响力？
- Wide＆Deep模型的进化——Deep＆Cross模型？

> [点击查看答案](https://articles.zsxq.com/id_di0tp2qalgdx.html)

### 3.5 FM与深度学习模型的结合 篇

- 为什么需要 DeepFM？
- DeepFM 结构 介绍一下？
- DeepFM 思路？
- DeepFM 与 Deep＆Cross模型 异同点？

> [点击查看答案](https://articles.zsxq.com/id_wpbdemx6amp9.html)

## 四、推荐系统 落地篇


## 五、多角度审视推荐系统篇


## 六、推荐系统 评估方法篇

## 七、推荐系统 工程落地篇
