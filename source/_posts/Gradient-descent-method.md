---
title: Gradient descent method
date: 2024-12-17 15:08:58
author: GaoPan
categories: 
- 最优化
tags: 
- 梯度下降
- 无约束优化算法
---
# 梯度下降法
梯度下降法是一种用于优化函数的优化算法，它通过迭代地更新参数以找到函数的最小值。
<!--more-->
## 算法介绍
对于光滑的函数$f(\boldsymbol{x})$, 在给定起始点$\boldsymbol{x}^k$处，我们需要选择一个下降方向$\boldsymbol{d}^k$。 由此可以得到新的迭代点$\boldsymbol{x}^{k+1} = \boldsymbol{x}^k + \alpha \boldsymbol{d}^k$，其中$\alpha$为步长。最自然的想法是选取函数下降最快的方向作为$\boldsymbol{d}^k$。那么，函数下降最快的方向是什么方向呢？
由函数的泰勒展开式可以得到：
$$
f(\boldsymbol{x}^k + \alpha \boldsymbol{d}^k) = f(\boldsymbol{x}^k) + \alpha\nabla f(\boldsymbol{x}^k)^\top \boldsymbol{d}^k + \mathcal{O}(\alpha^2\|\boldsymbol{d}^k\|^2)
$$
因此，$\boldsymbol{d}^k$应该使得$\nabla f(\boldsymbol{x}^k)^\top \boldsymbol{d}^k$尽可能小，所以$\boldsymbol{d}^k$应该取梯度的反方向作为最快的下降方向， 即$\boldsymbol{d}^k = -\nabla f(\boldsymbol{x}^k)$. 因此，梯度下降算法的迭代公式为：
$$
\boldsymbol{x}^{k+1} = \boldsymbol{x}^k - \alpha \nabla f(\boldsymbol{x}^k)
$$
其中步长$\alpha$可以是固定步长也可以是动态步长。

## 应用举例-LASSO问题求解
本节将介绍如何使用梯度下降法求解LASSO问题。LASSO问题是线性回归问题加上L1范数正则化项，即：
$$
\min f(\boldsymbol{x}) = \frac{1}{2}\|A\boldsymbol{x}- b\|^2 + \mu\|\boldsymbol{x}\|_1
$$
注意LASSO问题并不是一个光滑的问题，在某些点上其导数可能不存在，因此不能直接使用梯度法求解。考虑到非光滑项为L1范数，如果能找到一个光滑的函数来近似L1范数，就可以通过梯度下降法求解。实际上，我们可以考虑下面的一维光滑函数：
$$l_{\delta}(x)=\begin{cases}\frac{1}{2\delta}x^{2},&|x|<\delta,\\|x|-\frac{\delta}{2},&\text{其他}.&&&\end{cases}$$
当$\delta\rightarrow0$时，光滑函数$l_{\delta}(x)$会和绝对值函数$|x|$越来越相似。因此，我们构造光滑化的LASSO函数为：
$$
\min f(\boldsymbol{x}) = \frac{1}{2}\|A\boldsymbol{x}- b\|^2 + \mu L_{\delta}(x),
$$
其中$\mu$是正则化参数，$\delta$是光滑化参数。$L_{\delta}(x)=\sum_i^nl_\delta(x_i)$。
计算出光滑化的LASSO函数的梯度为：
$$\nabla f_\delta(x)=A^\mathrm{T}(Ax-b)+\mu\nabla L_\delta(x),$$
其中$\nabla L_\delta(x)$是针对逐个变量定义的：
$$(\nabla L_{\delta}(x))_{i}=\begin{cases}\mathrm{sign}(x_{i}),&|x_{i}|>\delta,\\\frac{x_{i}}{\delta},&|x_{i}|\leqslant\delta.&&&\end{cases}$$