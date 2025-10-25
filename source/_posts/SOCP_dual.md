---
title: 二阶锥规划示例及其对偶
date: {{2025-4-14}}
author: GaoPan
categories: [最优化]
tags: [二阶锥优化, SOCP]
# 插件
katex: true
---
## 二阶锥规划(SOCP)
二阶锥规划(Second-order cone program)，是指目标函数为线性函数，约束条件中的决策变量在二阶锥中。
## SOCP的原始和对偶问题
SOCP的原始问题为：
$$
\begin{array}{ll}
\min   & \boldsymbol{f}^\top\boldsymbol{x}\\
\mathrm{s.~t.} & \|\boldsymbol{A}_i\boldsymbol{x} + \boldsymbol{b}_i\|_2 \leq \boldsymbol{c}_i^\top \boldsymbol{x} + d_i, \quad i=1,\ldots,m 
\end{array}
$$
其中$A_i\in\mathbb{R}^{n_i\times n},\boldsymbol{b}_i\in\mathbb{R}^{n_i},\boldsymbol{c}_i\in\mathbb{R}^n,d_i\in\mathbb{R},\boldsymbol{x}\in\mathbb{R}^n, i=1,\ldots,m$。

<!--more-->

对偶问题为：
$$
\begin{array}{ll}
\max   & -\sum_{i=1}^m(\boldsymbol{u}_i^\top \boldsymbol{b}_i + v_id_i)\\
\mathrm{s.~t.} & \sum_{i=1}^m \boldsymbol{A}_i^\top\boldsymbol{u}_i + \sum_{i=1}^mv_i\boldsymbol{c}_i = \boldsymbol{f},\quad i=1,\ldots,m, \\
&\|\boldsymbol{u}_i\|_2\leq v_i, \quad i=1,\ldots,m,
\end{array}
$$
其中，$\boldsymbol{u}_i\in\mathbb{R}^n$，$v_i\in\mathbb{R}$，$i=1,\ldots,m$。

## 证明: 
令$\boldsymbol{y}_i = \boldsymbol{A}_i\boldsymbol{x} + \boldsymbol{b}_i$，$t_i = \boldsymbol{c}_i^\top + d_i$，则SOCP原问题等价于：
$$
\begin{array}{ll}
\min   & \boldsymbol{f}^\top\boldsymbol{x}\\
\mathrm{s.~t.} & \|\boldsymbol{y}_i\|_2 \leq t_i, \quad i=1,\ldots,m \\
& \boldsymbol{y}_i = \boldsymbol{A}_i^\top\boldsymbol{x} + \boldsymbol{b}_i, \quad i=1,\ldots,m \\
& t_i = \boldsymbol{c}_i^\top\boldsymbol{x}+d_i, \quad i=1,\ldots,m \\
\end{array}
$$
引入拉格朗日乘子$\boldsymbol{\lambda}\in\mathbb{R}^m,\boldsymbol{\mu}_i\in\mathbb{R}^{n_i},i=1,\ldots,m，\boldsymbol{\nu}\in\mathbb{R}^m$，则可以写出上述约束优化问题的拉格朗日函数：
$$\begin{aligned}
L(\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{t}, \boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu})& =  \boldsymbol{f}^\top\boldsymbol{x} + \sum_{i=1}^m\lambda_i(\|\boldsymbol{y}_i\|_2 -t_i) + \sum_{i=1}^m\boldsymbol{\mu}_i^\top(\boldsymbol{y}_i-\boldsymbol{A}_i^\top\boldsymbol{x}-\boldsymbol{b}_i) + \sum_{i=1}^m\nu_i(t_i - \boldsymbol{c}_i^\top\boldsymbol{x})\\
&=  (\boldsymbol{f}-\sum_{i=1}^m\boldsymbol{A}_i^\top -\sum_{i=1}^m\boldsymbol{u}_i\boldsymbol{c}_i)^\top\boldsymbol{x} + \sum_{i=1}^m(\lambda_i\|\boldsymbol{y}_i\|_2+\boldsymbol{\mu}_i^\top\boldsymbol{y}_i) \\
&\quad+ \sum_{i=1}^m(-\lambda_i+\nu_i)t_i-\sum_{i=1}^m(\boldsymbol{\mu}_i^\top\boldsymbol{b}_i+\nu_id_i)
\end{aligned}
$$
原始问题的极小极大表示为：
$$\begin{aligned}
p^\star = \min_{\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{t}}\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu}} L(\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{t}, \boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu})
\end{aligned}
$$
那么对偶问题的极大极小表示为：
$$\begin{aligned}
d^\star = \max_{\boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu}}\min_{\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{t}} L(\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{t}, \boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu})
\end{aligned}
$$
针对内层关于$\boldsymbol{x}$的$\min$问题，当且仅当满足：
$$
\boldsymbol{f}-\sum_{i=1}^m\boldsymbol{A}_i^\top -\sum_{i=1}^m\boldsymbol{u}_i\boldsymbol{c}_i = 0,
$$
该问题才有下界。
针对内层关于$\boldsymbol{y}_i$的$\min$问题，我们可以得到：
$$
\min_{\boldsymbol{y}} \lambda_i\|\boldsymbol{y}_i\|_2+\boldsymbol{\mu}_i^\top\boldsymbol{y}_i = \left\{\begin{array}{ll}{0}&{\|\boldsymbol{\mu}_{i}\|_{2}\leq\lambda_{i}}\\{-\infty}&{\mathrm{otherwise}.}\end{array}\right.
$$
当且仅当$\lambda_i = \nu_i$，内层关于$\boldsymbol{t}$的$\min$才有下界。
因此，SOCP的对偶问题等价于：
$$
\begin{array}{ll}
\max   & -\sum_{i=1}^m(\boldsymbol{\mu}_i^\top\boldsymbol{b}_i+\nu_id_i)\\
\mathrm{s.~t.} & \boldsymbol{f} - \sum_{i=1}^m \boldsymbol{A}_i^\top\boldsymbol{\mu}_i - \sum_{i=1}^m\nu_i\boldsymbol{c}_i =0,\quad i=1,\ldots,m, \\
&\|\boldsymbol{\mu}_i\|\leq \lambda_i, \quad i=1,\ldots,m,\\
&-\lambda_i+\nu_i=0, \quad i=1,\ldots,m,
\end{array}
$$
令$\boldsymbol{\mu}_i=\boldsymbol{u}_i, \lambda_i=\nu_i=v_i$，则可以得到SOCP的对偶问题。

## 示例
基于椭球不确定集合的鲁棒线性规划是一个典型的二阶锥规划问题。下面列举一个鲁棒对应(RC)的形式：
$$\begin{aligned}&\min\quad c^\top x\\
&\mathrm{s.~t.}\quad(a_i+u_i)^Tx\leq b_i\text{ for all }\|u_i\|_2\leq1,\quad i=1,\ldots,m,
\end{aligned}$$
其中，$a_i\in\mathbb{R}^n$，$b_i\in\mathbb{R}$，$u_i\in\mathbb{R}^n$，$c\in\mathbb{R}^n$。
不确定参数$u_i$的集合被限制为一个$\ell_2$范数构成的椭球集合，即$\|u_i\|_2\leq1$。
通过对偶转换可以得到RC的可处理形式:
$$\begin{aligned}&\min\quad c^Tx\\&\mathrm{s.~t.}\quad a_i^Tx+\|x\|_2\leq b_i,\quad i=1,\ldots,m,\end{aligned}$$


## Python实现求解
CVXPY库可以直接求解SOCP。
```python
import cvxpy as cp
import numpy as np

# Generate a random feasible SOCP.
m = 3
n = 5
np.random.seed(2)
c = np.random.random(n)
A = np.random.random((m, n))
x0 = np.random.random(n)
b = A@x0

x = cp.Variable(n)
obj = cp.Minimize(c @ x)
soc_constraints = [
    cp.SOC(b[i] - A[i]@x, x) for i in range(m)
]
prob = cp.Problem(obj, soc_constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)
for i in range(m):
    print("soc constraint", soc_constraints[i].dual_value)
```
求解结果如下：
```
status: optimal
optimal value -2.487555966760468
optimal var [-1.37881038 -0.81856901 -1.7036923  -1.23789512 -0.9273785 ]
soc constraint [array([3.71192161]), array([[1.82467162],
       [1.08325999],
       [2.25507434],
       [1.63818725],
       [1.22738155]])]
soc constraint [array([1.2546655]), array([[0.61681542],
       [0.36620225],
       [0.76212621],
       [0.55379853],
       [0.41483849]])]
soc constraint [array([3.29587551e-08]), array([[1.10774611e-08],
       [8.15413818e-10],
       [1.51970753e-08],
       [1.14059208e-08],
       [1.06823249e-08]])]
```
