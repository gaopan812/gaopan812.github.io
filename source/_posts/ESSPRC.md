---
title: 资源受限的基本最短路问题
date: 2025-11-06
author: GaoPan
categories: 
- 运筹优化
tags: 
- 动态规划
- 标签算法
katex: true
---

## 资源受限的基本最短路问题
在使用列生成算法求解带时间窗限制的车辆路径规划问题时，定价子问题是一个资源受限的基本最短路问题（Elementary Shortest Path Problem with Resource Constraints，ESSPRC）。在文献中[^1]证明了ESSPRC问题是强$NP$难的，因此针对大规模的问题，使用求解器求解速度较慢。
<!--more-->

## 问题描述
有一辆容量为$Q$的货车，从起点$o$出发，为途中的客户送货，最后到达终点$d$， 货车在配送时需要满足时间和容量的限制。假设有$m$个客户，这些客户分布在不同的地点, 客户用集合$C$表示，$C = \{1, 2, \ldots, m\}$。每个客户$i$有不同的服务时间窗$[a_i, b_i]$和需求量$q_i$。本问题考虑的是硬时间窗，也就是说给客户$i$的服务时间必须落在时间窗 $[a_i, b_i]$内。将起点$o$记作0，终点$d$设为$m+1$，则所有的节点用集合$V$表示，$V = \{0, 1, 2, \ldots, m+1\}$。从节点$i$到节点$j$的运输时间为$t_ij$, 运输成本为$c_ij$。问题的目标是，在货车容量不超载，每个客户最多被访问一次且服务开始时刻落在相应的服务时间窗内的情况下，最小化运输总成本。

## 数学模型
主要参考的是 Column Generation 书[^2]中的第三章，引用 Brian Kallehauge 等人提出的模型：

- 变量：
    - $x_{ij} = 
        \begin{cases}
        1, & \text{如果最短路径中包含弧 } (i,j) \in V \times V \\
        0, & \text{否则}
        \end{cases}$

    - $s_i$: 到达点 $i\in V$ 的时刻
- 目标函数：最小化运输总成本
   $$\min\quad\sum_{i\in V}\sum_{j\in V}c_{ij}x_{ij}$$
- 约束条件：
    - 容量约束，规定总运输量不能超过货车容量。
    $$\sum_{i\in C}q_i\sum_{j\in V}x_{ij}\leq Q$$
    - 从起点出发只能一次。
    $$\sum_{j\in V}x_{0j}=1$$
    - 到达终点只能一次。
    $$\sum_{i\in V}x_{i,m+1}=1$$
    - 客户节点之间的流平衡约束。
    $$\sum_{i\in V}x_{ih}=\sum_{i\in V}x_{hj}\quad\forall h\in C$$
    - 硬时间窗约束。
    $$a_i\leq s_i\leq b_i\quad\forall i\in V$$
    - 车辆的行驶时间约束，也实现了子回路消除功能。
    $$s_i+t_{ij}\leq s_j+M_{ij}(1-x_{ij})\quad\forall i\in V,\forall j\in V$$
    - 变量值域。
    $$x_{ij}\in\{0,1\}\quad\forall i\in V,\forall j\in V$$
    - 任何节点都不能是自己的前继节点或者后继节点。
    $$x_{ii}=0\quad\forall i\in V$$
**补充：** $M_{ij}$的取值，满足足够大但是尽量小，可以设为$(b_i-a_j+t_{ij})^+\:\forall i\in V,\forall j\in V$。

## ESPPRC的松弛
由于ESPPRC是强$NP$难问题，在求解时通常会松弛问题中的某些约束。其中，应用非常广泛的就是允许路径中存在子回路，即允许客户被多次访问。如此，问题就变成了资源受限最短路问题（SPPRC）。

### 数学模型
基于前面介绍过的ESPPRC的模型，Brian Kallehauge 等人对SPPRC也给出了如下数学模型：
在模型中定义了一个额外的集合$K= \{1, 2, \ldots, |K|\}$，用来表示路径中弧的顺序。由于每个客户都可以多次访问，所以无法确定最后求得的路径中包含几条弧，但是路径所包含的弧的个数的上限$|K|$是可以确定的。路径中弧的数量受限于容量资源和时间资源，所以$|K|=\min\{\lfloor\frac{Q}{\min_{i\in V}\{q_i\}}\rfloor,\lfloor\frac{b_{m+1}}{\min_{(i,j)\in V\times V}\{t_{ij}\}}\rfloor\}$。
需要注意的是，虽然在SPPRC中，每个客户允许被多次访问，但是，起点和终点在路径中还是只能出现一次。
- 变量：
    - $x_{ijk}=\begin{cases}1,&\text{如果路径中的第 }k\in K\text{ 条弧是 }(i,j)\in V\times V\\0,&\text{否则}&\end{cases}$

    - $s_{ik}$: 第$k\in K$条弧上到达点$i\in V$的时刻
- 目标函数：最小化运输总成本
 $$\min\quad\sum_{i\in V}\sum_{j\in V}\sum_{k\in K}c_{ij}x_{ijk}$$
- 约束条件：
    - 路径中的弧必须按序连续出现：若第 $k$ 条弧在路径中（$k \geq 2$），则第 $k-1$ 条弧也必在路径中。
    $$\sum_{i\in V}\sum_{j\in V}x_{ijk}\leq\sum_{i\in V}\sum_{j\in V}x_{i,j,k-1}\quad\forall k\in K\setminus\{1\}$$
    - 容量约束，规定总运输量不能超过货车容量。
    $$\sum_{i\in C}q_i\sum_{j\in V}\sum_{k\in K}x_{ijk}\leq Q$$
    - 从起点出发只能一次。
    $$\sum_{j\in V}\sum_{k\in K}x_{0jk}=1$$
    - 起始节点为0的弧必须是路径中的第一条弧。
    $$\sum_{j\in V}x_{0j1} = 1$$
    - 流平衡约束，规定对于每个客户节点来说，在第 $(k-1)\in K\setminus\{1\}$ 条弧的入度等于在第$k\in K\setminus\{1\}$条弧上的出度。
    $$\sum_{i\in V}x_{ih,k-1}=\sum_{j\in V}x_{hjk}\quad\forall h\in C,\forall k\in K\setminus\{1\}$$
    - 到达终点只能一次。
    $$\sum_{i\in V}\sum_{k\in K}x_{i,m+1,k}=1$$
    - 硬时间窗约束。
    $$a_i\leq s_{ik}\leq b_i\quad\forall i\in V,\forall k\in K$$
    - 车辆的行驶时间约束。
    $$s_{ik}+t_{ij}\leq s_{jk}+M_{ij}(1-x_{ijk})\quad\forall i\in V,\forall j\in V,\forall k\in K$$
    - 变量值域。
    $$x_{ijk}\in\{0,1\}\quad\forall i\in V,\forall j\in V,\forall k\in K$$
    - 任何节点都不能是自己的前继或者后继节点。
    $$x_{iik}=0\quad\forall i\in V,\forall k\in K$$
    - 不能存在两条下标（index）一样的弧。
    $$\sum_{i\in v}\sum_{j\in V}x_{ijk}\leq1\quad\forall k\in K$$
    - 节点$m+1$不能有后继节点。
    $$\sum_{i\in V}\sum_{k\in K}x_{m+1,j,k}=0$$
    - 节点0不能有前继节点。
    $$\sum_{i\in V}\sum_{k\in K}x_{i0k}=0$$
    - 服务开始时刻必须是单调递增的。
    $$s_{i,k-1}\leq s_{ik}\quad\forall i\in V,\forall k\in K\setminus\{1\}$$

[^1]: Desrochers, M., Desrosiers, J., and Solomon, M. (1992). A new optimization algorithm for the vehicle routing problem with time windows. Operations Research, 40:342-354.

[^2]: Desaulniers, Guy, Jacques Desrosiers, and Marius M. Solomon, eds. Column generation. Vol. 5. Springer Science & Business Media, 2006.

