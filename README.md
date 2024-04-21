## 项目简介

这是一个没有使用深度学习框架，只借助Python和Numpy实现的RNN网络。由于没有类似Pytorch的自动求梯度的功能，我们需要手动计算Loss到网络各层参数的梯度，这涉及到数学公式的推导，我们将数学公式的推导过程呈现在README文档中。
如果文档不能正常显示数学公式，请移步我的博客 [Recurrent Neural Networks](https://onexiaophai.gitee.io/2022/05/19/数学/循环神经网络介绍/) 查看。

## Conda环境

```shell
conda create --name env_for_rnn python=3.9 numpy pandas matplotlib sympy ipykernel scikit-learn
conda activate env_for_rnn
```

## Recurrent Neural Networks

![img](https://xiaophai-typora.oss-cn-shanghai.aliyuncs.com/2256672-cf18bb1f06e750a4.jpg)

## RNN的数学描述

### 输入层

网络的输入是一串**m维向量序列**  $\boldsymbol{x^1},\boldsymbol{x^2},\cdots,\boldsymbol{x^t},\cdots$

```math
\boldsymbol{x^1} =
\begin{bmatrix}
x^1_1\\x^1_2\\\vdots\\x^1_m
\end{bmatrix},
\boldsymbol{x^2} =
\begin{bmatrix}
x^2_1\\x^2_2\\\vdots\\x^2_m
\end{bmatrix},
\cdots,
\boldsymbol{x^t} =
\begin{bmatrix}
x^t_1\\x^t_2\\\vdots\\x^t_m
\end{bmatrix},
\cdots
```

### 循环层

网络的状态是一串**n维向量序列** $\boldsymbol{s^0},\boldsymbol{s^1},\boldsymbol{s^2}\cdots,\boldsymbol{s^t},\cdots$

```math
\begin{gather*}
\begin{bmatrix}
s^t_1\\s^t_2\\\vdots\\s^t_n
\end{bmatrix}=
f\left(
\begin{bmatrix}
u_{11}&u_{12}&\cdots&u_{1m}\\
u_{21}&u_{22}&\cdots&u_{2m}\\
\vdots&\vdots&\ddots&\vdots\\
u_{n1}&u_{n2}&\cdots&u_{nm}\\
\end{bmatrix}
\begin{bmatrix}
x^t_1\\x^t_2\\\vdots\\x^t_m
\end{bmatrix}
+
\begin{bmatrix}
w_{11}&w_{12}&\cdots&w_{1n}\\
w_{21}&w_{22}&\cdots&w_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
w_{n1}&w_{n2}&\cdots&w_{nn}\\
\end{bmatrix}
\begin{bmatrix}
s^{t-1}_1\\s^{t-1}_2\\\vdots\\s^{t-1}_n
\end{bmatrix}
+
\begin{bmatrix}
b^R_1\\b^R_2\\\vdots\\b^R_n
\end{bmatrix}
\right)
\\
t = 1,2,\cdots
\end{gather*}
```

### 输出层

网络的输出是一串**m维的向量序列** $\boldsymbol{o^{1}},\boldsymbol{o^{2}},\cdots,\boldsymbol{o^{t}},\cdots$

```math
\begin{gather*}
\begin{bmatrix}
o^t_1\\o^t_2\\\vdots\\o^t_m
\end{bmatrix}=
g\left(
\begin{bmatrix}
v_{11}&v_{12}&\cdots&v_{1n}\\
v_{21}&v_{22}&\cdots&v_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
v_{m1}&v_{m2}&\cdots&v_{mn}\\
\end{bmatrix}
\begin{bmatrix}
s^t_1\\s^t_2\\\vdots\\s^t_n
\end{bmatrix}
+
\begin{bmatrix}
b^O_1\\b^O_2\\\vdots\\b^O_m
\end{bmatrix}
\right)
\\
t = 1,2,\cdots
\end{gather*}
```

### 网络的输出

网络在 $t$ 时刻的输出 $\boldsymbol{o^t}$ 由前面各时刻的输入 $\boldsymbol{x^t},\boldsymbol{x^{t-1}},\cdots,\boldsymbol{x^1}$和初始状态 $\boldsymbol{s^0}$ 决定

(下面的推导式中省略了偏置项 $\boldsymbol{b}$)

```math
\begin{split}
\boldsymbol{o^t} &= g\left( V\boldsymbol{s^t}\right) \\
&= g\left( Vf\left(U\boldsymbol{x^t}+W\boldsymbol{s^{t-1}}\right)\right) \\
&=g\left( Vf\left(U\boldsymbol{x^t}+Wf\left(U\boldsymbol{x^{t-1}}+W\boldsymbol{s^{t-2}}\right)\right)\right) \\
&\vdots\\
&=g\left( Vf\left(U\boldsymbol{x^t}+Wf\left(U\boldsymbol{x^{t-1}}+Wf\left(U\boldsymbol{x^{t-2}}+\cdots+ Wf\left(U\boldsymbol{x^1}+W\boldsymbol{s^0}\right)\right)\right)\right)\right) \\
\end{split}
```

### 网络输出的误差

网络在每个 $t$ 时刻的输出 $\boldsymbol{o^t}$ 都对应一个目标向量 $\boldsymbol{t}^t$ (target),   每个时刻都对应一个误差,   用$E^t$来表示 ,   $E^t$ 是关于 $\boldsymbol{o^t}$和 $\boldsymbol{t}^t$ 的函数,   例如采用二范数的平方表示误差,   误差函数如下计算

```math
\begin{split}
E^t &= \frac{1}{2}\|\boldsymbol{o}^t-\boldsymbol{t}^t\|_2^2
\\
&=
\frac{1}{2}\sum_{i=1}^m (o^t_i-t^t_i)^2
\end{split}
```

## 梯度的计算(Back Propagate Through Time, BPTT)

### 循环层到输出层

记输出层 $t$ 时刻的输入向量为 $\boldsymbol{\xi}^{t}$

```math
\begin{split}
\begin{bmatrix}
o^t_1\\o^t_2\\\vdots\\o^t_m
\end{bmatrix}=
g\left(
\begin{bmatrix}
\xi^t_1\\
\xi^t_2\\
\vdots\\
\xi^t_m
\end{bmatrix}
\right)
,\quad
\begin{bmatrix}
\xi^t_1\\
\xi^t_2\\
\vdots\\
\xi^t_m
\end{bmatrix}=
\begin{bmatrix}
v_{11}&v_{12}&\cdots&v_{1n}\\
v_{21}&v_{22}&\cdots&v_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
v_{m1}&v_{m2}&\cdots&v_{mn}\\
\end{bmatrix}
\begin{bmatrix}
s^t_1\\s^t_2\\\vdots\\s^t_n
\end{bmatrix}
+
\begin{bmatrix}
b^O_1\\b^O_2\\\vdots\\b^O_m
\end{bmatrix}
\end{split}
```

```math
\begin{split}
\frac{\partial E^t}{\partial v_{ij}} &=
\frac{\partial E^t}{\partial \xi^t_i}\cdot\frac{\partial \xi^t_i}{\partial v_{ij}}
=\frac{\partial E^t}{\partial \xi^t_i}\cdot s^t_j
\\
\frac{\partial E^t}{\partial b^O_{i}} &=
\frac{\partial E^t}{\partial \xi^t_i}\cdot\frac{\partial \xi^t_i}{\partial b^O_{i}}
=\frac{\partial E^t}{\partial \xi^t_i}\cdot 1
\end{split}
\qquad i=1,\cdots,m\quad j=1,\cdots,n
```

向量化计算梯度

```math
\frac{\partial E^t}{\partial \boldsymbol{b^O}} =
\begin{bmatrix}
\frac{\partial E^t}{\partial \xi^t_1}\\\frac{\partial E^t}{\partial \xi^t_2}\\\vdots\\\frac{\partial E^t}{\partial \xi^t_m}
\end{bmatrix},
\qquad
\frac{\partial E^t}{\partial V} =
\begin{bmatrix}
\frac{\partial E^t}{\partial \xi^t_1}\\\frac{\partial E^t}{\partial \xi^t_2}\\\vdots\\\frac{\partial E^t}{\partial \xi^t_m}
\end{bmatrix}
\begin{bmatrix}
s^t_1&s^t_2&\cdots&s^t_n
\end{bmatrix}=
\begin{bmatrix}
\frac{\partial E^t}{\partial \xi^t_1} s^t_1 & \frac{\partial E^t}{\partial \xi^t_1} s^t_2 & \cdots & \frac{\partial E^t}{\partial \xi^t_1} s^t_n \\
\frac{\partial E^t}{\partial \xi^t_2} s^t_1 & \frac{\partial E^t}{\partial \xi^t_2} s^t_2 & \cdots & \frac{\partial E^t}{\partial \xi^t_2} s^t_n \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial E^t}{\partial \xi^t_m} s^t_1 & \frac{\partial E^t}{\partial \xi^t_m} s^t_2 & \cdots & \frac{\partial E^t}{\partial \xi^t_m} s^t_n \\
\end{bmatrix}
```

### 输入层到循环层

记循环层 $t$ 时刻的输入向量为 $\boldsymbol{\eta}^t$

```math
\begin{bmatrix}
s^t_1\\s^t_2\\\vdots\\s^t_n
\end{bmatrix}=
f\left(
\begin{bmatrix}
\eta^t_1 \\ \eta^t_2 \\ \vdots \\ \eta^t_n
\end{bmatrix}
\right),
\qquad
\begin{bmatrix}
\eta^t_1 \\ \eta^t_2 \\ \vdots \\ \eta^t_n
\end{bmatrix}=
\begin{bmatrix}
u_{11}&u_{12}&\cdots&u_{1m}\\
u_{21}&u_{22}&\cdots&u_{2m}\\
\vdots&\vdots&\ddots&\vdots\\
u_{n1}&u_{n2}&\cdots&u_{nm}\\
\end{bmatrix}
\begin{bmatrix}
x^t_1\\x^t_2\\\vdots\\x^t_m
\end{bmatrix}
+
\begin{bmatrix}
w_{11}&w_{12}&\cdots&w_{1n}\\
w_{21}&w_{22}&\cdots&w_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
w_{n1}&w_{n2}&\cdots&w_{nn}\\
\end{bmatrix}
\begin{bmatrix}
s^{t-1}_1\\s^{t-1}_2\\\vdots\\s^{t-1}_n
\end{bmatrix}
+
\begin{bmatrix}
b^R_1\\b^R_2\\\vdots\\b^R_n
\end{bmatrix}
```

关于矩阵U的偏导

由上面的记号, $t$ 时刻循环层的输入为$\boldsymbol{\eta}^t$, $\boldsymbol{\eta}^t$ 是网络在 $t$ 时刻的输入 $\boldsymbol{x}^t$ 和 上一时刻的状态 $\boldsymbol{s}^{t-1}$ 的线性变换

```math
\begin{gather}
\boldsymbol{\eta}^t = U\boldsymbol{x}^t + W\boldsymbol{s}^{t-1}+\boldsymbol{b}^R\\
\boldsymbol{s}^{t-1} = f(\boldsymbol{\eta}^{t-1})
\end{gather}
```

下面的公式推导出一个 $\partial E^t/\partial U$ 关于时间的递推式, 我们记 $\frac{\partial E^t}{\partial U}(t)$ 为 $t$ 时刻网络输出的误差 $E$ 关于

```math
\begin{split}
\frac{\partial E^t}{\partial U}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial U}
\\
(\boldsymbol{\eta}^t = U\boldsymbol{x}^t + W\boldsymbol{s}^{t-1}+\boldsymbol{b}^R)\rightarrow
% 第二个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
% 第二个等号括号中的内容
\left(
\frac{\partial U\boldsymbol{x}^t}{\partial U} +
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial U}
\right)
\\
% 第三个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
% 第三个等号括号中的内容
\left(
\frac{\partial U\boldsymbol{x}^t}{\partial U} +
W\frac{\partial \boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial U}
\right)
\\
% 第四个等号
将\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}乘进括号中去\rightarrow
&=
% 第四个等号加号左边的内容
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U} +
% 第四个等号加号右边的内容
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial U}
\\
\left(\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}=
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}\right)\rightarrow
% 第五个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U} +
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial U}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U} +
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial U}
\end{split}
```

由这个递推式可以得到

```math
\begin{split}
\frac{\partial E^t}{\partial U}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial U}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial U}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial U\boldsymbol{x}^{t-1}}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial \boldsymbol{\eta}^{t-2}}{\partial U}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial U\boldsymbol{x}^t}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial U\boldsymbol{x}^{t-1}}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial U\boldsymbol{x}^{t-2}}{\partial U}
+
\cdots
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{2}}
\frac{\partial U\boldsymbol{x}^{2}}{\partial U}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{1}}
\frac{\partial U\boldsymbol{x}^{1}}{\partial U}
\end{split}
```

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}\frac{\partial U\boldsymbol{x}^k}{\partial U}$

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}$

```math
\begin{split}
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
&=
\frac{\partial E^t}{\partial \boldsymbol{\xi}^t}
\frac{\partial \boldsymbol{\xi}^t}{\partial \boldsymbol{s}^t}
\frac{\partial \boldsymbol{s}^t}{\partial \boldsymbol{\eta}^t}
\\
&=
% 第二行的行向量
\begin{bmatrix}
\frac{\partial E^t}{\partial \xi^t_1}&\frac{\partial E^t}{\partial \xi^t_2}&\cdots&\frac{\partial E^t}{\partial \xi^t_m}
\end{bmatrix}
% 第二行的第一个矩阵
\begin{bmatrix}
\frac{\partial \xi^t_1}{\partial s^t_1} & \frac{\partial \xi^t_1}{\partial s^t_2} & \cdots & \frac{\partial \xi^t_1}{\partial s^t_n} \\
\frac{\partial \xi^t_2}{\partial s^t_1} & \frac{\partial \xi^t_2}{\partial s^t_2} & \cdots & \frac{\partial \xi^t_2}{\partial s^t_n} \\
\vdots&\vdots&\ddots&\vdots\\
\frac{\partial \xi^t_m}{\partial s^t_1} & \frac{\partial \xi^t_m}{\partial s^t_2} & \cdots & \frac{\partial \xi^t_m}{\partial s^t_n} \\
\end{bmatrix}
% 第二行的第二个矩阵
\begin{bmatrix}
\frac{\partial s^t_1}{\partial \eta^t_1} & \frac{\partial s^t_1}{\partial s^t_2} & \cdots & \frac{\partial s^t_1}{\partial s^t_n} \\
\frac{\partial s^t_2}{\partial \eta^t_1} & \frac{\partial s^t_2}{\partial \eta^t_2} & \cdots & \frac{\partial s^t_2}{\partial \eta^t_n} \\
\vdots&\vdots&\ddots&\vdots\\
\frac{\partial s^t_n}{\partial \eta^t_1} & \frac{\partial s^t_n}{\partial \eta^t_2} & \cdots & \frac{\partial s^t_n}{\partial \eta^t_n} \\
\end{bmatrix}
\\
&=
% 第三行的行向量
\begin{bmatrix}
\frac{\partial E^t}{\partial \xi^t_1}&\frac{\partial E^t}{\partial \xi^t_2}&\cdots&\frac{\partial E^t}{\partial \xi^t_m}
\end{bmatrix}
% 第三行的V矩阵
\begin{bmatrix}
v_{11}&v_{12}&\cdots&v_{1n}\\
v_{21}&v_{22}&\cdots&v_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
v_{m1}&v_{m2}&\cdots&v_{mn}\\
\end{bmatrix}
% 第三行的对角矩阵
\begin{bmatrix}
\frac{\partial s^t_1}{\partial \eta^t_1} & 0 & \cdots & 0 \\
0 & \frac{\partial s^t_2}{\partial \eta^t_2} & \cdots & 0 \\
\vdots&\vdots&\ddots&\vdots\\
0 & 0 & \cdots & \frac{\partial s^t_n}{\partial \eta^t_n} \\
\end{bmatrix}
\\
&=
\left[
\frac{\partial s^t_1}{\partial \eta^t_1}
\sum_{i=1}^m(\frac{\partial E^t}{\partial \xi^t_i}v_{i1})
,\quad
\frac{\partial s^t_2}{\partial \eta^t_2}
\sum_{i=1}^m(\frac{\partial E^t}{\partial \xi^t_i}v_{i2})
,\quad
\cdots
,\quad
\frac{\partial s^t_n}{\partial \eta^t_n}
\sum_{i=1}^m(\frac{\partial E^t}{\partial \xi^t_i}v_{in})
\right]
\\
\text{记为}&=
\begin{bmatrix}
\delta^{tt}_1&\delta^{tt}_2&\cdots&\delta^{tt}_n
\end{bmatrix}
\end{split}
```

$\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}$ 的结果记为 $\boldsymbol{\delta^{tt}}$,   称为循环层 $t$ 时刻(第二个 $t$)的输入的**误差项** (**网络 $t$ 时刻输出的误差**关于**循环层 $t$ 时刻输入**的偏导数)

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}$

```math
\begin{split}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}
&=
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}=
W\frac{\partial \boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}=
W
% 第一个矩阵
\begin{bmatrix}
\frac{\partial s^{t-1}_{1}}{\partial \eta^{t-1}_{1}}&
\frac{\partial s^{t-1}_{1}}{\partial \eta^{t-1}_{2}}&
\cdots&
\frac{\partial s^{t-1}_{1}}{\partial \eta^{t-1}_{n}}
\\
\frac{\partial s^{t-1}_{2}}{\partial \eta^{t-1}_{1}}&
\frac{\partial s^{t-1}_{2}}{\partial \eta^{t-1}_{2}}&
\cdots&
\frac{\partial s^{t-1}_{2}}{\partial \eta^{t-1}_{n}}
\\
\vdots&\vdots&\ddots&\vdots\\
\frac{\partial s^{t-1}_{n}}{\partial \eta^{t-1}_{1}}&
\frac{\partial s^{t-1}_{n}}{\partial \eta^{t-1}_{2}}&
\cdots&
\frac{\partial s^{t-1}_{n}}{\partial \eta^{t-1}_{n}}
\end{bmatrix}=
% 第二个矩阵
W\begin{bmatrix}
\frac{\partial s^{t-1}_{1}}{\partial \eta^{t-1}_{1}}&0&\cdots&0
\\
0&\frac{\partial s^{t-1}_{2}}{\partial \eta^{t-1}_{2}}&\cdots&0
\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\frac{\partial s^{t-1}_{n}}{\partial \eta^{t-1}_{n}}
\end{bmatrix}
\\
&=
W\begin{bmatrix}
f'(\eta^{t-1}_{1})&0&\cdots&0
\\
0&f'(\eta^{t-1}_{2})&\cdots&0
\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&f'(\eta^{t-1}_{n})
\end{bmatrix}
\end{split}
```

```math
\begin{split}
\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}
&=
\frac{\partial E^t}{\partial \boldsymbol{\xi}^t}
\frac{\partial \boldsymbol{\xi}^t}{\partial \boldsymbol{s}^t}
\frac{\partial \boldsymbol{s}^t}{\partial \boldsymbol{\eta}^t}
\left(
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}
\cdots
\frac{\partial \boldsymbol{\eta}^{k+1}}{\partial \boldsymbol{\eta}^{k}}
\right)\\
&=
\begin{bmatrix}
\delta^{tt}_1&\delta^{tt}_2&\cdots&\delta^{tt}_n
\end{bmatrix}
% 连乘
\prod_{i=(t-1)}^{k}
\left(
W\begin{bmatrix}
f'(\eta^{i}_{1})&\cdots&0\\
\vdots&\ddots&\vdots\\
0&\cdots&f'(\eta^{i}_{n})
\end{bmatrix}
\right)\\
记为&=
\begin{bmatrix}
\delta^{tk}_1&\delta^{tk}_2&\cdots&\delta^{tk}_n
\end{bmatrix}
\qquad (t \ge k \ge 1)
\end{split}
```

$\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}$ 的结果记为 $\boldsymbol{\delta^{tk}}$, 称为循环层 $k$ 时刻输入的误差项 (**网络 $t$ 时刻输出的误差**关于**循环层 $k$ 时刻输入**的偏导数)

实际计算中我们会一步一步地计算 $\boldsymbol{\delta}^{tt},\boldsymbol{\delta}^{t(t-1)},\cdots,\boldsymbol{\delta}^{t1}$, 而不是使用连乘运算

```math
\begin{split}
% 第一行
\begin{bmatrix}
\delta^{t(t-1)}_1&\delta^{t(t-1)}_2&\cdots&\delta^{t(t-1)}_n
\end{bmatrix}
&=
\begin{bmatrix}
\delta^{tk}_1&\delta^{tk}_2&\cdots&\delta^{tk}_n
\end{bmatrix}
W\begin{bmatrix}
f'(\eta^{t-1}_{1})&\cdots&0
\\
\vdots&\ddots&\vdots\\
0&\cdots&f'(\eta^{t-1}_{n})
\end{bmatrix}
% 第二行
\\
\begin{bmatrix}
\delta^{t(t-2)}_1&\delta^{t(t-2)}_2&\cdots&\delta^{t(t-2)}_n
\end{bmatrix}
&=
\begin{bmatrix}
\delta^{t(t-1)}_1&\delta^{t(t-1)}_2&\cdots&\delta^{t(t-1)}_n
\end{bmatrix}
W\begin{bmatrix}
f'(\eta^{t-2}_{1})&\cdots&0
\\
\vdots&\ddots&\vdots\\
0&\cdots&f'(\eta^{t-2}_{n})
\end{bmatrix}
\\
&\vdots
\\
% 第三行
\begin{bmatrix}
\delta^{t1}_1&\delta^{t1}_2&\cdots&\delta^{t1}_n
\end{bmatrix}
&=
\begin{bmatrix}
\delta^{t(2)}_1&\delta^{t(2)}_2&\cdots&\delta^{t(2)}_n
\end{bmatrix}
W\begin{bmatrix}
f'(\eta^{1}_{1})&\cdots&0
\\
\vdots&\ddots&\vdots\\
0&\cdots&f'(\eta^{1}_{n})
\end{bmatrix}
\end{split}
```

计算 $\frac{\partial U\boldsymbol{x}^k}{\partial U}$

```math
\frac{\partial U\boldsymbol{x}^k}{\partial U}=
% 第一个大矩阵
\begin{bmatrix}
\left(\begin{smallmatrix}
\frac{\partial \eta^k_1}{\partial u_{11}} & \cdots & \frac{\partial \eta^k_1}{\partial u_{1m}}\\
\vdots & \ddots & \vdots \\
\frac{\partial \eta^k_1}{\partial u_{n1}} & \cdots & \frac{\partial \eta^k_1}{\partial u_{nm}}\\
\end{smallmatrix}\right)
\\ \vdots \\
\left(\begin{smallmatrix}
\frac{\partial \eta^k_i}{\partial u_{11}} & \cdots & \frac{\partial \eta^k_i}{\partial u_{1m}}\\
\vdots & \ddots & \vdots \\
\frac{\partial \eta^k_i}{\partial u_{n1}} & \cdots & \frac{\partial \eta^k_i}{\partial u_{nm}}\\
\end{smallmatrix}\right)
\\ \vdots \\
\left(\begin{smallmatrix}
\frac{\partial \eta^k_n}{\partial u_{11}} & \cdots & \frac{\partial \eta^k_n}{\partial u_{1m}}\\
\vdots & \ddots & \vdots \\
\frac{\partial \eta^k_n}{\partial u_{n1}} & \cdots & \frac{\partial \eta^k_n}{\partial u_{nm}}\\
\end{smallmatrix}\right)
\end{bmatrix}=
% 第二个大矩阵
\begin{bmatrix}
\left(\begin{smallmatrix}
x^k_1 & x^k_2 &\cdots & x^k_m\\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0\\
\end{smallmatrix}\right)
\\ \vdots \\
\left(\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
\vdots & \vdots &   & \vdots \\
x^k_1 & x^k_2 &\cdots & x^k_m\\
\vdots & \vdots &   & \vdots \\
0 & 0 & \cdots & 0\\
\end{smallmatrix}\right)
\begin{smallmatrix}
1\\\vdots\\i\\\vdots\\n
\end{smallmatrix}
\\ \vdots \\
\left(\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0\\
x^k_1 & x^k_2 &\cdots & x^k_m\\
\end{smallmatrix}\right)
\end{bmatrix}
\qquad
(t \ge k \ge 1)
```

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}\frac{\partial U\boldsymbol{x}^k}{\partial U}$

```math
\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}
\cdot
\frac{\partial U\boldsymbol{x}^k}{\partial U} =
\begin{bmatrix}
\delta^{tk}_1&\delta^{tk}_2&\cdots&\delta^{tk}_n
\end{bmatrix}
% 大矩阵
\begin{bmatrix}
\left(\begin{smallmatrix}
x^k_1 & x^k_2 &\cdots & x^k_m\\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0\\
\end{smallmatrix}\right)
\\ \vdots \\
\left(\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
\vdots & \vdots &   & \vdots \\
x^k_1 & x^k_2 &\cdots & x^k_m\\
\vdots & \vdots &   & \vdots \\
0 & 0 & \cdots & 0\\
\end{smallmatrix}\right)
\begin{smallmatrix}
1\\\vdots\\i\\\vdots\\n
\end{smallmatrix}
\\ \vdots \\
\left(\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0\\
x^k_1 & x^k_2 &\cdots & x^k_m\\
\end{smallmatrix}\right)
\end{bmatrix}
% 第二个等号
=\begin{bmatrix}
\delta^{tk}_1 \\ \delta^{tk}_2 \\ \vdots \\ \delta^{tk}_n
\end{bmatrix}
\begin{bmatrix}
x^k_1 &  x^k_2 & \cdots & x^k_m
\end{bmatrix}
\qquad
(t \ge k \ge 1)
```

最后结果U的梯度

```math
\frac{\partial E^t}{\partial U}
% 第一个等号
=\sum_{k=1}^t
\left(
\begin{bmatrix}
\delta^{tk}_1 \\ \delta^{tk}_2 \\ \vdots \\ \delta^{tk}_n
\end{bmatrix}
\begin{bmatrix}
x^k_1 &  x^k_2 & \cdots & x^k_m
\end{bmatrix}
\right)
```

### 关于矩阵W的偏导

```math
\begin{split}
\frac{\partial E^t}{\partial W}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial W}
\\
(\boldsymbol{\eta}^t = U\boldsymbol{x}^t + W\boldsymbol{s}^{t-1}+\boldsymbol{b}^R)\rightarrow
% 第二个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
% 第二个等号括号中的内容
\left(
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial W}
\right)
\\
(莱布尼茨法则)\rightarrow
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
% 第二个等号括号中的内容
\left(
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
W\frac{\partial \boldsymbol{s}^{t-1}}{\partial W}
\right)
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
W\frac{\partial \boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial W}
\\
\left(\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}=
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}\right)\rightarrow
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial W}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial W}
\end{split}
```

```math
\begin{split}
\frac{\partial E^t}{\partial W}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial W}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial W}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-2}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial \boldsymbol{\eta}^{t-2}}{\partial W}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-2}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial W}{\partial W}\boldsymbol{s}^{t-3}
+
\cdots
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{2}}
\frac{\partial W}{\partial W}\boldsymbol{s}^{1}
+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{1}}
\frac{\partial W}{\partial W}\boldsymbol{s}^{0}
\end{split}
```

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}\frac{\partial W}{\partial W}\boldsymbol{s}^{k-1}$

计算 $\frac{\partial W}{\partial W}$

```math
\begin{split}
\frac{\partial W}{\partial W}&=
\frac
{\partial
\left(\begin{smallmatrix}
w_{11}&\cdots&w_{n1}\\
\vdots&\ddots&\vdots\\
w_{n1}&\cdots&w_{nn}
\end{smallmatrix}\right)}
{\partial
\left(\begin{smallmatrix}
w_{11}&\cdots&w_{n1}\\
\vdots&\ddots&\vdots\\
w_{n1}&\cdots&w_{nn}
\end{smallmatrix}\right)}
\\
&=
% 大矩阵
\begin{bmatrix}
\left(
\begin{smallmatrix}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
&
\cdots
&
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 1 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
\\
\vdots&\ddots&\vdots\\
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
1 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
&
\cdots
&
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 1 \\
\end{smallmatrix}
\right)
\end{bmatrix}
\end{split}
```

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}\frac{\partial W}{\partial W}\boldsymbol{s}^{k-1}$

```math
\begin{split}
\frac{\partial E^t}{\partial \boldsymbol{\eta}^k}\frac{\partial W}{\partial W}\boldsymbol{s}^{k-1}
&=
\begin{bmatrix}
\delta^{tk}_1&\delta^{tk}_2&\cdots&\delta^{tk}_n
\end{bmatrix}
% 大矩阵
\begin{bmatrix}
\left(
\begin{smallmatrix}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
&
\cdots
&
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 1 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
\\
\vdots&\ddots&\vdots\\
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
1 & 0 & \cdots & 0 \\
\end{smallmatrix}
\right)
&
\cdots
&
\left(
\begin{smallmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots& \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 1 \\
\end{smallmatrix}
\right)
\end{bmatrix}
\begin{bmatrix}
s^{k-1}_1\\s^{k-1}_2\\\vdots\\s^{k-1}_n
\end{bmatrix}
\\
&=
\begin{bmatrix}
\delta^{tk}_1\\\delta^{tk}_2\\\vdots\\\delta^{tk}_n
\end{bmatrix}
\begin{bmatrix}
s^{k-1}_1&s^{k-1}_2&\cdots&s^{k-1}_n
\end{bmatrix}
\qquad
(t \geq k \geq 1)
\end{split}
```

最后结果W的梯度

```math
\frac{\partial E^t}{\partial W}
% 第一个等号
=\sum_{k=1}^t
\left(
\begin{bmatrix}
\delta^{tk}_1\\\delta^{tk}_2\\\vdots\\\delta^{tk}_n
\end{bmatrix}
\begin{bmatrix}
s^{k-1}_1&s^{k-1}_2&\cdots&s^{k-1}_n
\end{bmatrix}
\right)
```

关于偏置项 $\boldsymbol{b}^R$ 的偏导

```math
\begin{split}
\frac{\partial E^t}{\partial \boldsymbol{b}^R}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{b}^R}
\\
(\boldsymbol{\eta}^t = U\boldsymbol{x}^t + W\boldsymbol{s}^{t-1}+\boldsymbol{b}^R)\rightarrow
% 第二个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
% 第二个等号括号中的内容
\left(
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{b}^R}
\right)
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial \boldsymbol{b}^R}
\\
\left(\frac{\partial W\boldsymbol{s}^{t-1}}{\partial \boldsymbol{\eta}^{t-1}}=
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}\right)\rightarrow
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial \boldsymbol{b}^R}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial \boldsymbol{b}^R}
\end{split}
```

```math
\begin{split}
\frac{\partial E^t}{\partial \boldsymbol{b}^R}
% 第一个等号
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{\eta}^t}{\partial \boldsymbol{b}^R}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{\eta}^{t-1}}{\partial \boldsymbol{b}^R}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial \boldsymbol{\eta}^{t-2}}{\partial \boldsymbol{b}^R}
\\
&=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^t}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-1}}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{t-2}}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}+
\cdots+
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{1}}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}
\end{split}
```

计算 $\frac{\partial E^t}{\partial \boldsymbol{\eta}^{k}}\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}$

```math
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{k}}
\frac{\partial \boldsymbol{b}^R}{\partial \boldsymbol{b}^R}=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{k}}
\cdot
I_{nn}=
\frac{\partial E^t}{\partial \boldsymbol{\eta}^{k}}=
\begin{bmatrix}
\delta^{tk}_1\\\delta^{tk}_2\\\vdots\\\delta^{tk}_n
\end{bmatrix}
```

最后结果 $\boldsymbol{b}^R$ 的梯度

```math
\frac{\partial E^t}{\partial \boldsymbol{b}^R}=
\sum_{k=1}^t
\left(
\begin{bmatrix}
\delta^{tk}_1\\\delta^{tk}_2\\\vdots\\\delta^{tk}_n
\end{bmatrix}
\right)
```
