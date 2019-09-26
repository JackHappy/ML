
# 梯度下降算法
$$\theta= \theta - \alpha { \partial J( \theta) \over \partial ( \theta )}$$

在机器学习的算法中，最小化损失函数，可以通过梯度下降法一步步迭代求解  
迭代求取解有可能是局部最优解，可以通过多次不同初始值迭代来接近全局最优解  
损失函数是凸函数，则获得是全局最优解


### 线性回归

拟合函数：$$h_\theta(x)=\sum _{i=0}^n{\theta_i x_i}$$
损失函数：$$J(\theta_0,\theta_1,...\theta_n)={1 \over 2m}\sum _{j=0}^m{(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)^2}$$
梯度方向：$${ \partial J( \theta_0,\theta_1,..\theta_n,) \over \partial ( \theta_i )} = {1 \over m}\sum _{j=1}^m{(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)x_i^{(j)}}$$
迭代求解：$$\theta_i = \theta_i - \alpha{1 \over m}\sum _{j=1}^m{(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)x_i^{(j)}}$$

矩阵描述：$$J(\theta)={1 \over 2}(X\theta-Y)^T(X\theta-Y)$$

$${\partial J(\theta) \over \partial \theta} = X^T(X\theta -Y)$$


相关矩阵公式：$${\partial(XX^T) \over \partial X} = 2X$$

$${\partial (X\theta) \over \theta} = X^T$$

### 批量梯度下降法（Batch Gradient Descent） BGD
$$\theta_i = \theta_i - \alpha \sum _{j=1}^m{(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)x_i^{(j)}}$$
遍历所有样本后才会更新$\theta$

### 随机梯度下降法（Stochastic Gradient Descent） SGD
$$\theta_i = \theta_i - \alpha {(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)x_i^{(j)}}$$
遍历每个样本都会更新$\theta$

### 小批量梯度下降法（Min-batch Gradient Descent） MGD
$$\theta_i = \theta_i - \alpha \sum _{j=t}^{t+x-1}{(h_\theta(x_0^{(j)},x_1^{(j)},...x_n^{(j)})-y_j)x_i^{(j)}}$$
遍历x (1<x<m) 样本后才会更新$\theta$


收敛速度：MGD>SGD>BGD

# 最小二乘

最小二乘为解析求解，在导数为0的时候求取$\theta$
$${\partial J(\theta) \over \partial \theta} = X^T(X\theta -Y) = 0$$

=> $$\theta = (X^TX)^{-1}X^TY$$


### 最小二乘与梯度下降比较：
1，最小二乘为解析求解，在特征维数或样书数量较小的情况下，求解高效简洁，但只适用于线性拟合，当特征位数大于10000，则建议使用梯度下降迭代求解或者PCA后使用最小二乘  
2，梯度下降要选择迭代步长，而且有可能获得是局部最优解，需要初始化不同值多次迭代求解来接近全局最优解  
3，牛顿法/拟牛顿法是用二阶的海森矩阵的逆矩阵或伪逆矩阵求解，牛顿法/拟牛顿法收敛更快，但每次迭代的时间比梯度下降法长  

# 线性回归

为防止线性回归模型过拟合，通常会在损失函数中添加$\theta$的L1正则或者L2正则
### Lasso回归 
线性回归的L1正则称为Lasso回归，Lasso回归使一些特征系数变小，一些绝对值较小的系数直接为0，增加模型的泛化能力 

$$J(\theta)={1 \over 2}(X\theta-Y)^T(X\theta-Y)+\alpha||\theta||_{1}$$

### Ridge回归
线性回归的L2正则成为Ridge回归
$$J(\theta)={1 \over 2}(X\theta-Y)^T(X\theta-Y)+\alpha{1 \over 2}||\theta||_{2}^2$$

Ridge回归在不抛弃任何一个特征的情况下，缩小了回归系数，使得模型相对而言比较的稳定，但和Lasso回归比，这会使得模型的特征留的特别多，模型解释性差  
优化求解：
梯度：$${\partial J(\theta) \over \partial \theta} = X^T(X\theta -Y) + \alpha \theta $$
梯度下降迭代求解：
$$\theta = (1- \alpha) \theta - X^T(X\theta -Y) $$
最小二乘解析求解：
$$\theta = (X^TX+\alpha E)^{-1}X^TY$$

### Lasso回归求解