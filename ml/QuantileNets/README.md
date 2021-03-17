# Quantile Nets
This folder contains example code of how to use the quantile networks. The general principal behind them is to learn the quantile function of a set of data, which is the inverse cumulative distribtuion funcion. Using this quantile function, we can draw random samples from a distribution, as uniform samples of the range (0,1) when input to the quantile function give random samples from the full distribution. x,y,z

When tasked with learning multiple dimensions x, y, z, given data D, the network learns p(x|D), p(y|x,D), p(z|x,y,D), as p(x,y,z|D) =p(x|D)p(y|x,D)p(z|x,y,D). This requires the network to make n calls to sample from n dimensional data.

In quantileNetwork.py are three important pieces of code, the QuantileNetwork Model, a makeDataset function, and a callNet function. The model's function is clear enough, but makeDataset is used to turn a normal ml dataset into a quantile one, and callNet is used to make actual predictions.

The file trainQuantileNet.py contains code to actually train a network, and useQuantileNet.py contains code to make predictions from that network. Note that is demonstrates an issue with the networks, when given insufficient data and a rapidly varying function, it will assume the variance is noise. This may not actual be a failing, but it is a fact of the model.

For an interesting paper dealing with a version of quantile networks for generating distributions, see  https://arxiv.org/pdf/1806.05575.pdf
