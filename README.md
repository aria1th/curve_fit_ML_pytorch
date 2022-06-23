# curve_fit_ML_pytorch
Curve fitting for functions with pytorch machine learning

# Requirements
Numpy, Pytorch(Cuda if available), Matplotlib

# How to use
```
func = lambda x : np.cos(x) * x**2 - 3*x
a, b = fit(func, low = 0.2, high = 6, val_low = 0.1, val_high = 10, batch_size = 1024, layer_count = 4, epoch = 1000, activation = nn.ReLU, features = 100)
```

Will simply plot and show this:
![image](https://user-images.githubusercontent.com/35677394/175302869-1e4380f0-1a6e-4fe9-96b5-c4a3c47e772f.png)

low, high = Range for 'training set'
val_low, val_high = Range for 'test set'

layer_count, features, etc... - model structure

