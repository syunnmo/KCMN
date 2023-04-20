# KCMN
The manuscript has not been published.
This project is the Pytorch implementation for KCMN. 

# Dataset
In 'data' folder, we have provided the processed datasets. 
If you would like to access the raw datasets, the raw datasets are placed in the following links:
* ASSISTments2009  : [address](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
* ASSISTments2012 : [address](https://sites.google.com/site/assistmentsdata/2012-13-school-data-with-affect)
* slepemapy.cz. : [address](https://www.fi.muni.cz/adaptivelearning/?a=data)
* EdNet : [address](https://github.com/riiid/ednet)

# Setups

__Service__: 
* Linux operation system

__Environment__:

* python 3+
* sklearn  0.21.3
* tqdm 4.54.1
* torch 1.7.0
* numpy 1.19.2

# Running FGKT
Here is a example for using FGKT model (on EdNet):  
```
  python main.py --dataset ednet  
```

Explanation of parameters:  
* gpu: Specify the GPU to be used, e.g '0,1,2,3'. If CPU is used then fill in -1.
* patience: Maximum number of times if validation loss does not decrease.
* lr: Learning rate
* num_heads: Number of head attentions.
* mode: Selection of integration function.
* exercise_embed_dim: Number of exercise embedding dimensions.
* batch_size: Number of batch size.
