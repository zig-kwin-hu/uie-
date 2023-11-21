# UIE++

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.7)
- Pytorch (^2.0.0)
- Transformers (4.30.2)
- DeepSpeed (0.9.5)

You can install the required libraries by running 

```
bash setup.sh
```


## Data

Our models are trained and evaluated on **IE INSTRUCTIONS**. 
You can download the data from [Baidu NetDisk](https://pan.baidu.com/s/1R0KqeyjPHrsGcPqsbsh1XA?from=init&pwd=ybkt) or [Google Drive](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view?usp=share_link).


## Training

A sample script for training the UIE++ model in our paper can be found at [`scripts/train_uie.sh`](scripts/train_uie.sh). You can run it as follows:

```
bash ./scripts/train_uie.sh
```




