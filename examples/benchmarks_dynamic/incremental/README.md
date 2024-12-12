# DoubleAdapt (KDD'23)

## Introduction
This is the official implementation of `DoubleAdapt`, an incremental learning framework for stock trend forecasting.

The paper has been accepted by KDD 2023, which is better to read in [[arXiv](https://arxiv.org/abs/2306.09862)].

To get rid of dependencies on qlib, please refer to our [API](https://github.com/SJTU-Quant/DoubleAdapt) repo. (This API repo is not well maintained and may have undiscovered bugs. We still recommend our qlib repo)

### Code Organization
The runner program is [./main.py](main.py).

The core implementation of the framework lies in [qlib/contrib/meta/incremental/](https://github.com/SJTU-Quant/qlib/blob/main/qlib/contrib/meta/incremental/).

The implementation of any forecast model lies in [qlib/contrib/model/](https://github.com/SJTU-Quant/qlib/blob/main/qlib/contrib/model/) (e.g., GRU is in [qlib/contrib/model/pytorch_gru.py](https://github.com/SJTU-Quant/qlib/blob/main/qlib/contrib/model/pytorch_gru.py)).

## IMPORTANT Suggestions before Deployment
Sorry for that our experimental settings followed prior works but are not optimal in practical usages. We provide some suggestions below to help you successfully customize DoubleAdapt for application.

### Combine incremental learning (IL) with rolling retraining (RR)
Though our paper consider RR as a comparison method against IL, they are orthogonal to each other. 
For more profits, a recommended setting is to retrain DoubleAdapt from scratch every month and, during the month, perform DoubleAdapt every 2~3 trading days.

### Re-devise the data adapter
We mainly experiment on a simple dataset Alpha360, and our proposed feature adaptation only involves 6$\times$6 affine transformation with a few parameters to learn. 
Since common practice in quantative investment is based on hundreds of factors (e.g., Alpha158), our fully connected layer is over-parameterized and achieves suboptimal performance. It would be better to design a new data adapter. Below are some more lightweight designs:
- Divide the factors into different groups and learn affine transformation within the same group, though ignoring interactions between factors of different groups.
- Or: Apply the same transformation on the embedding of each factor. Learning element-wise operations (e.g. normalizing flows) over all factor embeddings.

As for general multivariate time series forecasting, we empirically found that an channel-independent data adapter is desirable, which transforms the lookback/horizon window of each variable independently.

If you meet any question or issue, please let us know. We are glad to discuss with you.

### Grid search on learning rates during offline and online training
It is **necessary** to perform hyperparameter tuning for learning rates `lr_da`, `lr_ma` and `lr` (learning rate of the lower level). 
Note that the learning rates during online training could be different from those during offline training.

> Fill arg `--online_lr` to set different learning rates.
> Example: `--online_lr "{'lr': 0.0005, 'lr_da': 0.0001, 'lr_ma': 0.0005}"`

## Dataset
Following DDG-DA, we run experiments on the crowd-source version of qlib data which can be downloaded by
```bash
wget https://github.com/chenditc/investment_data/releases/download/2023-06-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
```
Arg `--data_dir crowd_data` and `--data_dir cn_data` for crowd-source data and Yahoo-source data, respectively.

Arg `--alpha 360` or `--alpha 158` for Alpha360 and Alpha 158, respectively. 
 
Note that we are to predict the stock trend **BUT NOT** the rank of stock trend, which is different from DDG-DA.

To this end, we use `CSZScoreNorm` in the learn_processors instead of `CSRankNorm`.

Pay attention to the arg `--rank_label False` (or `--rank_label True`) for the target label. 

As the current implementation is simple and may not suit rank labels, we recommend `--adapt_y False` when you have to set `--rank_label True`.  

## Scripts
```bash
# Naive incremental learning
python -u main.py run_all --forecast_model GRU --market csi300 --data_dir crowd_data --rank_label False --naive True
# DoubleAdapt
python -u main.py run_all --forecast_model GRU --market csi300 --data_dir crowd_data --rank_label False \ 
--num_head 8 --tau 10 --lr 0.001 --lr_da 0.01 --online_lr "{'lr': 0.001, 'lr_da': 0.0001, 'lr_ma': 0.001}"
```

### Carefully select `step` according to `horizon`
Arg `--horizon` decides the target label to be `Ref($close, -horizon-1}) / Ref($close, -1) - 1` in the China A-share market. 
Accordingly, there are always unknown ground-truth labels in the lasted `horizon` days of test data, and we can only use the rest for optimization of the meta-learners.
With a large `horizon` or a small `step`, the performance on the majority of the test data cannot be optimized, 
and the meta-learners may well be overfitted and shortsighted.
We provide an arg `--use_extra True` to take the nearest data as additional test data, while the improvement is often little.

It is recommended to let `step` be greater than `horizon` by at least 3 or 4, e.g., `--step 5 --horizon 1`.

> The current implementation does not support `step` $\le$ `horizon` (e.g., `--step 1 --horizon 1`) during online training.
> 
> As the offline training can be conducted as usual, you can freeze the meta-learners online, initialize a forecast model by the model adapter, and then incrementally update the forecast model throughout the online phase.

## Requirements

### Packages
On top of requirements in qlib, we use an additional package from [github.com/facebookresearch/higher](https://github.com/facebookresearch/higher)
```bash
conda install higher -c conda-forge
# pip install higher
```

### RAM

This implementation requires ~8GB RAM on CSI500 when the update interval `step` is set to 20 trading days.

If your RAM is limited, you can split the function `dump_data` into two functions that dump training data and test data, respectively. 
Then, free the storage of training data before testing. 

Moreover, in our implementation, we cast all slices of stock data in `pandas.DataFrame` to `torch.Tensor` during data preprocessing.
This trick largely reduces CPU occupation during training and testing while it results in duplicate storage.

You can also set `--preprocess_tensor False`, reducing RAM occupation to ~5GB (peak 8GB before training). 
Then, the data slices are created as virtual views of `pandas.DataFrame`, and the duplicates share the same memory address. 
Each batch will be cast as `torch.Tensor` when needed, requesting new memory of a tiny size.
However, `--preprocess_tensor False` can exhaust all cores of the CPU and the speed is lower consequently.

### GPU Memory
DoubleAdapt requires at most 10GB GPU memory when `step` is set to 20. 
The occupation will be smaller on CSI300 and on the default Yahoo data (which bears more missing values).

If your GPU is limited, try to set a smaller `step` (e.g., 5) which may take up ~2GB. And you can achieve higher performance.

> The reason why we set `step` to 20 rather than 5 is that 
RR and DDG-DA bear unaffordable time costs (e.g., 3 days for 10 runs) in experiments with `step` set to 5.   

## Cite
If you find this useful for your work, please consider citing it as follows:
```bash
@InProceedings{DoubleAdapt,
  author       = {Lifan Zhao and Shuming Kong and Yanyan Shen},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  title        = {{DoubleAdapt}: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting},
  year         = {2023},
  month        = {aug},
  publisher    = {{ACM}},
  doi          = {10.1145/3580305.3599315},
}
```
