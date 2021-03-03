

## Packages 
tensorflow 
sklearn
numpy
multiprocessing 
tqdm 

###First cretae out folder in cafem.py directory

```
mkdir out
```




### Train  with Order-1 & Order-2 operators on 20 datasets
```
python cafem.py --multiprocessing 6 --out_dir   out/ml --num_episodes 5

```

###Training for testing the code out

```
python cafem.py --out_dir   out/ml --cuda 0 --num_epochs 1 --meta_batch_size 1 --num_episodes 1
```


### Train  on 100 datasets and apply on each of other 20 datasets

```

python single_afem.py --load_weight   out/ml/model/model_5.ckpt --dataset 1049 --out_dir   out/o2_5_1049 --num_epochs 50 --buffer_size 1000 --num_episodes 1



```

###Training for testing the code out

```
python single_afem.py --load_weight   out/ml/cafem/model_1.ckpt --dataset 1049 --out_dir   out/o2_5_1049 --num_epochs 1 --buffer_size 1000 --num_episodes 1
```
