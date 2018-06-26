# Enhancing Sentence Embedding with Generalized Pooling
Source code for "Enhancing Sentence Embedding with Generalized Pooling" based on Theano.
If you use this code as part of any published research, please acknowledge the following paper.

**"Enhancing Sentence Embedding with Generalized Pooling"**
Qian Chen, Zhen-Hua Ling, Xiaodan Zhu. _COLING (2018)_ 

```
@InProceedings{Chen-Qian:2018:COLING,
  author    = {Chen, Qian and Ling, Zhen-Hua and Zhu, Xiaodan},
  title     = {Enhancing Sentence Embedding with Generalized Pooling},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
  month     = {August},
  year      = {2018},
  address   = {Santa Fe, USA},
  publisher = {ACL}
}
```
Homepage of the Qian Chen, http://home.ustc.edu.cn/~cq1231/

## Dependencies
To run it perfectly, you will need (recommend using Ananconda to set up environment):
* Python 2.7.13
* Theano 0.9.0

## Running the Script
1. Download and preprocess 
```
cd data
bash fetch_and_preprocess.sh
```

2. Train and test model
```
cd scripts/generalized-pooling/
bash train.sh
```

The result is in `scripts/generalized-pooling/log.txt` file.

3. Analysis the result for dev/test set (optional)
```
bash test.sh
```
