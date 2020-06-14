## CHANGE LOG

**v1.2:**

- add standard deviation of DSC in `coarse2fine_testing.py`
- Our codebase is also compatible with PyTorch 0.4.1.

**v1.1:**

- Thank *Qihang Yu* for finding the bug which affects performance when `batch > 1` in `model.py` and having fixed it elegantly.
- remove the redundant `clone()` in `model.py`
  
**v1.0:**

- make `get_parameters` in `model.py` more robust
  
**v0.5:**

- add **`logs/`** which contains training logs and testing results in `FOLD #0`. please see section 5
- add **RSTN pre-trained models** in section 5
- add **`oracle_testing.py` & `oracle_fusion.py`** to evaluate fine models. please see 4.6 & 4.7

**v0.4:**

- we introduce **`epoch` hyperparameter** to replace `max_iterations` because the size of datasets varies.
    - Epoch dict {2, 6, 8} for (S, I, J) is intended for NIH dataset. You may modify it according to your dataset.
- **Add `training_parallel.py` to support multi-GPU training:**
    - please see 4.3.4 section for details.
- Simplify the bilinear weight initialization in ConvTranspose layer (issue [#1](https://github.com/twni2016/OrganSegRSTN_PyTorch/issues/1))
- **Add `coarse_fusion.py`**
- `training.py` & `training_parallel.py`: print **coarse/fine/average** loss, giving more information of training loss
  - Thank Angtian Wang and Yingwei Li for finding bugs on multi-GPU training.

**v0.3:** no big improvements.

**v0.2:**

-  `utils.py`: two faster functions `post_processing` and `DSC_computation` are re-implemented in C for python3.6
   -  give instructions in section 4.8.3 on how to compile ` fast_functions.i` to get `_fast_functions.so` for different version of python like 3.5.
-  `training.py` : now trains by *iterations* instead of *epoches*, and learning rate will decay in `J` mode every 10k iterations.
   -  performance of current version is **84.3%** in NIH dataset, which is *slightly lower* than **84.4-84.6%** in CAFFE implementation.

**v0.1:** init version.

## Differences from [OrganSegRSTN](https://github.com/198808xc/OrganSegRSTN)

Improvements:

- We merge `indiv_training.py`, `joint_training.py` into `training.py`
- We merge all `*.prototxt` to `model.py`
- Our code runs almost **twice faster** than original one in CAFFE. 
- The *minimum* of DSC in test cases is **a little higher** (63.4%) than original minimum (62.8%).

Performance: in NIH Pancreas Dataset, average DSC is **a little poorer** (84.25% - 84.45%) than original one in CAFFE (84.4% - 84.6%). 
