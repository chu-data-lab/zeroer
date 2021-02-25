# ZeroER
Implementation for the paper [ZeroER: Entity Resolution using Zero Labeled Examples.](https://arxiv.org/abs/1908.06049)

## setup enviroment
    conda env create -f environment.yml
## how to use
example usage:
   
`python zeroer.py fodors_zagats --run_transitivity=True --LR_dup_free=False`

*If you want to utilize the transitivity constraint, set `run_transitivity=True`. Note this will generate features for self-join of the two tables (LxL and RxR) when `LR_dup_free=False`, which can take some time.

*If you know that your left table and right table are duplicate free, you can use this information by setting `run_transitivity=True` and `LR_dup_free=True`.
