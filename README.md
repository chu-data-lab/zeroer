# ZeroER
Implementation for the paper [ZeroER: Entity Resolution using Zero Labeled Examples.](https://arxiv.org/abs/1908.06049)

## setup enviroment
    conda env create -f environment.yml
    conda activate ZeroER

## how to use
example usage:

`python zeroer.py fodors_zagats --run_transitivity`

If you want to incorporate the transitivity constraint, use arg `--run_transitivity`: 

`python zeroer.py fodors_zagats --run_transitivity`

*Note this will generate features for self-join of the two tables (LxL and RxR) when arg `--LR_dup_free` is not present, which can take some time.

If you know that your left table and right table are duplicate free, you can incorporate this information by using arg `--run_transitivity --LR_dup_free`:

`python zeroer.py fodors_zagats --run_transitivity --LR_dup_free`