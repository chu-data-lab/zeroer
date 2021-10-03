# ZeroER
Implementation for the paper [ZeroER: Entity Resolution using Zero Labeled Examples.](https://arxiv.org/abs/1908.06049)

## Setup enviroment
    conda env create -f environment.yml
    conda activate ZeroER

## How to use
1. Put you dataset into the folder `datasets`. You should have a file `metadata.txt` in your data folder that specifies the file name of the table (and possibly right table and ground-truth table). For two table record linkage, you can refer to `datasets/fodors_zagats/metadata.txt`. 
For single table deduplication, you can refer to `datasets/fodors_zagats_single/metadata.txt`.
2. Write a blocking function for your dataset and put it in `blocking_functions.py`. 
   You can have a look at the blocking functions we wrote in that file to get some ideas of how to write your own blocking function.
   We use Magellan to do blocking so you can also refer to its [documentations](https://sites.google.com/site/anhaidgroup/projects/magellan/py_entitymatching).

3. **Two table record linkage**. <br />
    To run the code, for example you are using the fodors_zagats dataset:

    `python zeroer.py fodors_zagats`

    If you want to incorporate the transitivity constraint, use arg `--run_transitivity`: 

    `python zeroer.py fodors_zagats --run_transitivity`

    *Note this will generate features for self-join of the two tables (LxL and RxR) when arg `--LR_dup_free` is not present, which can take some time.

    If you know that your left table and right table are duplicate free, you can incorporate this information by using arg `--run_transitivity --LR_dup_free`:

    `python zeroer.py fodors_zagats --run_transitivity --LR_dup_free`
   
   **Single table deduplication**. <br />
  You must explictly tell the system that you are doing single table deduplication by arg `--LR_identical`:

   `python zeroer.py fodors_zagats_single --LR_identical`

   If you want to incorporate the transitivity constraint, add arg `--run_transitivity`:
    
   `python zeroer.py fodors_zagats_single --LR_identical --run_transitivity`

4. Final result for matches and unmatches is the file `pred.csv` that is saved to your dataset folder.

## Citation
If you use our work or found it useful, please cite our paper:
```
@inproceedings{wu2020zeroer,
  author    = {Renzhi Wu and Sanya Chaba and Saurabh Sawlani and Xu Chu and Saravanan Thirumuruganathan},
  title     = {ZeroER: Entity Resolution using Zero Labeled Examples},
  booktitle = {Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data},
  pages     = {1149–1164},
  year      = {2020}
}
```