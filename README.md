Motivation
---
Speed up the feature matching step in image stitching.

Technical details can be found in the project presentation slides [[link](https://drive.google.com/open?id=1amCMWRQyxwnu8AbeN6myjMdT64BYCb_nHBh_VRAnuIA)] and report (under the report directory).


Run
---
All development in this project was done on Wisconsin's commodity cluster Euler
[[link](http://wacc.wisc.edu/docs/)], where jobs are executed through the job
scheduler Slurm.

```
>> cmake -DCMAKE_BUILD_TYPE=Release .
>> make
>> sbatch submitToSlurm.sh
>> # View .stdout file
```


Results
---
Timing results are in XML files without a root element. These are found in the
`timingFiles` directory. These results are plotted in `scripts/plot.py` and
plots are found in the `outputImages` directory.

If you're going to run `scripts/plot.py`, remember to first run the following to compile a SQLite extension.

```
>> gcc -g -shared -fPIC extension-functions.c -o extension-functions.so
```


Author
---
Jia-Shen Boon
