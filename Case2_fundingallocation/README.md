# README AUSTROADS ML TOOL #

# What is this repository for? #

This repository contains code for running the meta-optimisation method to plot how level of service varies with different budget split and other considerations.

# How do I get set up? #

1. Add dependencies from `requirements.txt` with the command:
    ```
    pip install -r requirements.txt
    ```
2. Run ```python3 src/setup.py install``` or ```python3 src/setup.py develop``` to set up local packages. This step must be done in the folder ```Case2_fundingallocation```
3. Download datasets from SharePoint into `Case2_fundingallocation\data` folder. Make sure paths match src.config file
4. Run src.meta_opt.py file to generate solutions to project selection under budget constraint under a grid of penalties. Output is saved according to the config file.
5. Run `2.0.0_plot.ipynb`, `4.0_result_exploration.ipynb` to explore the output. The level of service at the origin (all penalties = 0) is the "best" split under budget constraint.
6. Run `3.0.0_differential_evolution.ipynb` to compute the optimal penalties.

### Who do I talk to? ###

* Repo owner or admin: david.rawlinson@wsp.com 
* Other community or team contact:
    - long.dang@wsp.com
    - ben.chu@wsp.com
