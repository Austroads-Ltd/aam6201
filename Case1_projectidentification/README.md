# README AUSTROADS ML TOOL #

# What is this repository for? #

This repository contains code for preprocessing raw data and modelling them to predict pavement treatments over a period of time horizons.

# How do I get set up? #

1. Add dependencies from `requirements.txt` with the command:
    ```
    pip install -r requirements.txt
    ```
2. For macOS users, please install libomp if you have not installed it in the past using ```brew install libomp```
3. Run ```python3 setup.py install``` and ```python3 setup.py develop``` to set up local packages.
4. Download datasets from SharePoint into `Case1_projectidentification\data` folder.
5. Verify data paths match those specified by config files.
6. Run all notebooks in increasing order of their numerical name for each jurisdiciton. Run notebooks in ```notebooks\shared``` directory after running the notebooks for the respective jursidictions.

    The directory structure is as followed:

    ```
    .
    +-- notebooks
    |   +-- jurisidiction_name
    |   |   +-- 2.0.0 name_preprocessing <- data preprocessing notebook for encoding, normalizing, filitering, etc.
    |   |   +-- 3.0.0 name_modelling <- modelling notebook
    |   |   +-- 3.0.1 name_validation <- validaiton notebook. Must be run after modelling notebook
    |   +-- shared
    |   |   +-- cluster_predictions.ipynb <- Work with project-level predictions instead of section-level predictions
    |   |   +-- feature_inspection.ipynb <- Perform methods for understanding models
    |   |   +-- shared_plotting.ipynb <- Plot results jointly for all jurisdictions
    |   |   +-- transfer_inspection.ipynb <- Provide methods for comparing jurisidictions' data distributions, label distributions, relationships between labels, etc.
    +-- src
    |   +-- folders containing utility methods
    ```

7. At the start of each notebook (or script), the names of the files for saved results (i.e. processed data, trained models, etc.) are defined. The most important element is the `experiment_suffix` variable, which is appended after the template filename. Once this variable is defined, other path variables are based on them; for example, the root folder for saving trained models is `/models/trained/{jurisdiction_name}/{experiment_folder} + dir`. For the modelling notebooks, there is also the `experiment_prefix` variable which helps clarify the type of dataset used to train the model, i.e. a train/test/valid set. (Note: There is an issue where if `experiment_prefix` is set to `""`, there may be naming inconsistencies between notebooks. `experiment_suffix` should therefore be set to a default `default` if no clear name exists.

# Utilties
## Exporting notebooks as pdf

* Once the environment has been made, run this command:
    ```
    pyppeteer-install
    ```
    * This installs chromimum engine which we rely on to generate pdf out of html without going through latex

* To output the PDF, run this command:
    ```
    python3 src/visualization/convert_notebook_to_pdf.py path/to/notebook.ipynb --output-dir /path/to/output_dir>
    ```

* Code copied and adapted from ```notebook-as-pdf``` under BSD-3 License. Package's github for reference: [https://github.com/betatim/notebook-as-pdf](https://github.com/betatim/notebook-as-pdf)

### Who do I talk to? ###

* Repo owner or admin: david.rawlinson@wsp.com 
* Other community or team contact:
    - long.dang@wsp.com
    - ben.chu@wsp.com
