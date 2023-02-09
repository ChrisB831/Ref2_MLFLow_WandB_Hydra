# Summary
This project focusses primarliy on MLFlow. 

This code contains the following MLflow components:
* eda
* basic_cleaning
* data_check
* train_val_test_split
* train_random_forest
* test_regression_model

Howvever various other tool are explored, specifically
* **Conda** for MLProject environment management
* **Weights** and biaes for model artifact management
* **Hydra** for configuration control and the running of hyperparameter tuning
* **Argparse** for CLI functionality
* **Pytest** (and conftest) for testing



Full details are given in the respective source code


The root contains the following files

* `config.yaml`. Contain run hyperparemeters
* `conda.yaml`. Defines the outer conda environment
* `MLProject`. Defines the project including the entry point
* `main.py`. Main code including the setup and individual MLFLow component calls



# Calling

1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project



Run all steps This uses the main.steps parameter to define the steps  
`mlflow run .`



Run individual steps (example)download and basic_cleaning steps  
`mlflow run . -P steps=download,basic_cleaning`



You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a `hydra_options` parameter. For example, say that we want to set the parameters `modeling.random_forest.n_estimators` to 10 and `etl.min_price` to 50

```
mlflow run . \
	-P steps=download,basic_cleaning \
	-P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```



To download version 1.0.0 of the inference pipeline from the GitHub repo, and apply it to a new dataset

```
mlflow run \
	https://github.com/ChrisB831/build-ml-pipeline-for-short-term-rental-prices.git \
	-v 1.0.0 \
-P hydra_options="etl.sample='sample2.csv'"
```



To download the **Live** version, use version 1.0.1

```
mlflow run \
	https://github.com/ChrisB831/build-ml-pipeline-for-short-term-rental-prices.git \
	-v 1.0.1 \
	-P hydra_options="etl.sample='sample2.csv'"
```



# Repos
W&B (this is private)
https://wandb.ai/chris-bonham831/nyc_airbnb?workspace=user-chris-bonham831

Github (as this is forked its public)
https://github.com/ChrisB831/build-ml-pipeline-for-short-term-rental-prices
