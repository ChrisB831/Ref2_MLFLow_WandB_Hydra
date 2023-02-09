#!/usr/bin/env python
'''
Entry point module to call the individual MLFLow components
Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3a) Run all steps This uses the main.steps parameter to define the steps
    mlflow run .
3b) Run individual steps (example)download and basic_cleaning steps
    mlflow run . -P steps=download,basic_cleaning
3c) You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a `hydra_options` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:
    mlflow run . \
      -P steps=download,basic_cleaning \
      -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
3d)  To download version 1.0.0 of the inference pipeline from my github repo, and apply it
     to a new dataset
        mlflow run \
            https://github.com/ChrisB831/build-ml-pipeline-for-short-term-rental-prices.git \
            -v 1.0.0 \
            -P hydra_options="etl.sample='sample2.csv'"

    Use version 1.0.1
    mlflow run \
        https://github.com/ChrisB831/build-ml-pipeline-for-short-term-rental-prices.git \
        -v 1.0.1 \
        -P hydra_options="etl.sample='sample2.csv'"
'''


import json
import tempfile
import os
import logging
import mlflow
import hydra
from omegaconf import DictConfig


# A list of "all" steps to run as a default
all_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    '''Main functionality call
    input:
        configs: Hydra DictConfig. The imported config.yaml file
    output:
        None
    '''


    logger.info("OUTER MLF COMPONENT: Called")
    logger.info("OUTER MLF COMPONENT: Project name: {0}".format(config["main"]["project_name"]))
    logger.info("OUTER MLF COMPONENT: Experiment name: {0}".format(config["main"]["experiment_name"]))


    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]


    # Get the steps to execute
    parsed_steps = config['main']['steps']
    if parsed_steps != "all":
        active_steps = parsed_steps.split(",")
    else:
        active_steps = all_steps
    logger.info("OUTER MLF COMPONENT: Active steps: {0}".format(active_steps))


    # Get the project root path
    root = hydra.utils.get_original_cwd()
    logger.info("OUTER MLF COMPONENT: Project root: {0}".format(root))


    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Run the get data component
        # NB This is stored remotely in a github repo, URL is
        # https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components/get_data
        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )


        # Run the basic cleaning component
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(root, "src", "basic_cleaning"),
                "main",
                parameters={
                    "mlf_input_artifact": "sample.csv:latest",
                    "mlf_output_artifact": "clean_sample.csv",
                    "mlf_output_type": "clean_sample",
                    "mlf_output_description": "Data with outliers and null values removed",
                    "mlf_min_price": config['etl']['min_price'],
                    "mlf_max_price": config['etl']['max_price']
                },
            )


        # Run the data check component
        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(root, "src", "data_check"),
                "main",
                parameters={
                    "mlf_csv": "clean_sample.csv:latest",
                    "mlf_ref": "clean_sample.csv:reference",
                    "mlf_kl_threshold": config['data_check']['kl_threshold'],
                    "mlf_min_price": config['etl']['min_price'],
                    "mlf_max_price": config['etl']['max_price']
                },
            )


        # Run the train and other split component
        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(root, "src", "train_val_test_split"),
                "main",
                parameters={
                    "mlf_input": "clean_sample.csv:latest",
                    "mlf_test_size": config['modeling']['test_size'],
                    "mlf_random_seed": config['modeling']['random_seed'],
                    "mlf_stratify_by": config['modeling']['stratify_by'],
                },
            )


        # Run the train random forest component
        if "train_random_forest" in active_steps:

            # Serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            # NB We pass the Random forest configuration as the JSON serialsed above
            _ = mlflow.run(
                os.path.join(root, "src", "train_random_forest"),
                "main",
                parameters={
                    "mlf_trainval_artifact": "trainval_data.csv:latest",
                    "mlf_val_size": config['modeling']['val_size'],
                    "mlf_random_seed": config['modeling']['random_seed'],
                    "mlf_stratify_by": config['modeling']['stratify_by'],
                    "mlf_rf_config": fp.name,
                    "mlf_max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "mlf_output_artifact": "random_forest_export"
                },
            )


        # Run the test model component
        # This run is not called by default as part of the job flow
        # Call this step explicitly ONCE you have a model package marked as Prod
        # mlflow run . -P steps=test_regression_model
        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(root, "src", "test_regression_model"),
                "main",
                parameters={
                    "mlf_mlflow_model": "random_forest_export:prod",
                    "mlf_test_dataset": "test_data.csv:latest"
            },
    )



# Top level script entry point
if __name__ == "__main__":
    go()
