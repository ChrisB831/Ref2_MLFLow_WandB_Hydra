#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset

Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3) Execute
    mlflow run . -P steps=test_regression_model
"""
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import wandb


# Create a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def go(args):
    '''Main functionality call
    input:
        args: Argparse ArgumentParser objects holding the command line arguments
    output:
        None
    '''
    logger.info("TRF TEST MLF COMPONENT: Called")
    logger.info("TRF TEST MLF COMPONENT: mlflow_model: {0}".format(args.mlflow_model))
    logger.info("TRF TEST MLF COMPONENT: test_dataset: {0}".format(args.test_dataset))


    # Initialise the run and add the command line arguments
    run = wandb.init(job_type="test_model")
    run.config.update(args)


    # Download model artifact to a local directory and return the path
    logger.info("TRF TEST MLF COMPONENT: Download model inference artifact")
    model_local_path = run.use_artifact(args.mlflow_model).download()


    # Download test dataset artifact to a local directory and return the path
    logger.info("TRF TEST MLF COMPONENT: Download test dataset artifact")
    test_dataset_path = run.use_artifact(args.test_dataset).file()


    # Load test dataset into a dataframe and create label and feature datasets
    logger.info("TRF TEST MLF COMPONENT: Load test data into df")
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")


    # Load model pipeline
    logger.info("TRF TEST MLF COMPONENT: Load model inference pipeine")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)


    # Perform inference on test data
    logger.info("TRF TEST MLF COMPONENT: Performing inference on test data")
    y_pred = sk_pipe.predict(X_test)


    # Generate and log r2 and MAE for test data
    logger.info("TRF TEST MLF COMPONENT: Calculating R2 and MAE")
    r_squared = sk_pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"TRF TEST MLF COMPONENT: R2: {r_squared}")
    logger.info(f"TRF TEST MLF COMPONENT: MAE: {mae}")
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae



# Top level script entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()
    go(args)
