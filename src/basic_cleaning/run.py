#!/usr/bin/env python
'''
A module to download development data, run some basic cleaning routing and load the resulting
.csv file as an artifact upton W&B.

Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3) Execute
    mlflow run . -P steps=basic_cleaning
'''


#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb
import os


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


    logger.info("BASIC_CLEANING MLF COMPONENT: Called")
    logger.info("BASIC_CLEANING MLF COMPONENT: input_artifact: {0}".format(args.input_artifact))
    logger.info("BASIC_CLEANING MLF COMPONENT: output_artifact: {0}".format(args.output_artifact))
    logger.info("BASIC_CLEANING MLF COMPONENT: output_type: {0}".format(args.output_type))
    logger.info("BASIC_CLEANING MLF COMPONENT: output_description: {0}".format(args.output_description))
    logger.info("BASIC_CLEANING MLF COMPONENT: min_price: {0}".format(args.min_price))
    logger.info("BASIC_CLEANING MLF COMPONENT: max_price: {0}".format(args.max_price))


    # Initialise the run
    run = wandb.init(job_type="process_data")


    # Download the artfact to the local drive, its location and import into a dataframe
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)
    logger.info("BASIC_CLEANING MLF COMPONENT: DataFrame imported")


    # Drop outlier records
    logger.info("BASIC_CLEANING MLF COMPONENT: DataFrame rows before outlier drop {0}".format(df.shape[0]))
    keep_idx = df['price'].between(args.min_price, args.max_price)
    df = df[keep_idx].copy()
    logger.info("BASIC_CLEANING MLF COMPONENT: DataFrame rows after outlier drop {0}".format(df.shape[0]))


    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])


    # Drop any records outside the correct geolocation
    # This fixes the induced error with the new training dataset 'sample2.csv'
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()


    # Save result to local file
    filename = "clean_sample.csv"
    df.to_csv(filename, index = False)


    # Create the artifact
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )
    logger.info("BASIC_CLEANING MLF COMPONENT: Clean DF upload artifact created")


    # Add the file to the artifact and upload to W&B
    artifact.add_file(filename)
    run.log_artifact(artifact)
    logger.info("BASIC_CLEANING MLF COMPONENT: Clean DF artifact uploaded to W&B")


    # Delete locally saved file
    os.remove(filename)
    logger.info("BASIC_CLEANING MLF COMPONENT: Temporary file deleted")



# Top level script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Fully-qualified name for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Type for the output",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price, used for clipping",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price, used for clipping",
        required=True,
    )

    # Get the command line argument and pass to the go function
    args = parser.parse_args()
    go(args)