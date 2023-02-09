#!/usr/bin/env python
'''
Download the cleaned dataset artifact, splits into a train and other dataset
an upload as artifacts to W&B

Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3) Execute
    mlflow run . -P steps=data_split
'''

import argparse
import logging
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
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
    logger.info("TVTS MLF COMPONENT: Called")
    logger.info("TVTS MLF COMPONENT: input_artifact: {0}".format(args.input))
    logger.info("TVTS MLF COMPONENT: test_size: {0}".format(args.test_size))
    logger.info("TVTS MLF COMPONENT: random_seed: {0}".format(args.random_seed))
    logger.info("TVTS MLF COMPONENT: stratify_by: {0}".format(args.stratify_by))


    # Initialise the run and add the command line arguments
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)


    # Download input artifact and write to Dataframe
    artifact_local_path = run.use_artifact(args.input).file()
    df = pd.read_csv(artifact_local_path)
    logger.info(f"TVTS MLF COMPONENT: Fetching artifact: {args.input}")
    logger.info(f"TVTS MLF COMPONENT: df shape: {df.shape}")


    logger.info("TVTS MLF COMPONENT: Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )
    logger.info(f"TVTS MLF COMPONENT: trainval shape: {trainval.shape}")
    logger.info(f"TVTS MLF COMPONENT: test shape: {test.shape}")


    # Save dataframes to tempfile and upload as artifacts to W&B
    for df, df_name in zip([trainval, test], ['trainval', 'test']):

        logger.info(f"VTS MLF COMPONENT: Creating {df_name}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:
            df.to_csv(fp.name, index=False)

            logger.info(f"VTS MLF COMPONENT: Uploading {df_name}_data.csv to W&B")
            artifact = wandb.Artifact(
                name = f"{df_name}_data.csv",
                type = f"{df_name}_data",
                description = f"{df_name} split of dataset"
            )
            artifact.add_file(fp.name)
            run.log_artifact(artifact)
            artifact.wait()



# Top level script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument(
        "input",
        type=str,
        help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False
    )

    args = parser.parse_args()
    go(args)
