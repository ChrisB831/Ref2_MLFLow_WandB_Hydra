#!/usr/bin/env python
"""
A module to build a random forest and upload the model to W&B and its reporting to W&B

Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3a) Execute with default parameters
    mlflow run . -P steps=basic_cleaning
3b) Execute and override a single parameter using hydra
    mlflow run . -P steps=train_random_forest \
                 -P hydra_options="modeling.random_forest.max_depth=10"
3c) Execute and override multiple parameters using a Hydra sweep
    mlflow run . -P steps=train_random_forest \
                 -P hydra_options="modeling.max_tfidf_features=10,15,30 \
                                   modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
"""
import argparse
import logging
import os
import shutil
import json
import matplotlib.pyplot as plt

import mlflow

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from mlflow.models import infer_signature
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

    logger.info("TRF MLF COMPONENT: Called")
    logger.info("TRF MLF COMPONENT: trainval_artifact: {0}".format(args.trainval_artifact))
    logger.info("TRF MLF COMPONENT: val_size: {0}".format(args.val_size))
    logger.info("TRF MLF COMPONENT: random_seed: {0}".format(args.random_seed))
    logger.info("TRF MLF COMPONENT: stratify_by: {0}".format(args.stratify_by))
    logger.info("TRF MLF COMPONENT: rf_config: {0}".format(args.rf_config))
    logger.info("TRF MLF COMPONENT: max_tfidf_features: {0}".format(args.max_tfidf_features))
    logger.info("TRF MLF COMPONENT: output_artifact: {0}".format(args.output_artifact))


    # Initialise the run and add the command line arguments
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)


    # Read the Random Forest configuration file and add to W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)


    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed


    # Download the train and validation artifact locally, save the path and immport to a df
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(trainval_local_path)
    logger.info(f"TRF MLF COMPONENT: Fetching trainval_artifact artifact: {args.trainval_artifact}")
    logger.info(f"TRF MLF COMPONENT: df shape: {X.shape}")


    # Get train and validation features and lablel
    y = X.pop("price")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )
    logger.info(f"TRF MLF COMPONENT: X_train dimensions: {X_train.shape}")
    logger.info(f"TRF MLF COMPONENT: y_train dimensions: {y_train.shape}")
    logger.info(f"TRF MLF COMPONENT: X_val dimensions: {X_val.shape}")
    logger.info(f"TRF MLF COMPONENT: y_val dimensions: {y_val.shape}")


    logger.info("TRF MLF COMPONENT: Preparing sklearn pipeline")
    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    # Train the model
    # Then fit it to the X_train, y_train data
    logger.info("TRF MLF COMPONENT: Model training")
    sk_pipe.fit(X_train, y_train)


    # Compute r2 and MAE for the validation dataset and add to logher
    # Score method automatically predicts using the model before calculating default
    # performance metric, mean_absolute_error doesnt
    logger.info("TRF MLF COMPONENT: Assessment on Validation")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    logger.info(f"TRF MLF COMPONENT: Validation r2: {r_squared}")
    logger.info(f"TRF MLF COMPONENT: Validation MAE: {mae}")


    # Save model package in sklearn format
    # Remove local directory (random_forest_dir) if it exists
    logger.info(f"TRF MLF COMPONENT: Exporting model to local drive")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=infer_signature(X_val, y_pred),
        input_example=X_val.iloc[:2],
    )


    # Upload the model artifact to W&B
    # Add the rf_config
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type="model_export",
        description="Random Forest pipeline export",
    )
    artifact.add_dir("random_forest_dir")
    run.config.update(rf_config)
    run.log_artifact(artifact)
    artifact.wait()


    # Remove the directory
    shutil.rmtree("random_forest_dir")


    # Save r2 and MAE to the run
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


    # Plot feature importanceand upload the visualisation to W&B
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )



def delta_date_feature(dates):
    """Derive a new feature from a date
    Given a 2d array containing dates (in any format recognized by pd.to_datetime),
    it returns the delta in days between each date and the most recent date in its
    column

    input
        dates: pandas series. Containing the dates to adjust
    output:
        pandas series adjsted dates
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()



def plot_feature_importance(pipe, feat_names):
    '''Generate the feature importances plot

    input
        pipe: sklearn pipeline. Inference pipeline
        featnames: List. List of features in model
    output:
        fig_feat_imp: Matploitlib fig object
    '''

    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]

    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)

    # Plot the importances
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    
    return fig_feat_imp



def get_inference_pipeline(rf_config, max_tfidf_features):
    '''Get the inference pipeline, this consists of three steps
    Define the component tranforms
    Define the preprocessing pipelines
    Define the inference pipeline (preprocessing + decision tree regressor)

    input
        rf_config: Dict containing the decision tree hyperparameters
        max_tfidf_features. Int number of features in vocabulary
    output:
        (sk_pipe, processed_features):
            sk_pipe: sklearn pipeline. Inference pipeline
            processed_features: List. List of features in model
    '''

    # Define the ordinal and nominal (non-ordinal) categorical fields
    # NB We do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in
    # production or during training
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]


    # Define the ordinal transformation
    # The ordinal encoder assigns an ordinal integer to each level of the field {0,1,2 etc.}
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
    ordinal_categorical_preproc = OrdinalEncoder()


    # Define the nominal (non-ordinal) transformation pipeline
    # This is simple (most frequent) imputer and a one hot encoder
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )


    # Define the continnuous feature tranformation (replace missing with zero
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)


    # Create a feature that represents the number of days passed since the last review
    # 1) Impute the missing review date with an old date (indicating there hasn't been
    # a review for a long time)
    # 2) Create a new feature from it. The FunctionTransformer sklearn function creates a
    # transformer from a callable (in this instance delta_date_feature)
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )


    # Some minimal NLP for the "name" column
    # 1) Reshape the text to 1d
    # 2) Fill missing with ""
    # 3) Vecorise
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )


    # Create the final column level preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get e list of all the preprocessed model features (i.e. not dropped by the
    # ColumnTransformer)
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]


    # Initialise the random forest regressor with the parameters defined in rf_config
    random_forest = RandomForestRegressor(**rf_config)


    # Create the inference pipeline
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )

    return sk_pipe, processed_features



# Top level script entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )


    args = parser.parse_args()
    go(args)
