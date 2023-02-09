#!/usr/bin/env python
'''
A module to run tests against the artifacts downloaded from W&B

Author: Chris Bonham
Date: 25th January 2023

MLflow component call (in isolation)
1) Create and activate a conda environment that contains Mlflow v2.0.1
2) Go to the route of this project
3) Execute
    mlflow run . -P steps=data_check
'''
import numpy as np
import scipy.stats



def test_column_names(data):
    '''Test input data contains the expected columns and by implication are also in the same order

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365"
    ]

    assert list(expected_colums) == list(data.columns.values)



def test_neighborhood_names(data):
    '''Test expected list of neighbourhood are present in the data

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)



def test_proper_boundaries(data):
    """Test proper longitude and latitude boundaries for properties in and around NYC

    input:
        data: pandas dataframe. Test data
    output:
        None
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0



def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    """Use a KL divergence test (using a kl_threshold) to determine if the distribution of
    the new data is significantly different than that of the reference dataset

    input:
        data: pandas dataframe. Test data
        ref_data: pandas dataframe. Reference data
        kl_threshold: float test threshold
    output:
        None
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold



def test_row_count(data):
    '''Check the number of rows in the dataset falls between 15k and 1m

    input:
        data: pandas dataframe. Test data
    output:
        None
    '''
    assert 15000 < data.shape[0] < 1000000



def test_price_range(data, min_price, max_price):
    '''Check the price always falls between the minimum and maximum price

    input:
        data: pandas dataframe. Test data
        min_price: flaot. Minimum allowed price
        max_price: flaot. Maximum allowed price
    output:
        None
    '''
    assert data["price"].dropna().between(min_price, max_price).all()
