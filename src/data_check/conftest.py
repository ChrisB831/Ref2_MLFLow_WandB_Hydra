'''
Pretest cnonfiguration functionality
Author: Chris Bonham
Date: 7th January 2023
'''
import pytest
import pandas as pd
import wandb



def pytest_addoption(parser):
    '''Get the command line arguments passed at the Pytest call

    input:
        parse: PyTest parser object containing the parsed command line
        arguments
    output:
        None
    '''
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")



@pytest.fixture(scope='session')
def data(request):
    '''Fixture to download and return the input artifact
    NB Scope is session so input dataset can be modified by a test

    input:
        request: PyTest request object
    output:
        df: Pandas dataframe containing the input data
    '''

    # Initialise the run
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact to local drive and return the path
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df



@pytest.fixture(scope='session')
def ref_data(request):
    '''Fixture to download and return the reference artifact
    NB Scope is session so input dataset can be modified by a test

    input:
        request: PyTest request object
    output:
        df: Pandas dataframe containing the reference data
    '''


    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact to local drive and return the path
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df



@pytest.fixture(scope='session')
def kl_threshold(request):
    ''' Fixture to return the kl_threshold
    NB Scope is session so value can be modified by a test

    input:
        request: PyTest request object
    output:
        float: kl test threshold
    '''
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)



@pytest.fixture(scope='session')
def min_price(request):
    ''' Fixture to return the min_price
    NB Scope is session so value can be modified by a test

    input:
        request: PyTest request object
    output:
        float: minimum allowed price
    '''
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)



@pytest.fixture(scope='session')
def max_price(request):
    '''Fixture to return the max_price
    NB Scope is session so value can be modified by a test

   input:
        request: PyTest request object
    output:
        float: maximum allowed price
    '''
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)