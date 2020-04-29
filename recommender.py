# /usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   Containerized Amazon Recommender System (CARS) Project
#
#   Authors: Brianna Blain-Castelli, Nikkolas Irwin, Adam Cassell, and Andrew Munoz
#   Date: 04/01/2020
#   Purpose: Build a Big Data application using a Conda environment and Docker.
#   Course: CS 636 Big Data Systems
#   Project: CARS is an application that builds a recommender system from datasets provided by
#            UCSD (see citation below). 
#
#   Dataset URL: https://nijianmo.github.io/amazon/index.html
#
#   ***IMPORTANT*** You must download the dataset files for a particular category to your local machine yourself due
#                   to their size. As long as your dataset files are in the same directory as the Dockerfile, then
#                   they will be added to the volume and usable by the container as expected.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   Citation: Justifying recommendations using distantly-labeled reviews and fined-grained aspects
#             Jianmo Ni, Jiacheng Li, Julian McAuley
#             Empirical Methods in Natural Language Processing (EMNLP), 2019
#             PDF: http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   Python Recommender File: Python file for driving the core recommender system logic for CARS, using the ALS algorithm.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse, os, re, sys, textwrap as tw, webbrowser
from sys import platform
from pyspark import SparkContext
from pyspark.sql.functions import rand
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import pandas
from IPython.display import display
import matplotlib.pyplot as plt
import vis


# Set the environment
if platform == "linux" or platform == "linux2":
    # Linux
    # Set the Java PATH for JAVA_HOME so that PySpark can utilize the SDK.
    os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME',
                                             default='/usr/lib/jvm/java-8-openjdk-amd64')
    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--master local[2] pyspark-shell'
elif platform == "darwin":
    # OS X
    # Set the Java PATH for JAVA_HOME so that PySpark can utilize the SDK.
    os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME',
                                             default='/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home')
    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--master local[2] pyspark-shell'
elif platform == "win32":
    os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', 
                                             default='C:\\Program Files\\Java\\jdk1.8.0_121')


def welcome_message():
    print('\n\nWelcome to the Containerized Amazon Recommender System (CARS)!')


def select_dataset(file=None):
    if file is None:
        dataset_directory = os.listdir(path='datasets')
        files = dataset_directory
        if platform == "darwin":
            files.remove('.DS_Store')

        file_count = 1
        print(f'\n\nSelect a dataset to run from {file_count} files listed below.\n\n')
        for file in files:
            print('File', str(file_count).zfill(2), '-', file)
            file_count += 1

        dataset = str(input('\n\nDataset: '))
    else:
        dataset = file
        
    if dataset.endswith('.json'):
        print(f'\n\nRunning CARS using the {dataset} dataset...\n')
    else:
        dataset = dataset + '.json'
        print(f'\n\nRunning CARS using the {dataset} dataset...\n')

    return dataset


def configure_core_count():
    logical_cores_to_allocate = str(input('Select the number of logical cores to use for the Spark Context: '))
    return logical_cores_to_allocate


def initialize_spark_context(cores_allocated='*'):
    print(f'\nInitializing Spark Context with {cores_allocated} logical cores...\n\n')

    sc = SparkContext(f'local[{cores_allocated}]')

    print('\n\n...done!\n')

    return sc


def initialize_spark_session():
    print('\nCreating Spark Session...\n')

    ss = SparkSession.builder.appName('Recommendation_system').getOrCreate()

    print('\n...done!\n')

    return ss


def activate_spark_application_ui():
    print('\nOpening Web Browser for the Spark Application UI...\n')
    webbrowser.open('http://localhost:4040/jobs/')
    print('\n...done!\n')


def run_spark_jobs(dataset=None, spark=None):
    print('\nProcessing the dataset...\n')
    df = spark.read.json(f'./datasets/{dataset}')
    print('\n...done!\n')

    print('\nShowing the first 100 results from the dataset...\n\n')
    df.show(100, truncate=True)
    print('\n...done!\n')

    print('\nSelecting the Product ID (ASIN), Overall Rating, and Reviewer ID from the dataset...\n')
    nd = df.select(df['asin'], df['overall'], df['reviewerID'])
    print('\n...done!\n')

    print('\nShowing the first 100 results from the filtered dataset...\n\n')
    nd.show(100, truncate=True)
    print('\n...done!\n')

    print('\nShowing summary statistics for the filtered dataset...\n\n')
    overall = nd.select(nd['overall']).toPandas()
    print(overall.describe())
    summary_vis = vis.Vis("summary",overall)

    print('\n...done!\n')

    print('\nConverting the Product ID (ASIN) and Reviewer ID columns into index form...\n')
    indexer = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in
               list(set(nd.columns) - {'overall'})]
    pipeline = Pipeline(stages=indexer)
    transformed = pipeline.fit(nd).transform(nd)
    print('\n...done!\n')

    print('\nShowing the first 100 results from the converted dataset...\n\n')
    transformed.show(100, truncate=True)
    print('\n...done!\n')

    print('\nCreating the training and test datasets with an 80/20 split respectively...\n')
    (training, test) = transformed.randomSplit([0.8, 0.2])
    print('\n...done!\n')

    print('\nCreating the ALS model...\n')
    als = ALS(maxIter=5,
              regParam=0.09,
              rank=25,
              userCol="reviewerID_index",
              itemCol="asin_index",
              ratingCol="overall",
              coldStartStrategy="drop",
              nonnegative=True)
    print('\n...done!\n')

    print('\nFitting and training the data using ALS...\n\n')
    model = als.fit(training)
    print('\n\n...done!\n')

    print('\nGenerating predictions...\n')
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")
    predictions = model.transform(test)
    print('\n...done!\n')

    print('\nCalculating the Root Mean Square Error (RMSE)...\n')
    root_mean_square_error = evaluator.evaluate(predictions)
    print("\nROOT MEAN SQUARE ERROR = " + str(root_mean_square_error), "\n")
    print('\n...done!\n')

    print('\nDisplaying the first 100 predictions...\n\n')
    predictions.show(100, truncate=True)
    print('\n...done!\n')

    print('\nDisplaying the first 20 recommendations for the first 100 users...\n\n')
    user_recs = model.recommendForAllUsers(8).show(100, truncate=False, vertical=True)
    print('\n...done!')


def exit_message(sc=None, browser_on=False):
    
    while browser_on:
        choice = input('\n\nShutdown the program? [\'y\' for yes, \'n\' for no]: ')
        if choice == str('y').lower():
            browser_on = False
        else:
            continue

    print('\n\nStopping the Spark Context...\n')
    sc.stop()
    print('\n...done!\n')


def execute_recommender_system(command_line_arguments=None):
    try:  # Attempt to run the recommender system and associated startup methods.
        welcome_message()
        amazon_dataset = select_dataset(command_line_arguments.file)
        logical_cores = None
        if command_line_arguments.cores is None:
            logical_cores = configure_core_count()
        else:
            logical_cores = command_line_arguments.cores
        spark_context = initialize_spark_context(cores_allocated=logical_cores)
        spark_session = initialize_spark_session()
        activate_spark_application_ui()
        run_spark_jobs(dataset=amazon_dataset, spark=spark_session)
        exit_message(sc=spark_context, browser_on=command_line_arguments.online)
    except Exception as execution_err:  # Catch any error type, print the error, and exit the program.
        print(execution_err)
        sys.exit('\nExiting the program due to an unexpected error. The details are shown above.')

# Initialize Parser
def init_argparser() -> argparse.ArgumentParser:
    formatter = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=140, width=150)
    
    parser = argparse.ArgumentParser(
                        formatter_class=formatter,
                        prog="cars",
                        usage="recommender.py",
                        description="cars is a application that runs a recommender system in a containerized environment.",
                        )

    # Get core count on host machine or Docker container
    workers = os.cpu_count()
    cores = range(1, workers + 1)
    cores_min = cores[0]
    cores_max = len(cores)
    
    parser.add_argument("-c", "--cores",
                        choices=range(1, (cores_max + 1)),
                        default=1,
                        type=int,
                        help="specify the logical core count for the Spark Context",
                        metavar="[{0}-{1}]".format(cores_min, cores_max),
                        )
    
    # Remove Mac OS .DS_Store File
    dataset_directory = os.listdir(path='datasets')
    files = dataset_directory
    if platform == "darwin":
            files.remove('.DS_Store')
    
    # Remove the brackets, pair of single quotes, and comma from all files.
    # Add a newline character at the end of each file
    # Create a line of text between each file, the help message, and the next option
    filestring = 'available files are shown below\n' + \
                 '-' * 31 + '\n' + '\n'.join(files) + '\n' + \
                 '-' * 31 + '\n'
    
    parser.add_argument("-f", "--file",
                        help=filestring,
                        metavar="<filename>.json",
                        )
    
    parser.add_argument("-l", "--log-file",
                        help="save output to log",
                        metavar="/path/to/<filename>.log",
                        )
    
    parser.add_argument("-o", "--online",
                        action="store_true",
                        help="turn on Spark UI",
                        )                       
    
    parser.add_argument("-s", "--show-visualizations",
                        action="store_true",
                        help="turn on data visualizations",
                        )
    
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="enable verbose command line output for intermediate spark jobs",
                        )

    parser.add_argument("--version",
                        action="version",
                        version="%(prog)s 1.0.0",
                        help="displays the current version of %(prog)s",
                        )
    
    return parser


try:  # Run the program only if this module is set properly by the interpreter as the entry point of our program.
    if __name__ == '__main__':
        # Execute the command line parser.
        parser = init_argparser()
        args = parser.parse_args()
        print('\n\nNo exceptions were raised.')
    else:  # If this module is imported raise/throw an ImportError.
        raise ImportError
except ImportError:  # If an ImportError is thrown exit the program immediately.
    sys.exit('Import Error: recommender.py must be run directly, not imported.')
except Exception as err:  # Print any other exception that causes the program to not start successfully.
    print(err)
else:  # Call the main function if no exceptions were raised    
    # After getting command line arguments, execute the application if no errors occur.
    print('\n\nStarting the program.')
    execute_recommender_system(command_line_arguments=args)
    print('\nExiting the program.')
