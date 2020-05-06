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

import argparse, logging, os, re, sys, textwrap as tw, webbrowser
from sys import platform
from pyspark import SparkContext
from pyspark.sql.functions import rand, col
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
    logging.info('\n\nWelcome to the Containerized Amazon Recommender System (CARS)!')


def select_dataset(file=None):
    if file is None:
        dataset_directory = os.listdir(path='datasets')
        files = dataset_directory
        if platform == "darwin":
            files.remove('.DS_Store')

        file_count = 1
        print(f'\n\nSelect a dataset to run from {file_count} files listed below.\n\n')
        logging.info(f'\n\nSelect a dataset to run from {file_count} files listed below.\n\n')
        for file in files:
            print('File', str(file_count).zfill(2), '-', file)
            file_count += 1

        dataset = str(input('\n\nDataset: '))
    else:
        dataset = file
        
    if dataset.endswith('.json'):
        print(f'\n\nRunning CARS using the {dataset} dataset...\n')
        logging.info(f'\n\nRunning CARS using the {dataset} dataset...\n')
    else:
        dataset = dataset + '.json'
        print(f'\n\nRunning CARS using the {dataset} dataset...\n')
        logging.info(f'\n\nRunning CARS using the {dataset} dataset...\n')

    return dataset


def configure_core_count():
    logical_cores_to_allocate = str(input('Select the number of logical cores to use for the Spark Context: '))
    return logical_cores_to_allocate


def initialize_spark_context(cores_allocated='*'):
    print(f'\nInitializing Spark Context with {cores_allocated} logical cores...\n\n')
    logging.info(f'\nInitializing Spark Context with {cores_allocated} logical cores...\n\n')

    sc = SparkContext(f'local[{cores_allocated}]')

    print('\n\n...done!\n')
    logging.info('\n\n...done!\n')

    return sc


def initialize_spark_session():
    print('\nCreating Spark Session...\n')
    logging.info('\nCreating Spark Session...\n')

    ss = SparkSession.builder.appName('Recommendation_system').getOrCreate()

    print('\n...done!\n')
    logging.info('\n...done!\n')

    return ss


def activate_spark_application_ui():
    print('\nOpening Web Browser for the Spark Application UI...\n')
    logging.info('\nOpening Web Browser for the Spark Application UI...\n')
    webbrowser.open('http://localhost:4040/jobs/')
    print('\n...done!\n')
    logging.info('\n...done!\n')


def run_spark_jobs(dataset=None, num_predictions=5, rows=10, spark=None, verbose=False):
    if verbose:
        print('\nProcessing the dataset...\n')
        logging.info('\nProcessing the dataset...\n')
    df = spark.read.json(f'./datasets/{dataset}')
    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print(f'\nShowing the first {rows} results from the dataset...\n\n')
        logging.info(f'\nShowing the first {rows} results from the dataset...\n\n')
        df.show(rows, truncate=True)
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nSelecting the Product ID (ASIN), Overall Rating, and Reviewer ID from the dataset...\n')
        logging.info('\nSelecting the Product ID (ASIN), Overall Rating, and Reviewer ID from the dataset...\n')
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

    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print(f'\nShowing the first {rows} results from the filtered dataset...\n\n')
        logging.info(f'\nShowing the first {rows} results from the filtered dataset...\n\n')
        nd.show(rows, truncate=True)
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nShowing summary statistics for the filtered dataset...\n\n')
        logging.info('\nShowing summary statistics for the filtered dataset...\n\n')
        nd.describe(['asin', 'overall', 'reviewerID']).show()
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nConverting the Product ID (ASIN) and Reviewer ID columns into index form...\n')
        logging.info('\nConverting the Product ID (ASIN) and Reviewer ID columns into index form...\n')

    indexer = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in
               list(set(nd.columns) - {'overall'})]
    pipeline = Pipeline(stages=indexer)
    transformed = pipeline.fit(nd).transform(nd)
    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print(f'\nShowing the first {rows} results from the converted dataset...\n\n')
        logging.info(f'\nShowing the first {rows} results from the converted dataset...\n\n')
        transformed.show(rows, truncate=True)
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nCreating the training and test datasets with an 80/20 split respectively...\n')
        logging.info('\nCreating the training and test datasets with an 80/20 split respectively...\n')
    (training, test) = transformed.randomSplit([0.8, 0.2])
    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nCreating the ALS model...\n')
        logging.info('\nCreating the ALS model...\n')
    als = ALS(maxIter=5,
              regParam=0.09,
              rank=25,
              userCol="reviewerID_index",
              itemCol="asin_index",
              ratingCol="overall",
              coldStartStrategy="drop",
              nonnegative=True)
    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nFitting and training the data using ALS...\n\n')
        logging.info('\nFitting and training the data using ALS...\n\n')
    model = als.fit(training)
    if verbose:
        print('\n\n...done!\n')
        logging.info('\n\n...done!\n')

    if verbose:
        print('\nGenerating predictions...\n')
        logging.info('\nGenerating predictions...\n')
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")
    predictions = model.transform(test)
    if verbose:
        print('\n...done!\n')
        logging.info('\n...done!\n')

    if verbose:
        print('\nCalculating the Root Mean Square Error (RMSE)...\n')
        logging.info('\nCalculating the Root Mean Square Error (RMSE)...\n')
        root_mean_square_error = evaluator.evaluate(predictions)
        print("\nROOT MEAN SQUARE ERROR = " + str(root_mean_square_error), "\n")
        logging.info("\nROOT MEAN SQUARE ERROR = " + str(root_mean_square_error) + " \n")
        print('\n...done!\n')
        logging.info('\n...done!\n')

    # print(f'\nDisplaying the first {rows} predictions...\n\n')
    # logging.info(f'\nDisplaying the first {rows} predictions...\n\n')
    # predictions.show(rows, truncate=True)
    predictions.show(10, truncate=True)
    print('\n...done!\n')
    logging.info('\n...done!\n')

    # Randomly sample small subset of prediction data for better plotting performance
    print("Sampling prediction results and converting to pandas for visualization...")
    predictions_sample = predictions.sample(False, 0.01, seed=0)
    print("\n...done!")

    prediction_vis = vis.Vis("prediction",predictions_sample)

    # print(f'\nDisplaying the first {num_predictions} recommendations for the first {rows} users...\n\n')
    # logging.info(f'\nDisplaying the first {num_predictions} recommendations for the first {rows} users...\n\n')
    # user_recs = model.recommendForAllUsers(num_predictions).show(rows, truncate=False, vertical=True)
    user_recs = model.recommendForAllUsers(10).show(10, truncate=False, vertical=True)
    print('\n...done!')
    logging.info('\n...done!')


def exit_message(sc=None, browser_on=False):
    
    while browser_on:
        choice = input('\n\nShutdown the program? [\'y\' for yes, \'n\' for no]: ')
        if choice == str('y').lower():
            browser_on = False
        else:
            continue

    print('\n\nStopping the Spark Context...\n')
    logging.info('\n\nStopping the Spark Context...\n')
    sc.stop()
    print('\n...done!\n')
    logging.info('\n...done!\n')


def execute_recommender_system(command_line_arguments=None):
    try:  # Attempt to run the recommender system and associated startup methods.
        if command_line_arguments.log_file:
            filename = command_line_arguments.log_file
            if not filename.endswith('.log'):
                filename = filename + '.log'
            logging.basicConfig(filename=filename, filemode='w', format='%(message)s', level=logging.INFO)
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
        run_spark_jobs(dataset=amazon_dataset,
                       num_predictions=command_line_arguments.predictions,
                       rows=command_line_arguments.rows,
                       spark=spark_session,
                       verbose=command_line_arguments.verbose)
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
                        description="""
# ------------------------------------------------------------------------------------------------------------------
#                        Containerized Amazon Recommender System (CARS) Project
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
# ------------------------------------------------------------------------------------------------------------------
#
#   Citation: Justifying recommendations using distantly-labeled reviews and fined-grained aspects
#             Jianmo Ni, Jiacheng Li, Julian McAuley
#             Empirical Methods in Natural Language Processing (EMNLP), 2019
#             PDF: http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
# ------------------------------------------------------------------------------------------------------------------
                        \ncars\n""",
                        )

    # Set available datasets for --help
    available_datasets = parser.add_argument_group('available datasets')

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
                 '*' * 31 + '\n' + '\n'.join(files) + '\n' + \
                 '*' * 31 + '\n'
    
    available_datasets.add_argument("-f", "--file",
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
    
    npredictions = 20
    parser.add_argument("-p", "--predictions",
                        choices=range(1, (npredictions + 1)),
                        type=int,
                        help="number of predictions to calculate",
                        metavar="[{0}-{1}]".format(1, npredictions),
                        )   
    
    nrows = 5000
    parser.add_argument("-r", "--rows",
                        choices=range(1, (nrows + 1)),
                        type=int,
                        help="top (n) rows to display",
                        metavar="[{0}-{1}]".format(1, nrows),
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
