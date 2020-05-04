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
#   Python Vis Module: Python module containing Vis class. To be used for all visualization types.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas
import matplotlib.pyplot as plt
import matplotlib.style as sty
from IPython.display import display

class Vis:

    vis_dict = {}

    def __init__(self, type, data, spark=None):
        self.type = type    # instance variable unique to each instance
        self.data = data
        if (self.type == "summary"):
            self.vis_summary(self.data)
        elif (self.type == "time"):
            self.vis_timeseries(self.data,spark)
        else:
            raise Exception("Invalid visualization type")

    def vis_summary(self, data):
        # fig1, ax1 = plt.subplots()
        # data.boxplot(ax=ax1)
        # ax1.set_title('Ratings Summary')
        fig1, ax1 = plt.subplots()
        num_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        data.hist(ax=ax1,bins=num_bins, edgecolor='white')
        ax1.set_title('Ratings Distribution')
        ax1.grid(False)
        plt.show()

    def vis_timeseries(self, data, spark):
        k = 20
        # data.createOrReplaceTempView('TBL_RATING_TIMESERIES')
        # top_items = spark.sql('''SELECT count(ASIN) AS `itemCount`,
        #                             asin AS `itemID`
        #                      FROM TBL_RATING_TIMESERIES
        #                      GROUP BY ASIN
        #                      ORDER BY `itemCount` DESC''')

        # print(f'\n\nShowing the top {k} items for the dataset.', '\n')
        # top_items.show(n=k)

        data.createOrReplaceTempView('TBL_RATING_TIMESERIES')
        df = spark.sql('''SELECT count(ASIN) as `Review Count`, unixReviewTime AS `Review Date`  FROM TBL_RATING_TIMESERIES WHERE asin = 
        (SELECT asin AS `itemID`
                FROM TBL_RATING_TIMESERIES
                GROUP BY ASIN
                ORDER BY count(ASIN) DESC LIMIT 1) 
        GROUP BY `Review Date` ORDER BY `Review Date` ''')

        print(f'\n\nShowing the popularity over time of the most-reviewed item of the dataset.', '\n')
        #df.show(n=k)

        # Convert milliseconds to date

        df_pandas = df.toPandas()

        # Convert date string to pyspark date type
        df_pandas['Review Date'] = pandas.to_datetime(df_pandas['Review Date'], unit='s')
        display(df_pandas)

        # gca stands for 'get current axis'
        ax = plt.gca()
        df_pandas.plot(kind='line',x='Review Date',y='Review Count',figsize=(16, 4),ax=ax)
        #df.plot(kind='line',x='RT',y='num_pets', color='red', ax=ax)
        plt.show()