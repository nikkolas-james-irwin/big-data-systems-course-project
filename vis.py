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
from IPython.display import display
import plotly.express as px

class Vis:

    def __init__(self, type, data):
        self.type = type    # instance variable unique to each instance
        self.data = data
        if (self.type == "summary"):
            self.vis_summary(self.data)
        elif (self.type == "prediction"):
            self.vis_prediction(self.data)
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

    def vis_prediction(self, data):
        df = data.select(data['overall'], data['prediction']).toPandas()
        print("\nPlotting visualizations...")
        fig = px.scatter(df, x="overall", y="prediction", width=400, height=400,render_mode='webgl')
        fig.show()

        df['error'] = df['prediction'] - df['overall']
        fig2 = px.histogram(df, x="error")
        fig2.update_layout(
            bargap=0.1,
            title={
                'text': "ALS Prediction Error Distribution",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig2.show()