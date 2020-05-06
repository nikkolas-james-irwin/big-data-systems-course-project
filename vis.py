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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

class Vis:

    def __init__(self, type, data):
        self.type = type    # instance variable unique to each instance
        self.data = data
        if (self.type == "summary"):
            self.vis_summary(self.data)
        elif (self.type == "helpful"):
            self.vis_helpful_review(self.data)
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


    def vis_helpful_review(self,data):

        df = data.toPandas()
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scattergl(x=df['reviewerID'], y=df['overall'], name="Ratings", mode='markers',),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scattergl(x=df['reviewerID'], y=df['vote'], name="Votes", mode='markers',),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Ratings/Votes Correlation"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Reviews")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Ratings</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Votes</b>", secondary_y=True)

        fig.show()