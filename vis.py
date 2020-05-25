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
import matplotlib.style as sty
from plotly.subplots import make_subplots
from IPython.display import display

class Vis:
    
    plotly_table_styles = {
        "header_color": "rgb(49, 130, 189)",
        "row_even_color": "rgb(239, 243, 255)",
        "row_odd_color": "rgb(189, 215, 231)",
        "font_header_color": "white",
        "font_header_size": 14,
        "font_cell_color": "black",
        "font_cell_size": 11,
    }
    
    # Captures graph objects into dictionary for reference
    vis_dict = {}

    def __init__(self, type, data, spark=None, rows=None):
        self.type = type    # instance variable unique to each instance
        self.data = data
        if (self.type == "summary"):
            self.vis_summary(self.data)
        elif (self.type == "helpful"):
            self.vis_helpful_review(self.data,spark)
        elif (self.type == "prediction"):
            self.vis_prediction(self.data)
        elif (self.type == "time"):
            self.vis_timeseries(self.data,spark,rows)
        else:
            raise Exception("Invalid visualization type")

    def vis_summary(self, data):

        #colors = ['lightslategray',] * 5
        dark_blue = self.plotly_table_styles.get('header_color', None)
        light_blue = self.plotly_table_styles.get('row_odd_color', None)
        colors = [[dark_blue,light_blue]*5]

        fig = px.histogram(data, x="overall", color_discrete_sequence =colors, template = "plotly_white")

        fig.update_layout(
            title_text='Ratings Distribution',
            xaxis_title="Ratings",
            yaxis_title="Review Count",
            bargap = 0.1,
            margin=dict(l=240, r=240, t=120, b=0),
        )

        fig.show()

    def vis_helpful_review(self,data,spark):

        data.createOrReplaceTempView('TBL_HELPFUL_REVIEWS')
        result = spark.sql('''SELECT reviewerID, overall, vote FROM TBL_HELPFUL_REVIEWS WHERE asin = 
        (SELECT asin AS `itemID`
                FROM TBL_HELPFUL_REVIEWS
                GROUP BY ASIN
                ORDER BY count(ASIN) DESC LIMIT 1) ''')

        #print(f'\n\nShowing the popularity over time of the most-reviewed item of the dataset...', '\n')


        df = result.toPandas()

        fig0 = make_subplots(rows=1, cols=2)

        fig0.add_trace(
            go.Scattergl(x=df['overall'], y=df['vote'], mode='markers', name="Rating/Vote Correlation",marker_color=self.plotly_table_styles.get('header_color', None)),
            row=1, col=1
        )

        fig0.add_trace(
            go.Bar(x=df['overall'], y=df['vote'], name= "Vote Disbrution Across Ratings",marker_color=self.plotly_table_styles.get('row_odd_color', None)),
            row=1, col=2
        )

        fig0.update_layout(width=900,height=450,title_text="Ratings vs. Votes for Most Popular Item",template="plotly_white")

        fig0.update_xaxes(title_text="Ratings")
        fig0.update_yaxes(title_text="Votes")

        fig0.show()

        # fig0 = px.scatter(df, x="overall", y="vote", title="Rating/Vote Correlation" ,width=400,height=400, render_mode="webgl")
        # fig0.show()

        # fig_bar = px.bar(df, x="overall", y="vote", title="Vote Disbrution Across Ratings", width=400,height=400)
        # fig_bar.show()

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scattergl(x=df['reviewerID'], y=df['overall'], name="Ratings", mode='markers',marker_color=self.plotly_table_styles.get('header_color', None)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scattergl(x=df['reviewerID'], y=df['vote'], name="Votes", mode='markers',marker_color='rgb(220, 0, 0)'),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Ratings and Votes Concentration for Most Popular Item",
            template="plotly_white"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Reviewer IDs")

        # Set y-axes titles
        fig.update_yaxes(title_text="Ratings", secondary_y=False)
        fig.update_yaxes(title_text="Votes", secondary_y=True)

        fig.show()
        
    def vis_prediction(self, data):
        df = data.select(data['overall'], data['prediction']).toPandas()
        print("\nPlotting visualizations...")

        dark_blue = self.plotly_table_styles.get('header_color', None)
        light_blue = self.plotly_table_styles.get('row_odd_color', None)

        colors_cor = [dark_blue]*len(df)
        fig = px.scatter(df, x="overall", y="prediction", width=400, height=400,render_mode='webgl', color_discrete_sequence=colors_cor, template = "plotly_white")
        
        fig.update_layout(
            title={
                'text': "Predicted vs. Actual Ratings Correlation"},
            xaxis_title="Actual",
            yaxis_title="Predicted")
        
        fig.show()

        colors = [[dark_blue,light_blue]*len(df)]

        df['error'] = df['prediction'] - df['overall']
        fig2 = px.histogram(df, x="error",template="plotly_white",color_discrete_sequence=colors)
        fig2.update_layout(
            bargap=0.1,
            title={
                'text': "ALS Prediction Error Distribution",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title="Error",
            yaxis_title="Prediction Count")
        fig2.show()

    def vis_timeseries(self, data, spark,rows):
        data.createOrReplaceTempView('TBL_RATING')
        top_items = spark.sql('''SELECT count(ASIN) AS `Review Count`,
                                 asin AS `ASIN`
                                 FROM TBL_RATING
                                 GROUP BY ASIN
                                 ORDER BY `Review Count` DESC''')

        print(f'\n\nShowing the {rows} most reviewed items for the dataset.', '\n')
        #top_items.show(n=k)
        ti = top_items.toPandas().head(rows)
        #display(ti[0:rows])

        header_list = ["Review Count", "ASIN"]

        fig_table = go.Figure(data=[go.Table(
            columnwidth=[50,50],
            header=dict(
                    values=header_list,
                    fill_color=self.plotly_table_styles.get('header_color', None),
                    align='left',
                    font=dict(color=self.plotly_table_styles.get('font_header_color', None), 
                              size=self.plotly_table_styles.get('font_header_size', None))
                    ),
            cells=dict(values=[ti['Review Count'], 
                               ti.ASIN],
                    fill_color = [[self.plotly_table_styles.get('row_odd_color', None),self.plotly_table_styles.get('row_even_color', None)]*rows],
                    align='left',
                    font=dict(color=self.plotly_table_styles.get('font_cell_color', None), size=self.plotly_table_styles.get('font_cell_size', None))))
        ])

        fig_table.update_layout(
            margin=dict(l=370, r=370, t=0, b=0),
        )

        fig_table.show()

        # Note: 'Most Popular Item' in the below query really refers to the 'Review Count' of the most popular item.
        #       Using 'Most Popular Item' alias for simplicity for plot labeling purposes
        data.createOrReplaceTempView('TBL_RATING_TIMESERIES')
        df = spark.sql('''SELECT count(ASIN) as `Review Count`, unixReviewTime AS `Review Date`  FROM TBL_RATING_TIMESERIES WHERE asin = 
        (SELECT asin AS `itemID`
                FROM TBL_RATING_TIMESERIES
                GROUP BY ASIN
                ORDER BY count(ASIN) DESC LIMIT 1) 
        GROUP BY `Review Date` ORDER BY `Review Date` ''')

        print(f'\n\nShowing the popularity over time of the most-reviewed item of the dataset...', '\n')

        # Convert milliseconds to date

        df_pandas = df.toPandas()

        # Convert date string to pyspark date type
        df_pandas['Review Date'] = pandas.to_datetime(df_pandas['Review Date'], unit='s')

        blue = self.plotly_table_styles.get('header_color', None)
        colors = ["rgb(49, 130, 189)"]*len(df_pandas)
        ######
        # fig = px.bar(df, 
        #      x='x', y='y', 
        #      color_discrete_sequence =['green']*len(df),
        #      title=title,
        #      labels={'x': 'Some X', 'y':'Some Y'})
        #############
        # Display ploty time series line graph
        fig = px.line(df_pandas, x="Review Date", y="Review Count", title='Popularity Over Time for the Most Popular Item',color_discrete_sequence = colors, template ="plotly_white")
        fig.show()
