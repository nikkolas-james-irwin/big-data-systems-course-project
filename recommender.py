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
#   Python Recommender File: Python file for driving the core recommender system logic for CARS.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()

df = spark.read.json("Musical_Instruments_5.json")
df.show(100,truncate=True)

nd=df.select(df['asin'],df['overall'],df['reviewerID'])
nd.show()

indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(nd.columns)-set(['overall'])) ]
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(nd).transform(nd)
transformed.show()

(training,test)=transformed.randomSplit([0.8, 0.2])

als=ALS(maxIter=5, \
        regParam=0.09, \ 
        rank=25, \
        userCol="reviewerID_index", \
        itemCol="asin_index", \
        ratingCol="overall", \
        coldStartStrategy="drop", \
        nonnegative=True)
model=als.fit(training)

evaluator=RegressionEvaluator(metricName="rmse",labelCol="overall",predictionCol="prediction")
predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)
print("RMSE="+str(rmse))
predictions.show()

user_recs=model.recommendForAllUsers(20).show(10)

recs=model.recommendForAllUsers(10).toPandas()
nrecs=recs.recommendations.apply(pd.Series) \
            .merge(recs, right_index = True, left_index = True) \
            .drop(["recommendations"], axis = 1) \
            .melt(id_vars = ['reviewerID_index'], value_name = "recommendation") \
            .drop("variable", axis = 1) \
            .dropna() 
nrecs=nrecs.sort_values('reviewerID_index')
nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['reviewerID_index']], axis = 1)
nrecs.columns = [
        
        'ProductID_index',
        'Rating',
        'UserID_index'
       
     ]
md=transformed.select(transformed['reviewerID'],transformed['reviewerID_index'],transformed['asin'],transformed['asin_index'])
md=md.toPandas()
dict1 =dict(zip(md['reviewerID_index'],md['reviewerID']))
dict2=dict(zip(md['asin_index'],md['asin']))
nrecs['reviewerID']=nrecs['UserID_index'].map(dict1)
nrecs['asin']=nrecs['ProductID_index'].map(dict2)
nrecs=nrecs.sort_values('reviewerID')
nrecs.reset_index(drop=True, inplace=True)
new=nrecs[['reviewerID','asin','Rating']]
new['recommendations'] = list(zip(new.asin, new.Rating))
res=new[['reviewerID','recommendations']]  
res_new=res['recommendations'].groupby([res.reviewerID]).apply(list).reset_index()
print(res_new)
