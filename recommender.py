# recommender.py

# Purpose: Runs recommender system based on ALS and prints associated outputs

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()

df = spark.read.json("Musical_Instruments_5.json")
df.show(100,truncate=True)

nd=df.select(df['asin'],df['overall'],df['reviewerID'])
nd.show()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(nd.columns)-set(['overall'])) ]
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(nd).transform(nd)
transformed.show()

(training,test)=transformed.randomSplit([0.8, 0.2])

als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="reviewerID_index",itemCol="asin_index",ratingCol="overall",coldStartStrategy="drop",nonnegative=True)
model=als.fit(training)

evaluator=RegressionEvaluator(metricName="rmse",labelCol="overall",predictionCol="prediction")
predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)
print("RMSE="+str(rmse))
predictions.show()

user_recs=model.recommendForAllUsers(20).show(10)

import pandas as pd
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