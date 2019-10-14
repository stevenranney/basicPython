import boto3
import s3fs
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

import botocore

bucket = 'sranney-modeling-data'
file_name = "aws_data.csv"

s3 = boto3.client(
    's3', 
    aws_access_key_id = '',
    aws_secret_access_key = ''
) 
# 's3' is a key word. create connection to S3 using default config and all buckets within S3

obj = s3.get_object(Bucket = bucket, 
                    Key= file_name) 
# get object and file (key) from bucket

failure = pd.read_csv(obj['Body']) # 'Body' is a key word

plt.matshow(failure[failure.columns[3:]].corr())
plt.xticks(range(len(failure.columns)), failure.columns)
plt.yticks(range(len(failure.columns)), failure.columns)
plt.colorbar()
plt.show()