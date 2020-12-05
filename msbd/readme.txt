1. used language: python 3.7

2. required pacakge:
- scikit-learn==0.23.1
- pandas==1.0.3
- tqdm==4.46.0

3. Please put the code and data (train.csv and test.csv) in the same folder, then run `python msbd5001.py`, the result will be stored at test.csv

In the code, it uses 2017's full data to generate train dataset. Specially, it use a sliding window to add past 30 days' speed information at the same time as feature. In addition, the speed of the prior 30 time periods of the current time is also added as a feature. 
Then, it use the datasets to train GradientBoostingRegressor. When in prediction, program loops the test.csv, if the row's speed is nan then program will add features for the row as above, and use GradientBoostingRegressor to predict the speed.