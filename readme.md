Develop time series cross validation framework in utils.py and use the framework to compare model performance on predicting CO2 volume in time_series_model_selection.ipynb

Models compared:
- naive forecasting model
- moving average
- holt winter's model
- Facebook prophet
- sarima
- auto-sarima


Conclusions:
- sarima has lowest Root Mean Squared Error (rmse)
- auto-sarima takes much longer time to search for best p, d, q and results are not necessarily the best, maybe due to over-fitting
- Facebook Prophet has moderately high performance and can be used in more complex forecasting situation. It allows to add customized Holiday and special events.





