import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ts_cross_validation(X, model, initial, horizon, fitted=False, plot_cv=True):
    """
    check the cross validation error of a model based on defined time window, and plot the prediction
    and actual values
    parameters:
    X (df): training data with date as index
    initial (int): number of days that will be used for training
    horizon (int): number of days that will be used for validating during cross validation
    fitted (bool): if the model is fit with training data. 
    return:
    cross validation errors in rmse
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    cv_errors = []
    # get number of n folds based on setting
    train_size = X.shape[0]
    folds = (train_size - initial) // horizon

    # reset the index as number for easy access later
    X = X.reset_index()

    # create time series split
    tscv = TimeSeriesSplit(n_splits=int(folds), test_size=horizon)

    # seperate initial training data with cv data
    initial_index = np.arange(initial)
    cv_set = X.loc[initial:]

    cnt = 0  # count nth fold

    for train_index, test_index in tscv.split(cv_set):

        cv_train_idx, cv_test_idx = (
            np.append(initial_index, initial + train_index),
            initial + test_index,
        )
        train_set, test_set = X.loc[cv_train_idx,], X.loc[cv_test_idx,]

        cnt += 1

        print(
            "Train:{}-{}, size={}".format(
                cv_train_idx.min(), cv_train_idx.max(), cv_train_idx.shape[0]
            ),
            "\n"
            "Test:{}-{},  size={}".format(
                cv_test_idx.min(), cv_test_idx.max(), cv_test_idx.shape[0]
            ),
        )

        if not fitted:
            model.fit(train_set)
        try:
            pred = model.predict(n=len(cv_test_idx))
        except:
            pred = model.forecast(len(cv_test_idx))
        pred_error = np.sqrt(mean_squared_error(test_set.loc[:, "volume"], pred))
        cv_errors.append(pred_error)

        if plot_cv:
            # plotting cross validation error
            # plot train and test

            plot_x = np.concatenate((cv_train_idx, cv_test_idx))
            plot_y = np.concatenate(
                (train_set.loc[:, "volume"], test_set.loc[:, "volume"])
            )

            plt.plot(plot_x, plot_y, color="b", alpha=0.3)

            # plot predicted values
            plt.plot(cv_test_idx, pred, color="r")
            plt.legend(["actual", "pred"])

        print("CV_error RMSE:{:.2f}".format(pred_error))

    return cv_errors


def ts_split(X, initial, horizon):
    """
    splits time series data and returns an iterable of splitted train, and test index
    """
    from sklearn.model_selection import TimeSeriesSplit

    # get number of n folds based on setting
    train_size = X.shape[0]
    folds = (train_size - initial) // horizon

    # reset the index as number, so that we can locate them by sequence later
    X = X.reset_index()

    # create time series split
    tscv = TimeSeriesSplit(n_splits=int(folds), test_size=horizon)

    # seperate initial training data with cv data
    initial_index = np.arange(initial)
    cv_set = X.loc[initial:]

    cnt = 0  # count nth fold

    for train_index, test_index in tscv.split(cv_set):

        cv_train_idx, cv_test_idx = (
            np.append(initial_index, initial + train_index),
            initial + test_index,
        )
        print(
            "Train:{}-{}, size={}".format(
                cv_train_idx.min(), cv_train_idx.max(), cv_train_idx.shape[0]
            ),
            "\n"
            "Test:{}-{},  size={}".format(
                cv_test_idx.min(), cv_test_idx.max(), cv_test_idx.shape[0]
            ),
        )
        yield cv_train_idx, cv_test_idx


def set_outliers_as_null(df):
    """
    detect outliers using 1.5 IQR rule and set outliers as null
    Outliers are any data below Q1 - 1.5 IQR and any data above Q3 + 1.5 IQR
    
    parameters: 
    df: a pandas dataframe with date as index
    
    returns:
    new_df_volume: a pandas dataframe with outliers as null
    outliers: a pandas dataframe with only outliers
    """
    years = df.index.year.unique()
    new_df_volume = pd.DataFrame()
    outliers = pd.DataFrame()
    for year in years:
        ev_year = df.loc[
            df.index.year == year,
        ]
        q1 = np.percentile(ev_year, 25)  # ev_year.quantile([.25]).values
        q3 = np.percentile(ev_year, 75)  # ev_year.quantile([.75]).values
        iqr = q3 - q1
        low_b = q1 - 1.5 * iqr
        upper_b = q3 + 1.5 * iqr
        outlier_each_year = ev_year.loc[
            (ev_year.volume < low_b) | (ev_year.volume > upper_b),
        ]
        outliers = pd.concat([outliers, outlier_each_year])

        ev_year.loc[
            (ev_year.volume < low_b) | (ev_year.volume > upper_b), "volume"
        ] = np.nan
        new_df_volume = pd.concat([new_df_volume, ev_year])
    return new_df_volume, outliers
