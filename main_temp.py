import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random_forest_regressor import RandomForestRegressor
from mylogg import mylogger
import logging

logger = mylogger("main", logging.INFO)


def load_daily_min_temperatures():
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/'
                    'Datasets/master/daily-min-temperatures.csv')
    # Minimum Daily Temperatures Dataset over 10 years (1981-1990)
    # in Melbourne, Australia. The units are in degrees Celsius.
    # These are the features to regress:
    day = pd.DatetimeIndex(df.Date).day.to_numpy() # 1...31
    month = pd.DatetimeIndex(df.Date).month.to_numpy() # 1...12
    year = pd.DatetimeIndex(df.Date).year.to_numpy() # 1981...1999
    X = np.vstack([day, month, year]).T # np array of 3 columns
    y = df.Temp.to_numpy()
    return X, y

def test_regression(last_years_test=1):
    X, y = load_daily_min_temperatures()
    logger.info("dataset downloaded")
    plt.figure()
    plt.plot(y,'.-')
    plt.xlabel('day in 10 years'), plt.ylabel('min. daily temperature')
    idx = last_years_test*365
    Xtrain = X[:-idx,:] # first years
    Xtest = X[-idx:]
    ytrain = y[:-idx] # last years
    ytest = y[-idx:]
    max_depth=10
    min_size=5
    ratio_samples=0.5
    num_trees=50
    num_random_features=2
    criterion = 'sse'
    logger.info("hyperparameters setted ")

    randomforest = RandomForestRegressor(max_depth, min_size, ratio_samples,num_trees, num_random_features, criterion,do_multiprocessing=True,extra_trees='extra_trees')

    randomforest.fit(Xtrain,ytrain)
    ypred = randomforest.predict(Xtest)

    plt.figure()
    x = range(idx)
    for t, y1, y2 in zip(x, ytest, ypred):
        plt.plot([t, t], [y1, y2],'k-')
    plt.plot([x[0], x[0]],[ytest[0], ypred[0]],'k-', label='error')
    plt.plot(x, ytest,'g.', label='test')
    plt.plot(x, ypred,'y.', label='prediction')
    plt.xlabel('day in last {} years'.format(last_years_test))
    plt.ylabel('min. daily temperature')
    plt.legend()
    errors = ytest - ypred
    rmse = np.sqrt(np.mean(errors**2))
    plt.title('root mean square error : {:.3f}'.format(rmse))
    plt.show()

if __name__ == '__main__':
    test_regression()