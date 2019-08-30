"""
PPECTR exercise

@author: Mate Kormos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def estimate_beta(vY, mX):
    """
        OLS estimator of linear model coefficients.

        Intputs
        =======
        vY: 2-dimensional numpy array, column vector.
            Endogenous variable.
        mX: 2-dimensional numpy array.
            Exogenous variables.

        Returns
        =======
        vBetahat: estimated oefficient vector
    """
    
    mA = mX.T @ mX
    vB = mX.T @ vY
    vBetahat = np.linalg.lstsq(mA, vB, rcond=None)[0]  # solve the normal equation
    return vBetahat



def predict_y(mX, vBetahat):
    """
        Predict endogenous variable from linear model.

        Intputs
        =======
        mX: 2-dimensional numpy array.
            Exogenous variables.
        vBetahat: 2-dimensional numpy array, column vector.
            OLS estimate of linear model coefficient.

        Returns
        =======
        vYhat: 2-dimensional numpy array, column vector.
            Predicted endogenous variable.
    """
    
    vYhat = mX @ vBetahat
    return vYhat


def transform_data(dfRawdata):
    """
        Transform the time series data, generate regressors.

    Inputs
    ======
    dfRawdata: DataFrame.
               Raw data read from csv
    Returns
    =======
    DataFrame of transformed variables
    """
    # convert to datetime
    dfRawdata['Period'] = pd.to_datetime(dfRawdata['Period'], format='%Y/%m')
    # drop old observations
    dfShort = dfRawdata[dfRawdata['Period'] >= '1958-01-01']
    # set datetime as index
    dfShort = dfShort.set_index(pd.DatetimeIndex(dfShort.Period))
    dfShort = dfShort.drop(['Period'], axis=1)

    # percentage inflation
    dfShort['pct_inflation'] = 100 * np.log(dfShort.SA0).diff()
    # adjust number of observation for differencing
    dfShort = dfShort[1:]
    # generate regressors
    gen_x(dfShort)
    
    return dfShort


def get_variables(df):
    """
        Returns the endogenous and exogenous variables from transformed data.

    Inputs
    ======
    df: DataFrame.
        Transformed data

    Returns
    =======
    (vY, mX, lX_names) tuple where lX_names is the list of regressor names
    """
    # obtain y
    vY = df.pct_inflation
    # obtain x
    lX_names = list(df)
    lX_names.remove('pct_inflation')
    lX_names.remove('SA0')
    mX = df[lX_names].values
    print(df.head())
    print(mX)
    return (vY, mX, lX_names)


def gen_x(df):
    """
        Generates regressors.

    Inputs
    ======
    df: DataFrame.
        Data.

    Returns
    =======
    df: DataFrame with generated regressors included
    """
    # generate month dummies
    for m in range(1, 12):
        df['M%d' % m] = list(map(int, df.index.month == m))

    n = len(df.index)
    df['Constant'] = np.ones((n, 1))
    return df


def plot_results(df, vYhat):
    """
        Plots the original series and the predicted data.
    """
    # plot level
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df.SA0, label='SA0')
    # plot pct change
    plt.subplot(2,1,2)
    plt.plot(df.index, df.pct_inflation, label='y')
    plt.plot(df.index, vYhat, label='y_hat')
    plt.legend()
    plt.show()


def main():
    # magic numbers
    sData = './data/sa0_180827.csv'

    # read data
    dfRawdata = pd.read_csv(sData)
    # transform data
    dfTransformedData = transform_data(dfRawdata)
    # write data
    dfTransformedData.to_csv('./data/sa0_180827_transformed.csv')
    # get variables
    (vY, mX, lX_names) = get_variables(dfTransformedData)

    # ols
    vBetahat = estimate_beta(vY, mX)
    vYhat = predict_y(mX, vBetahat)

    # output
    print("Estimated coefficients: ")
    for name, betahat in zip(lX_names, vBetahat):
        print(name, ':    %.5f' %betahat)

    # plot
    plot_results(dfTransformedData, vYhat)


if __name__ == '__main__':
    main()
