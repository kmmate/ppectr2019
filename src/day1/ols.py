"""
Day 1 exercises: OLS
"""
import numpy as np

from linear_algebra import linearsolver

def generate_data(iN, dSigma, vBeta):
    """
        Generates data from linear model.

        Data generating process: y = X*beta + sigma * epsilon,
        where X[i] = [1, u1, u2], i=0,...,n-1 and u1, u2 ~iidUniform[0,1],
        epsilon_i ~ iidN(0, 1).

        Inputs
        ======
        iN: Integer.
            Sample size
        dSigma: Float.
                See data generating process.
        vBeta: 2-dimensional numpy array, column vector.
               See data generating process.

        Returns
        =======
        diData: dictionary with generated data. Keys: x, y,
                and disturbance=sigma*epsilon. values: generated data
    """
    iK = np.size(vBeta)  

    mX = np.ones((iN, iK))  # pre-allocate x
    mX[:, 1:3] = np.random.rand(iN, 2)  # generate x
    vEpsilon = np.random.randn(iN, 1)
    vDisturbance = dSigma * vEpsilon
    vY = mX @ vBeta + vDisturbance

    diData = {'x': mX, 'y': vY, 'disturbance': vDisturbance}
    return diData


def EstimateMM(vY, mX):
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
    mXtXinv = np.linalg.inv(mX.T @ mX)
    vBetahat = mXtXinv @ mX.T @ vY
    return vBetahat


def EstimateEB(vY, mX):
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
    vBetahat = linearsolver(mA, vB)  # solve the normal equation
    return vBetahat


def EstimatePF(vY, mX):
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
    vBetahat = np.linalg.lstsq(mA, vB)[0]  # solve the normal equation
    return vBetahat

def main():
    iN = 20  # sample size
    iK = 3  # no. of variables, including constant
    dSigma = 0.25  # std of epsilon
    vBeta = np.array([1, 2, 3]).reshape(iK,1)  # coeff vector

    diData = generate_data(iN, dSigma, vBeta)
    mX = diData['x']
    vY = diData['y']
    vDisturbance = diData['disturbance']  # sigma * epsilon

    vBetahat_MM = EstimateMM(vY, mX)
    vBetahat_EB = EstimateEB(vY, mX)
    vBetahat_PF = EstimatePF(vY, mX)
    print("MM estimator: \n     ", vBetahat_MM, '\n')
    print("EB estimator: \n     ", vBetahat_EB.reshape(3,1), '\n')
    print("PF estimator: \n     ", vBetahat_PF, '\n')
    

if __name__ == '__main__':
    main()
