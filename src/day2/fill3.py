"""
PPECTR Exercise

@author: Mate Kormos
"""


import numpy as np

def main():
    # magic numbers
    iN = 10
    iK = 10
    # list comprehension: fastest solution
    mX = [[i*j for j in range(1, iK + 1)] for i in range(1, iN + 1)]
    mX = np.array(mX)
    print("mX = \n", mX, "\n")

if __name__ == '__main__':
    main()
