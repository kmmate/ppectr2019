"""
PPECTR Exercise

@author: Mate Kormos
"""
import numpy as np

def main():
    # magic numbers
    iN = 10
    iK = 10
    # preallocate
    mX = np.zeros((iN, iK))
    # fill it with double loop (very slow solution)
    for i in range(1, iN + 1):
        for j in range(1, iK + 1):
            mX[i - 1, j - 1] = i * j  # -1 to adjust for 0-indexing
    print("mX = \n", mX, "\n")

if __name__ == '__main__':
    main()
