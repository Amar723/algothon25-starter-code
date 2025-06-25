
import numpy as np
from sklearn.linear_model import LinearRegression

nInst = 50  # number of instruments
lookback = 20  # number of days to look back
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    (nInst, nt) = prcSoFar.shape
    lookback = 20
    threshold = 0.01  # 1% expected return

    if nt < lookback + 1:
        return np.zeros(nInst, dtype=int)

    positions = np.zeros(nInst, dtype=int)

    for inst in range(nInst):
        prices = prcSoFar[inst, -lookback-1:]  # include 1 extra day to calculate returns
        log_returns = np.diff(np.log(prices))  # 20 returns from 21 prices
        X = np.arange(lookback).reshape(-1, 1)
        y = log_returns

        model = LinearRegression().fit(X, y)
        next_ret = model.predict(np.array([[lookback]]))[0]
        today_price = prcSoFar[inst, -1]

        print(f"Inst {inst} | Predicted return: {next_ret:.4f} | Price: {today_price:.2f}")

        if next_ret > threshold:
            pos = 100
        elif next_ret < -threshold:
            pos = -100
        else:
            pos = 0

        max_position = int(10000 // today_price)
        positions[inst] = np.clip(pos, -max_position, max_position)

    return positions



if __name__ == "__main__":
    # Load your .txt file
    prices = np.loadtxt('prices.txt')  # Adjust delimiter if space-separated
    positions = getMyPosition(prices)
    print("Final positions:\n", positions)
