# In this course project, we implemented:

1. No-arbitrage checks: For each day, we test whether market prices of European calls and puts satisfy static no-arbitrage conditions.

2. Point picker: Given an expiry date and strike price, we select the nearest available market quote for subsequent calibration.

3. Heston calibration: We calibrate the Heston model to fit that day’s option prices, compare model-generated data with market data in terms of both price and implied volatility errors, and produce volatility-surface plots.

4. Multi-month study and backtest: We perform daily calibrations over four consecutive months and further assess the quality of the Heston calibration from a delta-hedging backtesting perspective.
