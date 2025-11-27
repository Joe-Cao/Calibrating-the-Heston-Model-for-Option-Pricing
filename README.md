# Course Project: Option Pricing and Calibration with the Heston Model

In this course project, we implemented

## Data preprocessing

Here, the source data consists of a set of ZIP files named Optsum_YYYY-MM-DD. Each archive contains a CSV file with the same name. For this project, we use files from Optsum_2017-01-03.zip through Optsum_2017-05-31.zip. 

To simplify the calibration objective, we use pick_points_bs.py, which selects only option quotes at or near the specified maturities and strikes. 

merge_optsum_zips.py is used to combine all files named Optsum_YYYY-MM-DD; each archive contains option quotes for a single quoted date.

## Arbitrage check

For the source data, we checked the following discrete no-arbitrage conditions: bid ≤ ask; vertical spreads; butterfly convexity; calendar monotonicity; and put–call parity.

We identified regions that potentially violate these conditions (and thus admit arbitrage) and computed their proportions. See diagnostics_heston_qc.py for details.

## Heston Calibration

We calibrate the Heston model to fit that day’s option prices, compare model-generated data with market data in terms of both price and implied volatility errors, and produce volatility-surface plots.

calibrate_heston_bs.py demonstrates how to calibrate the Heston model to multi-day option data, producing a new set of model parameters for each day.

draw_IV_surface.py shows how to plot the implied-volatility surface from a given day’s market data and from the calibrated parameters, and how to compare the two surfaces by computing their errors.

## Multi-month study and backtest

From the daily Heston calibrations over four consecutive months, we further evaluate calibration quality through a delta-hedging backtest.

delta_hedging_backtest_tailored.py demonstrates how to build a delta-hedging backtest based on the calibrated Heston model, and how to decompose each hedging strategy’s PnL into components—TH (time decay), UM–delta-hedged (underlying move after delta hedging), SM (smile/vol effect), and ME (model error)—and then compute the daily sums for each component.


2. For each day, we test whether market prices of European calls and puts satisfy static no-arbitrage conditions.

3. Point picker: Given an expiry date and strike price, we select the nearest available market quote for subsequent calibration.

4. Heston calibration: We calibrate the Heston model to fit that day’s option prices, compare model-generated data with market data in terms of both price and implied volatility errors, and produce volatility-surface plots.

5. Multi-month study and backtest: We perform daily calibrations over four consecutive months and further assess the quality of the Heston calibration from a delta-hedging backtesting perspective.
