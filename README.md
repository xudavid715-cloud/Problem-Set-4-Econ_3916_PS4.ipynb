Regularization: Buying Bias to Sell Variance
Objective
I compared ordinary least squares (OLS), LASSO, and Ridge on a high-dimensional simulated data set to understand how penalizing coefficients changes prediction error and variable selection.
Methodology
- I generated 150 observations with 200 predictors, only five of which really affected the outcome. I fitted the models on 105 training observations and evaluated them on held-out test data.
- I used a standardized pipeline for LASSO and Ridge and compared training error, test error, and the number of nonzero coefficients.
- I plotted the LASSO coefficient path over 60 penalties to see which variables entered first.
- I used five-fold cross-validation with a new scaler inside each training fold to select the penalty, then checked the selected predictors against the simulated truth.
- I tested what happened when two true predictors were almost duplicates. I also compared Elastic Net and explored what happens if one feature is multiplied by 1,000 without standardizing.
- I wrote two AI prompts for an explicit cross-validation loop, checked both versions against the lab, and corrected the fold settings that caused the first version to disagree.
Key Findings
- OLS overfit: Its training MSE was 0.0000, but its test MSE was 12.9587. Ridge's test MSE was 10.1128, while LASSO's was much lower at 1.4219.
- LASSO found the strong signals early: All five true predictors were in the coefficient path by step 6; the first noise variable did not enter until step 17.
- Cross-validation chose a penalty: The five-fold loop with within-fold standardization selected lambda = 0.1995, with CV MSE = 1.5502.
- Selection was useful, not perfect: LASSO found all 5 of the 5 real predictors, but also selected 11 false positives.
- Correlation changed the coefficients: For near-duplicate variables 0 and 5, LASSO assigned 4.462 and 0.000, while Ridge assigned 1.618 and 1.609. Ridge shared the weight; LASSO kept one variable.
- Elastic Net did not improve test error here: It selected l1_ratio = 1.0, matching LASSO at test MSE 1.1055. Ridge's test MSE on the correlated design was 10.8464, although it kept both variables.
- Scaling affected the results: After multiplying one true predictor by 1,000, unscaled LASSO had test MSE 1.5718 versus 1.4252 after scaling at the same penalty. Even OLS's test MSE changed when I rescaled a column in this underdetermined setting.
- The AI-generated loop needed checking: Round 1 had unshuffled folds; its full-grid CV MSE was 1.6851 instead of 1.5502. After I required shuffle=True, random_state=42, round 2 matched both the selected lambda and CV MSE on both tested penalty grids.
