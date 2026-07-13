Feature Engineering Choices
* House_Age: Constructed by subtracting `Year_Built` from the current processing year (2026). Real estate trends show asset deprecation or premium vintage valuation correlates strongly with asset age rather than nominal calendar years.
* SqFt_per_Room: Derived by dividing total space by the aggregate of utility rooms. This metric defines the spatial efficiency and density of the layout design.

Regularization Insights (Ridge vs. Lasso)
* Lasso (L1 Regularization) applies a absolute penalty to model weights, enabling automated feature selection by driving non-impactful structural coefficients completely down to zero.
* Ridge (L2 Regularization) applies a squared magnitude penalty to coefficients, which shrinks weights uniformly to prevent overfitting caused by high feature collinearity (such as room counts interacting with overall square footage).

Residual Diagnostics Evaluation
* Normality of Residuals: Our distribution histogram checks whether error dispersion mirrors a zero-centered Gaussian curve. Regular deviations indicate an unbiased variance distribution.
* Homoscedasticity check: The scatter plot mapping Predicted Value vs. Residuals establishes whether error bounds remain uniform across all pricing ranges. A widening funnel pattern would indicate heteroscedasticity, signaling the need for logarithmic scaling transformations on target price features.
