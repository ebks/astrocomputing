---
# Appendix C
# Review of Essential Statistics for Astronomers
---
![imagem](imagem.png)

*This appendix provides a concise review of fundamental statistical concepts and methods frequently encountered in the analysis and interpretation of astronomical data. While not intended as a comprehensive statistics course, it aims to refresh key ideas discussed implicitly or explicitly throughout the book, such as probability distributions, parameter estimation, hypothesis testing, and correlation analysis. A solid grasp of these statistical foundations is crucial for drawing robust scientific conclusions from inherently noisy and often complex astronomical measurements (Feigelson & Babu, 2012; Wall & Jenkins, 2012; Gregory, 2005).*

---

**C.1 Probability Distributions**

Probability distributions describe the likelihood of different outcomes for a random variable. Understanding common distributions is essential for modeling data, characterizing uncertainties, and performing statistical inference.

*   **Gaussian (Normal) Distribution:** $P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
    *   Defined by mean $\mu$ (center) and standard deviation $\sigma$ (width). Variance is $\sigma^2$.
    *   Ubiquitous due to the Central Limit Theorem, which states that the sum of many independent random variables tends towards a Gaussian distribution.
    *   Often used to model measurement errors in astronomy, assuming they result from the sum of many small, random effects.
    *   Key property: ~68.3% of values fall within $\mu \pm 1\sigma$, ~95.4% within $\mu \pm 2\sigma$, and ~99.7% within $\mu \pm 3\sigma$.
    *   Implemented in `scipy.stats.norm`.

*   **Poisson Distribution:** $P(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$ (for integer $k \ge 0$)
    *   Describes the probability of observing $k$ discrete events occurring independently in a fixed interval (of time, space, etc.) when the average rate of occurrence is $\lambda$.
    *   Fundamental for modeling **photon counting statistics** (shot noise) from detectors like CCDs or photon-counting devices, where $k$ is the number of detected photons and $\lambda$ is the expected number.
    *   Mean and Variance: Both equal to $\lambda$. The standard deviation is $\sigma = \sqrt{\lambda}$. For large $\lambda$ (> ~20), the Poisson distribution approximates a Gaussian distribution with $\mu = \lambda$ and $\sigma = \sqrt{\lambda}$.
    *   Implemented in `scipy.stats.poisson`.

*   **Chi-Squared ($\chi^2$) Distribution:** $P(x|\nu)$
    *   Describes the distribution of a sum of squares of $\nu$ independent, standard normal random variables (mean 0, variance 1). $\nu$ is the **degrees of freedom**.
    *   Mean is $\nu$, Variance is $2\nu$.
    *   Crucial for **goodness-of-fit testing** in least-squares methods (Section 12.2.2). The minimized $\chi^2_{min}$ statistic approximately follows a $\chi^2$ distribution with $\nu = N_{data} - N_{params}$ degrees of freedom if the model is correct and errors are Gaussian.
    *   The reduced chi-squared $\chi^2_\nu = \chi^2 / \nu$ is expected to be around 1 for a good fit.
    *   Implemented in `scipy.stats.chi2`.

*   **Uniform Distribution:** $P(x|a, b) = 1/(b-a)$ for $a \le x \le b$, and 0 otherwise.
    *   Assigns equal probability to all values within a defined range $[a, b]$.
    *   Often used as an **uninformative prior** in Bayesian inference (Section 12.2.4) when parameter values are assumed equally likely within a certain range before considering the data.
    *   Implemented in `scipy.stats.uniform`.

*   **Log-Normal Distribution:** A variable $x$ follows a log-normal distribution if $\ln(x)$ is normally distributed. Often used for quantities that are strictly positive and have skewed distributions (e.g., possibly galaxy luminosities, stellar flare energies).

**C.2 Descriptive Statistics**

Descriptive statistics summarize the main features of a dataset.

*   **Measures of Central Tendency:**
    *   **Mean ($\bar{x}$ or $\mu$):** The arithmetic average ($\sum x_i / N$). Sensitive to outliers. (`numpy.mean`)
    *   **Median:** The middle value when data is sorted (50th percentile). Robust against outliers. (`numpy.median`)
    *   **Mode:** The most frequently occurring value. (Less common for continuous data, `scipy.stats.mode`)
*   **Measures of Dispersion (Spread):**
    *   **Variance ($\sigma^2$):** Average squared deviation from the mean ($\sum (x_i - \bar{x})^2 / (N-1)$ for sample variance). Units are square of data units. (`numpy.var`, `ddof=1` for sample variance)
    *   **Standard Deviation ($\sigma$):** Square root of the variance. Has the same units as the data. Represents typical deviation from the mean. (`numpy.std`, `ddof=1` for sample standard deviation)
    *   **Median Absolute Deviation (MAD):** Median of the absolute deviations from the data's median. Robust measure of spread. $\sigma \approx 1.4826 \times MAD$ for Gaussian data. (`astropy.stats.median_absolute_deviation`, `scipy.stats.median_abs_deviation`)
    *   **Interquartile Range (IQR):** Difference between the 75th percentile (Q3) and 25th percentile (Q1). Robust measure of spread containing the central 50% of data. (`scipy.stats.iqr`)
*   **Measures of Shape:**
    *   **Skewness:** Measures the asymmetry of the distribution. Positive skew indicates a longer tail to the right, negative skew a longer tail to the left. Zero for symmetric distributions like Gaussian. (`scipy.stats.skew`)
    *   **Kurtosis:** Measures the "tailedness" or peakedness of the distribution compared to a Gaussian. Positive kurtosis (leptokurtic) indicates heavier tails and a sharper peak. Negative kurtosis (platykurtic) indicates lighter tails and a flatter peak. Gaussian has kurtosis of 0 (using Fisher's definition, where 3 is subtracted). (`scipy.stats.kurtosis`)

**C.3 Parameter Estimation**

Parameter estimation involves using observed data to determine the values of parameters in a chosen model.

*   **Point Estimation:** Finding a single "best" value for each parameter.
    *   **Method of Moments:** Equating sample moments (mean, variance) to theoretical moments derived from the model distribution. Simple but often not efficient.
    *   **Least Squares Estimation (LSE):** Minimizing the $\chi^2$ statistic (Section 12.2.2). Optimal for Gaussian errors.
    *   **Maximum Likelihood Estimation (MLE):** Maximizing the likelihood function $P(Data | \theta)$ (Section 12.2.3). More general than LSE, applicable to non-Gaussian likelihoods (e.g., Poisson). MLE estimators have desirable asymptotic properties (consistency, efficiency, asymptotic normality).
    *   **Bayesian Estimators:** Using the posterior distribution $P(\theta | Data)$ (Section 12.2.4). Common point estimates include the posterior mean, median, or mode (Maximum A Posteriori - MAP estimate). MAP estimation is equivalent to MLE if the prior is uniform.

*   **Interval Estimation (Confidence/Credible Intervals):** Providing a range within which the true parameter value is likely to lie, quantifying the uncertainty.
    *   **Frequentist Confidence Intervals:** Constructed such that if the experiment were repeated many times, the interval would contain the true parameter value in a specified fraction (e.g., 68%, 95%) of the trials. Often derived from the standard error (from LSE/MLE covariance matrix) assuming asymptotic normality (e.g., $\hat{\theta} \pm 1\sigma_\theta$ for ~68% CI), or from likelihood profiles or bootstrapping. The interpretation is about the interval's long-run performance, not the probability of the true value being inside *this specific* interval.
    *   **Bayesian Credible Intervals (or Regions):** Derived directly from the posterior probability distribution $P(\theta | Data)$. A 95% credible interval is a range $[a, b]$ such that $\int_a^b P(\theta | Data) d\theta = 0.95$. Common choices include the Highest Posterior Density (HPD) interval. The interpretation is direct: there is a 95% probability (given the data and model) that the true parameter value lies within the credible interval. Often obtained from MCMC samples (Section 12.3) using percentiles.

**C.4 Hypothesis Testing**

Hypothesis testing provides a framework for making decisions between competing hypotheses based on observed data.

*   **Null Hypothesis ($H_0$):** A default statement about the system, often representing "no effect" or "no difference" (e.g., "the data are consistent with background noise only," "the mean values of two samples are equal").
*   **Alternative Hypothesis ($H_1$):** The statement contradicting the null hypothesis, often representing the effect the researcher is looking for (e.g., "a signal is present," "the means are different").
*   **Test Statistic:** A quantity calculated from the data whose distribution under the null hypothesis is known or can be approximated (e.g., t-statistic, $\chi^2$ statistic, likelihood ratio).
*   **Significance Level ($\alpha$):** The probability of rejecting the null hypothesis when it is actually true (Type I error), typically set beforehand (e.g., $\alpha = 0.05$ or $0.01$).
*   **P-value:** The probability of observing a test statistic value at least as extreme as the one actually observed, *assuming the null hypothesis is true*.
*   **Decision Rule:** If the p-value is less than the chosen significance level $\alpha$, the null hypothesis $H_0$ is rejected in favor of the alternative hypothesis $H_1$. If $p \ge \alpha$, we "fail to reject" $H_0$ (which does *not* prove $H_0$ is true, only that the evidence against it is insufficient at level $\alpha$).
*   **Common Tests:**
    *   **T-test (`scipy.stats.ttest_ind`, `ttest_rel`, `ttest_1samp`):** Comparing means of one or two samples, assumes normality (robust for large N).
    *   **Chi-Squared Test (`scipy.stats.chisquare`, `chi2_contingency`):** Testing goodness-of-fit (comparing observed frequencies/data to expected model) or independence in contingency tables.
    *   **Kolmogorov-Smirnov (K-S) Test (`scipy.stats.ks_2samp`, `kstest`):** Testing if two samples are drawn from the same underlying distribution, or if one sample is drawn from a specific distribution (non-parametric).
    *   **Likelihood Ratio Test (LRT):** Comparing the maximum likelihood values of two nested models (one being a special case of the other). The statistic $-2 \ln(\mathcal{L}_{null}/\mathcal{L}_{alt})$ approximately follows a $\chi^2$ distribution under $H_0$.

*   **Bayesian Hypothesis Testing (Model Comparison):** Instead of p-values, Bayesian methods compare models using the **Bayes Factor ($B_{10}$)**, the ratio of the evidences (marginal likelihoods) of the two models:
    $B_{10} = \frac{P(D | M_1)}{P(D | M_0)}$
    The Bayes factor quantifies the evidence provided by the data in favor of model $M_1$ over model $M_0$. Interpretation often uses scales like the Jeffreys scale (e.g., $B_{10} > 3$ is "substantial evidence," $B_{10} > 10$ is "strong," $B_{10} > 100$ is "decisive"). Requires calculating the often difficult marginal likelihood (Section 12.2.4), often achieved via nested sampling (Section 12.3).

**C.5 Correlation and Regression**

These methods explore relationships between two or more variables.

*   **Correlation:** Measures the strength and direction of a *linear* association between two variables.
    *   **Pearson Correlation Coefficient ($r$):** Ranges from -1 (perfect negative linear correlation) to +1 (perfect positive linear correlation), with 0 indicating no linear correlation. Sensitive to outliers. (`scipy.stats.pearsonr` returns $r$ and p-value for testing non-correlation).
    *   **Spearman Rank Correlation Coefficient ($\rho$):** Measures the strength and direction of a *monotonic* relationship (variables tend to increase/decrease together, but not necessarily linearly). Calculates Pearson correlation on the *ranks* of the data. Less sensitive to outliers and non-linear monotonic relationships. (`scipy.stats.spearmanr`)
    *   **Kendall's Tau ($\tau$):** Another non-parametric rank correlation coefficient, based on counting concordant and discordant pairs. (`scipy.stats.kendalltau`)
    **Important:** Correlation does not imply causation!

*   **Linear Regression:** Models the relationship between a dependent variable ($y$) and one or more independent variables ($x$) using a linear equation ($y = \beta_0 + \beta_1 x_1 + ... + \epsilon$).
    *   **Simple Linear Regression:** One independent variable ($y = \beta_0 + \beta_1 x$). Aims to find the best-fit slope ($\beta_1$) and intercept ($\beta_0$) usually via least squares. (`scipy.stats.linregress`, `numpy.polyfit(deg=1)`)
    *   **Multiple Linear Regression:** Two or more independent variables. (`sklearn.linear_model.LinearRegression`)
    *   **Assumptions:** Typically assumes linearity, independence of errors, homoscedasticity (constant error variance), and normally distributed errors for standard inference (confidence intervals, p-values on coefficients).
    *   **Evaluation:** $R^2$ (coefficient of determination) measures the proportion of variance in $y$ explained by the model. Residual analysis checks assumptions.

*   **Non-linear Regression:** Fitting models where the relationship between parameters and the dependent variable is non-linear (e.g., Gaussian fits, power laws, exponential decays). Requires iterative optimization algorithms (e.g., `scipy.optimize.curve_fit`, `astropy.modeling.fitting`).

**C.6 Monte Carlo Methods**

Monte Carlo methods use repeated random sampling to obtain numerical results, often used when analytical solutions are intractable.

*   **Error Propagation:** Simulating the effect of input uncertainties on derived quantities. Draw many random samples from the input parameter distributions (e.g., Gaussians based on measured value and error), calculate the derived quantity for each sample, and examine the distribution of the results to estimate its mean and uncertainty.
*   **Integration:** Estimating definite integrals by averaging function values evaluated at random points within the integration domain.
*   **MCMC (Markov Chain Monte Carlo):** Used for Bayesian parameter estimation (Section 12.3) by generating samples from the posterior distribution.
*   **Bootstrapping:** Resampling technique used to estimate uncertainties or confidence intervals for statistics derived from a dataset. Involves repeatedly drawing samples *with replacement* from the original dataset, recalculating the statistic of interest for each bootstrap sample, and examining the distribution of the bootstrapped statistics. (`scipy.stats.bootstrap`, or manual implementation).

A foundational understanding of these statistical concepts enables astronomers to correctly model their data, estimate parameters reliably, quantify uncertainties accurately, test hypotheses rigorously, and ultimately draw sound scientific conclusions from their computational analyses.
