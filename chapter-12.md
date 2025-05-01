---
# Chapter 12
# Physical Modeling and Parameter Estimation 
---

This chapter focuses on the crucial bridge between processed astronomical data and physical understanding: the process of fitting theoretical or empirical models to observations and quantitatively estimating the parameters that characterize those models. It explores the principles underlying scientific model fitting, emphasizing how models serve as mathematical representations of astrophysical phenomena or instrumental effects, allowing us to test hypotheses and extract physical insights from data. Fundamental statistical concepts that form the bedrock of model fitting are introduced, including probability distributions, the principles of least-squares minimization (chi-squared fitting), Maximum Likelihood Estimation (MLE) as a powerful parameter estimation framework, and the core tenets of Bayesian inference, which incorporates prior knowledge and provides a probabilistic interpretation of model parameters. The chapter delves into practical computational techniques for parameter estimation, particularly highlighting Markov Chain Monte Carlo (MCMC) methods as robust tools for exploring complex, high-dimensional parameter spaces and characterizing posterior probability distributions within a Bayesian framework, mentioning widely used Python implementations like `emcee` and `dynesty`. The utility of the `astropy.modeling` package, providing a standardized interface for defining models and accessing fitting algorithms, is discussed. Finally, a series of practical examples demonstrate the application of these modeling and parameter estimation techniques across diverse astronomical contexts—fitting spectral line profiles, modeling asteroid thermal emission, comparing stellar population models to observational data, fitting exoplanet transit light curves, decomposing complex emission line profiles, modeling galaxy surface brightness, and fitting cosmological relationships—illustrating how these methods transform observational data into quantitative physical knowledge.

---

**12.1 Principles of Scientific Model Fitting (`astropy.modeling`)**

At its core, scientific progress in astrophysics involves confronting theoretical predictions with observational data. **Scientific models** are mathematical representations or computational algorithms designed to describe physical processes, predict observable quantities, or characterize instrumental behavior. **Model fitting** is the quantitative process of adjusting the parameters of a chosen model to achieve the best possible agreement with a given set of observational data, while also assessing the uncertainties associated with the fitted parameters and potentially evaluating the appropriateness of the model itself (Bevington & Robinson, 2003; Wall & Jenkins, 2012). This process allows astronomers to estimate physical quantities (e.g., temperature, mass, distance, velocity, chemical abundance, cosmological parameters), test the validity of theoretical frameworks, and characterize the properties of celestial objects and the universe.

The general procedure involves several key components:
1.  **Data:** The observational measurements ($y_i$) with associated uncertainties ($\sigma_i$) at specific independent coordinates ($x_i$). These data have undergone reduction and calibration (Chapters 3, 4, 5).
2.  **Model:** A mathematical function or computational procedure $M(x; \theta)$ that predicts the expected data value at coordinate $x$ given a set of model parameters $\theta = \{\theta_1, \theta_2, ..., \theta_k\}$. The model should encapsulate the underlying hypothesis or physical understanding being tested. Models can range from simple empirical functions (e.g., polynomials, Gaussians) to complex, physically motivated theoretical models (e.g., stellar atmosphere models, cosmological simulations, radiative transfer codes).
3.  **Parameters ($\theta$):** The adjustable quantities within the model that are tuned to match the data. These are often the physically interesting quantities we wish to estimate.
4.  **Goodness-of-Fit Statistic:** A quantitative measure that assesses how well the model $M(x; \theta)$ matches the observed data $y_i$, considering the uncertainties $\sigma_i$. Common statistics include the chi-squared ($\chi^2$) statistic (for least-squares fitting) or the likelihood function (for MLE and Bayesian inference).
5.  **Fitting Algorithm (Optimizer/Sampler):** An algorithm that systematically explores the parameter space $\theta$ to find the set of parameters $\hat{\theta}$ that optimizes the chosen goodness-of-fit statistic (e.g., minimizes $\chi^2$, maximizes likelihood, or samples the posterior probability distribution).

The **`astropy.modeling`** package provides a powerful and flexible framework within Python for representing models and performing fits (Astropy Collaboration et al., 2022). Its key features include:
*   **Model Classes:** A library of pre-defined 1D and 2D model classes commonly used in astronomy (e.g., `Gaussian1D`, `Lorentz1D`, `Voigt1D`, `Polynomial1D`, `Chebyshev1D`, `PowerLaw1D`, `BlackBody1D`, `Sersic2D`, `Gaussian2D`, `Moffat2D`). These models encapsulate the function evaluation and parameter definitions.
*   **Custom Models:** Users can easily define their own models using the `@custom_model` decorator or by subclassing `FittableModel`.
*   **Compound Models:** Models can be combined through arithmetic operations (+, -, *, /, **) or functional composition (|) to create complex composite models (e.g., fitting multiple emission lines on a polynomial background: `Polynomial1D() + Gaussian1D() + Gaussian1D()`). `astropy.modeling` automatically handles the combined parameter sets and function evaluation.
*   **Parameter Handling:** Parameters are attributes of the model instances, allowing easy access and setting of values, bounds, or fixing parameters during fits. Units (`astropy.units`) can often be associated with parameters.
*   **Fitting Interfaces:** The `astropy.modeling.fitting` sub-package provides various fitting algorithms (fitters) that work seamlessly with `astropy.modeling` model objects. Common fitters include `LinearLSQFitter` (for models linear in parameters, like polynomials), `LevMarLSQFitter` (Levenberg-Marquardt algorithm for non-linear least squares), and wrappers for other `scipy.optimize` algorithms. Bayesian fitting via MCMC is often handled by dedicated packages like `emcee` or `dynesty` used in conjunction with model evaluation.

Using a standardized framework like `astropy.modeling` promotes code clarity, simplifies the definition and manipulation of complex models, and separates the model definition from the fitting algorithm, allowing different fitters to be applied to the same model structure. It forms a crucial component of the computational toolkit for extracting physical information from astronomical data.

**12.2 Statistical Foundations for Model Fitting**

Choosing the best model parameters $\hat{\theta}$ and assessing their uncertainties relies fundamentally on statistical principles. Different fitting philosophies (frequentist vs. Bayesian) lead to different optimization criteria and interpretations, but all depend on understanding probability distributions and how they relate models to data (Feigelson & Babu, 2012; Gregory, 2005).

*   **12.2.1 Probability Distributions (`scipy.stats`)**
    Probability distributions describe the likelihood of different outcomes for a random variable. They are essential for characterizing measurement uncertainties and defining likelihood functions.
    *   **Gaussian (Normal) Distribution:** $P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. This is ubiquitous in astronomy, often assumed to describe measurement errors due to the Central Limit Theorem. If measurement errors $\sigma_i$ are Gaussian, the deviation of a data point $y_i$ from a model prediction $M(x_i; \theta)$ follows a Gaussian distribution centered on zero. The `scipy.stats.norm` object provides functions for the PDF, CDF, random variates, etc.
    *   **Poisson Distribution:** $P(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$ (for integer $k \ge 0$). Describes the probability of observing $k$ events in a fixed interval when the average rate of events is $\lambda$. Fundamental for describing photon counting statistics (shot noise) where the uncertainty is $\sigma = \sqrt{\lambda} \approx \sqrt{k}$ for large counts. Relevant when working directly with photon counts before converting to flux. `scipy.stats.poisson`.
    *   **Chi-Squared ($\chi^2$) Distribution:** $P(x|\nu)$. Describes the distribution of the sum of squares of $\nu$ independent, standard normal random variables. Crucial for evaluating the goodness-of-fit in least-squares methods. `scipy.stats.chi2`.
    *   **Uniform Distribution:** $P(x|a, b) = 1/(b-a)$ for $a \le x \le b$, and 0 otherwise. Often used to represent prior probabilities when little information is available about a parameter's value within a given range. `scipy.stats.uniform`.
    Understanding the assumed distribution of measurement errors is critical for choosing the appropriate goodness-of-fit statistic.

*   **12.2.2 Least Squares Fitting (Chi-Squared Minimization)**
    This widely used method assumes that the measurement errors $\sigma_i$ associated with each data point $y_i$ are independent and Gaussian distributed. The goal is to find the model parameters $\theta$ that minimize the **chi-squared ($\chi^2$) statistic**, defined as the sum of squared, uncertainty-weighted residuals:
    $\chi^2(\theta) = \sum_{i=1}^{N} \left( \frac{y_i - M(x_i; \theta)}{\sigma_i} \right)^2$
    Minimizing $\chi^2$ corresponds to finding the parameters that make the model $M(x_i; \theta)$ pass as close as possible to the data points $y_i$, giving more weight to points with smaller uncertainties $\sigma_i$.
    *   **Optimization:** For models linear in parameters (e.g., $M(x) = c_0 + c_1 x$), $\chi^2$ minimization has an analytical solution (linear algebra). For non-linear models, iterative numerical optimization algorithms (e.g., Levenberg-Marquardt, implemented in `scipy.optimize.leastsq` or `astropy.modeling.fitting.LevMarLSQFitter`) are used to find the minimum $\chi^2$.
    *   **Goodness-of-Fit:** The value of the minimum chi-squared, $\chi^2_{min}$, can be used to assess how well the *best-fit* model represents the data. If the model is correct and errors are Gaussian and correctly estimated, $\chi^2_{min}$ should follow a chi-squared distribution with $\nu = N - k$ degrees of freedom (where $N$ is the number of data points and $k$ is the number of fitted parameters). The **reduced chi-squared** $\chi^2_\nu = \chi^2_{min} / \nu$ should be approximately 1. Values significantly larger than 1 indicate a poor fit (model is wrong or errors underestimated), while values significantly smaller than 1 might suggest overestimated errors. The probability of obtaining a $\chi^2_{min}$ value as large as or larger than observed, assuming the model is correct (the p-value), can be calculated from the $\chi^2$ CDF (`1 - scipy.stats.chi2.cdf(chi2_min, nu)`).
    *   **Parameter Uncertainties:** In the frequentist framework, uncertainties on the best-fit parameters $\hat{\theta}$ are often estimated from the **covariance matrix** $\mathbf{C}$, which is related to the second derivatives of the $\chi^2$ surface near the minimum ($C_{jk} \approx [ \frac{1}{2} \frac{\partial^2 \chi^2}{\partial \theta_j \partial \theta_k} ]^{-1}$). The square roots of the diagonal elements of $\mathbf{C}$ ($\sqrt{C_{jj}}$) provide the standard error estimates ($\sigma_{\theta_j}$) for each parameter. Off-diagonal elements describe the covariance (correlation) between parameter estimates. Many fitting routines (e.g., `scipy.optimize.curve_fit`, `astropy.modeling` fitters) can return this covariance matrix. Confidence intervals can be derived by exploring contours of constant $\Delta\chi^2 = \chi^2 - \chi^2_{min}$ around the minimum (e.g., $\Delta\chi^2 = 1$ corresponds approximately to the 68.3% confidence interval for one parameter).

*   **12.2.3 Maximum Likelihood Estimation (MLE)**
    MLE provides a more general framework than least squares, applicable even when errors are not Gaussian. It aims to find the model parameters $\theta$ that maximize the **likelihood function** $\mathcal{L}(\theta)$, which represents the probability of observing the actual data $D = \{y_i\}$ given the model parameters $\theta$:
    $\mathcal{L}(\theta) = P(D | \theta, M) = \prod_{i=1}^{N} P(y_i | M(x_i; \theta), \sigma_i)$
    Here, $P(y_i | M(x_i; \theta), \sigma_i)$ is the probability density of observing data point $y_i$ given the model prediction $M(x_i; \theta)$ and the uncertainty model (e.g., a Gaussian distribution with standard deviation $\sigma_i$). The product assumes independent measurements.
    *   **Relationship to Least Squares:** If the measurement errors are independent and Gaussian with known variances $\sigma_i^2$, then $P(y_i | ...) \propto \exp(-(y_i - M(x_i; \theta))^2 / (2\sigma_i^2))$. Maximizing $\mathcal{L}(\theta)$ is equivalent to maximizing its logarithm (the log-likelihood, $\ln \mathcal{L}$) which, in this case, is equivalent to *minimizing* $\frac{1}{2} \sum (\frac{y_i - M(x_i; \theta)}{\sigma_i})^2$. Thus, for Gaussian errors, MLE yields the same parameter estimates as least-squares fitting ($\hat{\theta}_{MLE} = \hat{\theta}_{LS}$).
    *   **Advantages:** Applicable to non-Gaussian error distributions (e.g., Poisson counts). Provides a statistically robust framework grounded in probability theory.
    *   **Optimization:** Finding the maximum likelihood often involves numerical optimization algorithms (similar to non-linear least squares) applied to the log-likelihood function (since maximizing $\ln \mathcal{L}$ is equivalent to maximizing $\mathcal{L}$ but often numerically more stable).
    *   **Parameter Uncertainties:** Uncertainties in MLE are typically estimated from the curvature of the log-likelihood function near its maximum, often using the Fisher information matrix (related to the second derivatives of $\ln \mathcal{L}$), which yields the covariance matrix similar to the least-squares case under certain conditions. Alternatively, likelihood ratio tests or profile likelihoods can be used to define confidence intervals.

*   **12.2.4 Bayesian Inference Fundamentals**
    Bayesian inference provides a fundamentally different approach by treating model parameters $\theta$ themselves as random variables described by probability distributions. It combines prior knowledge about the parameters with information from the data (via the likelihood function) to obtain the **posterior probability distribution (PDF)** of the parameters, $P(\theta | D, M)$, using Bayes' Theorem:
    $P(\theta | D, M) = \frac{P(D | \theta, M) \times P(\theta | M)}{P(D | M)}$ or, in words: **Posterior $\propto$ Likelihood $\times$ Prior**. 
    *   **Likelihood ( $P(D | \theta, M)$ ):** Same as in MLE, representing the probability of the data given specific parameter values. For Gaussian errors, $\mathcal{L}(\theta) \propto \exp(-\chi^2(\theta)/2)$.
    *   **Prior ( $P(\theta | M)$ ):** Represents our state of knowledge or belief about the parameter values *before* considering the data. Priors can be "uninformative" (e.g., uniform over a wide range) or "informative" (e.g., based on previous experiments, physical constraints, or theoretical expectations). The choice of prior can influence the posterior, especially when data are not very constraining.
    *   **Posterior ( $P(\theta | D, M)$ ):** The result of the Bayesian analysis. It represents the full probability distribution of the model parameters *after* incorporating the information from the data. It encapsulates all information about the parameters and their uncertainties. Parameter estimates are typically derived from the posterior PDF (e.g., mean, median, or mode), and uncertainties are represented by credible intervals (e.g., the range containing 68% or 95% of the posterior probability).
    *   **Evidence ( $P(D | M)$ ):** Also known as the marginal likelihood or Bayesian evidence. It is the integral of the likelihood times the prior over the entire parameter space: $P(D|M) = \int P(D|\theta, M) P(\theta|M) d\theta$. It acts as a normalization constant for the posterior. Crucially, the evidence is used for **Bayesian model comparison** – comparing the relative probabilities of different models $M_1, M_2$ given the same data $D$, via the ratio of their evidences (the Bayes factor). Calculating the evidence is often computationally challenging.
    *   **Advantages:** Provides a full probabilistic description of parameter uncertainties (the posterior PDF). Naturally incorporates prior information. Provides a consistent framework for model comparison via Bayesian evidence.
    *   **Disadvantages:** Requires defining prior distributions. Computing the posterior PDF and especially the evidence can be computationally intensive, often requiring sampling methods like MCMC (Section 12.3) or Nested Sampling. Interpretation is inherently probabilistic.

The choice between frequentist (Least Squares, MLE) and Bayesian approaches often depends on the specific problem, the nature of uncertainties, the availability of prior information, and the desired interpretation of results. For complex models or when a full characterization of parameter degeneracies and uncertainties is needed, Bayesian methods are increasingly favored in astrophysics.

**12.3 Markov Chain Monte Carlo (MCMC) Methods (`emcee`, `dynesty`)**

In Bayesian inference (Section 12.2.4), the goal is to determine the posterior probability distribution $P(\theta | D, M)$ of the model parameters $\theta$. For complex models with many parameters ($k > \sim 3-4$) or non-trivial correlations between parameters, calculating this posterior distribution analytically or mapping it directly on a grid becomes computationally infeasible due to the high dimensionality of the parameter space. **Markov Chain Monte Carlo (MCMC)** methods provide a powerful computational solution by generating a sequence of samples $\{\theta_1, \theta_2, ..., \theta_S\}$ drawn directly from the target posterior distribution, even without knowing its normalization constant (the evidence $P(D|M)$) (Metropolis et al., 1953; Hastings, 1970; Foreman-Mackey et al., 2013). The density of these samples in different regions of the parameter space is proportional to the posterior probability density in those regions. By analyzing this chain of samples, one can estimate properties of the posterior, such as marginal distributions for individual parameters, mean/median parameter values, credible intervals, and correlations between parameters.

**Markov Chain:** A sequence of random variables where the probability distribution of the next state depends only on the current state, not on the sequence of events that preceded it (the Markov property).
**Monte Carlo:** Utilizing random sampling to obtain numerical results.

The basic idea behind MCMC algorithms like the **Metropolis-Hastings algorithm** is to construct a Markov chain whose stationary distribution (the distribution it converges to after many steps) is the desired target posterior distribution $P(\theta|D)$. This is achieved through a carefully designed random walk through the parameter space:
1.  **Initialization:** Start at an initial parameter vector $\theta_0$.
2.  **Proposal:** At step $t$, propose a move to a new point $\theta_{prop}$ based on the current point $\theta_t$ using a proposal distribution $q(\theta_{prop} | \theta_t)$. Common choices include simple symmetric proposals like adding a small Gaussian random offset to $\theta_t$.
3.  **Acceptance Probability:** Calculate the acceptance probability $\alpha$, which depends on the ratio of the posterior probabilities (or equivalently, likelihood $\times$ prior, since the evidence cancels out) at the proposed and current points, and potentially the proposal densities (for non-symmetric proposals):
    $\alpha = \min\left( 1, \frac{P(\theta_{prop} | D) q(\theta_t | \theta_{prop})}{P(\theta_t | D) q(\theta_{prop} | \theta_t)} \right) \approx \min\left( 1, \frac{\mathcal{L}(\theta_{prop}) P(\theta_{prop})}{\mathcal{L}(\theta_t) P(\theta_t)} \right)$ (for symmetric proposals)
4.  **Accept/Reject:** Generate a random number $u$ from a uniform distribution $U(0, 1)$. If $u < \alpha$, accept the proposed move and set $\theta_{t+1} = \theta_{prop}$. Otherwise, reject the move and set $\theta_{t+1} = \theta_t$ (the chain stays at the current point).
5.  **Iteration:** Repeat steps 2-4 for a large number of iterations $S$.

After an initial **burn-in** phase (where the chain moves from the arbitrary starting point $\theta_0$ towards the high-probability regions of the posterior), the subsequent samples $\{\theta_{burn+1}, ..., \theta_S\}$ constitute a representative sample from the posterior distribution $P(\theta | D, M)$.

**Practical Considerations:**
*   **Convergence:** Ensuring the chain has converged to the stationary distribution is crucial. Diagnostics like visual inspection of trace plots (parameter value vs. iteration number), autocorrelation times, or the Gelman-Rubin statistic (comparing multiple independent chains) are used to assess convergence. The initial burn-in samples must be discarded.
*   **Proposal Distribution:** The choice of proposal distribution $q$ significantly impacts the efficiency of the MCMC sampler. An optimal proposal explores the parameter space effectively, leading to reasonable acceptance rates (typically targeted around 20-50%) and faster convergence. Adaptive MCMC algorithms adjust the proposal distribution based on the chain's history.
*   **Number of Samples:** A sufficiently large number of post-burn-in samples ($S - burn$) is needed to accurately map the posterior distribution and estimate parameters and credible intervals with low Monte Carlo error. Autocorrelation between successive samples means the effective number of independent samples is often much smaller than the total number of steps.
*   **Multi-modality:** Standard MCMC algorithms can struggle to explore posterior distributions with multiple isolated peaks (modes). More advanced algorithms or running multiple chains from different starting points may be necessary.

**Python MCMC Libraries:** Several popular Python libraries implement efficient MCMC algorithms tailored for astrophysical applications:
*   **`emcee`:** Implements the affine-invariant ensemble sampler proposed by Goodman & Weare (2010) (Foreman-Mackey et al., 2013). It uses multiple "walkers" that explore the parameter space simultaneously and propose moves based on the positions of other walkers in the ensemble. It is particularly effective for exploring correlated parameter spaces and is relatively easy to use. It requires the user to provide a function that calculates the log-posterior probability (log-likelihood + log-prior) for a given parameter vector $\theta$. `emcee` is widely used in astronomy, especially for exoplanet transit/RV fitting and cosmological parameter estimation.
*   **`dynesty`:** Implements **Nested Sampling** algorithms (Skilling, 2004; Speagle, 2020). Nested sampling is an alternative Monte Carlo method designed primarily for calculating the Bayesian evidence $P(D|M)$ (crucial for model comparison) while also producing posterior samples as a by-product. It works by exploring the likelihood space within nested contours of constant likelihood, iteratively replacing the lowest-likelihood sample within an "active set" with a new sample drawn from the prior subject to the constraint that its likelihood is higher than the discarded sample. `dynesty` provides static and dynamic nested sampling options and is increasingly popular for problems involving model selection or complex, potentially multi-modal posteriors.
*   **Other libraries:** `PyMC` (Salvatier et al., 2016) and `Stan` (via `cmdstanpy`) offer more general probabilistic programming frameworks with advanced samplers (like Hamiltonian Monte Carlo - HMC, and No-U-Turn Sampler - NUTS) that can be more efficient than `emcee` for very high-dimensional or complex posteriors, but may have a steeper learning curve.

MCMC methods, particularly implemented via user-friendly libraries like `emcee` and `dynesty`, have become standard tools in astrocomputing for robust parameter estimation and uncertainty quantification within the Bayesian framework, enabling inference from complex models applied to rich datasets.

**12.4 Practical Fitting Interface (`astropy.modeling.fitting`)**

While dedicated MCMC libraries are essential for Bayesian posterior sampling, many fitting tasks involve simpler optimization problems, particularly finding the best-fit parameters in a frequentist sense via least-squares or maximum likelihood optimization. The **`astropy.modeling`** package provides a convenient high-level interface that connects its model representations (Section 12.1) with various fitting algorithms, including those available in `scipy.optimize`. This allows users to define a model structure and then apply different fitters to it with relative ease.

The core components involved are:
1.  **Model Instance:** An instance of an `astropy.modeling.Model` (e.g., `model = models.Gaussian1D(amplitude=1.0, mean=0, stddev=1.0)`). This object holds the model parameters and provides the function evaluation method. Initial parameter guesses are set when creating the instance or by modifying its attributes (e.g., `model.mean = 0.1`). Parameters can also be fixed (`model.mean.fixed = True`) or constrained with bounds (`model.amplitude.bounds = (0, None)`). Compound models can be built by combining model instances (e.g., `compound_model = models.Polynomial1D(1) + models.Gaussian1D()`).
2.  **Fitter Instance:** An instance of a fitter class from `astropy.modeling.fitting`. Common choices include:
    *   `fitting.LinearLSQFitter`: For models that are linear in their parameters (e.g., `Polynomial1D`, `Chebyshev1D`, `Legendre1D`). Uses linear algebra (singular value decomposition) for an exact least-squares solution. Very fast and robust for appropriate models.
    *   `fitting.LevMarLSQFitter`: Implements the Levenberg-Marquardt algorithm, a standard iterative method for non-linear least-squares minimization (minimizing $\chi^2$). Suitable for fitting non-linear models like Gaussians, Lorentzians, Voigts, power laws, etc., assuming Gaussian errors.
    *   `fitting.SLSQPLSQFitter`: Uses the Sequential Least Squares Programming algorithm from `scipy.optimize.minimize`, allowing for bound constraints on parameters.
    *   `fitting.SimplexLSQFitter`: Uses the Nelder-Mead simplex algorithm, which does not require derivatives but can be slower to converge.
    *   Fitters designed for Maximum Likelihood Estimation (e.g., for Poisson data) might be available or require custom implementation wrappers.
3.  **Fitting Execution:** The fitting is performed by calling the fitter instance with the model instance and the data:
    `fitted_model = fitter(model_instance, x_data, y_data, weights=1.0/uncertainty**2)`
    *   `model_instance`: The model to be fitted (with initial parameter guesses).
    *   `x_data`, `y_data`: The independent and dependent data arrays. For 2D models, `x_data` might be a tuple `(x, y)`.
    *   `weights`: Optional array specifying the weights for each data point, typically inverse variance ($1/\sigma_i^2$) for chi-squared minimization.
4.  **Output:** The fitter returns a *new* model instance (`fitted_model`) where the parameters have been updated to their best-fit values. The fitter instance itself often stores additional information about the fit, such as the number of iterations, function evaluations, and potentially the covariance matrix (`fitter.fit_info['param_cov']`) which can be used to estimate parameter uncertainties (Section 12.2.2).

This interface allows users to easily switch between different fitting algorithms while keeping the model definition separate. For example, one could try fitting a Gaussian first with `LevMarLSQFitter` and then potentially with `SimplexLSQFitter` without redefining the `Gaussian1D` model itself. This separation facilitates experimentation and finding the most suitable algorithm for a given model and dataset. While `astropy.modeling.fitting` primarily focuses on optimization (finding the single best-fit point), its model evaluation capabilities are also crucial when used within Bayesian MCMC frameworks (like `emcee`), where the likelihood function often involves calculating the model prediction using an `astropy.modeling` model instance for each proposed parameter set $\theta$.

**12.5 Examples in Practice (Python): Model Fitting Applications**

The following examples illustrate the practical application of model fitting and parameter estimation techniques across different astronomical scenarios. They demonstrate fitting simple analytical models (Gaussian, linear), physical models (blackbody conceptually), and using MCMC for more complex Bayesian inference (transit fitting). Libraries like `astropy.modeling`, `scipy.optimize`, `scipy.stats`, `emcee`, and `batman` are utilized to showcase common workflows.

**12.5.1 Solar: Gaussian Fitting to Spectral Line Profile**
Solar spectra often exhibit numerous absorption or emission lines whose profiles carry information about the physical conditions (temperature, turbulence, magnetic fields) and kinematics of the solar atmosphere. Fitting a Gaussian profile to a relatively symmetric and isolated line is a common way to measure its central wavelength (for Doppler shifts), width (related to thermal/turbulent broadening), and depth/amplitude. This example uses `astropy.modeling` to define a Gaussian model and `astropy.modeling.fitting.LevMarLSQFitter` to fit it to a simulated solar spectral line profile, extracting the best-fit parameters.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Solar Gaussian fit example.")
    class Spectrum1D: pass # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

# --- Simulate Solar Line Profile (Continuum Normalized) ---
if specutils_available:
    wavelengths = np.linspace(5434.0, 5435.0, 150) * u.AA
    line_center = 5434.53 * u.AA
    line_depth = 0.6 # Absorption depth
    line_sigma = 0.06 * u.AA
    flux_normalized = 1.0 - line_depth * np.exp(-0.5 * ((wavelengths - line_center) / line_sigma)**2)
    # Add noise
    flux_normalized += np.random.normal(0, 0.02, size=wavelengths.shape)
    # Create Spectrum1D object
    solar_line_spec = Spectrum1D(flux=flux_normalized, spectral_axis=wavelengths)
    print("Simulated solar absorption line profile created.")

    # --- Define Gaussian Model and Initial Guesses ---
    print("Defining Gaussian model for fitting...")
    # Model = 1.0 + Gaussian1D(amplitude < 0)
    # Or fit Gaussian1D to depth profile (1 - flux)
    depth_profile = 1.0 - solar_line_spec.flux.value # Dimensionless depth
    # Provide initial guesses (use value for model initialization)
    amp_guess = line_depth
    mean_guess = line_center.value
    stddev_guess = line_sigma.value
    # Create Gaussian1D model instance for the depth profile
    gauss_init = models.Gaussian1D(amplitude=amp_guess, mean=mean_guess, stddev=stddev_guess)
    # Optional: Set bounds or fix parameters if needed
    # gauss_init.amplitude.bounds = (0, 1.0) # Depth cannot be > 1
    # gauss_init.mean.bounds = (wavelengths.value.min(), wavelengths.value.max())

    # --- Choose Fitter and Perform Fit ---
    print("Fitting Gaussian model to depth profile...")
    # Use Levenberg-Marquardt fitter for non-linear least squares
    fitter = fitting.LevMarLSQFitter()
    # Fit the model to the depth data
    try:
        fitted_gaussian = fitter(gauss_init, solar_line_spec.spectral_axis.value, depth_profile,
                                 # weights=1.0/uncertainty**2 # Provide weights if error available
                                 maxiter=500) # Increase max iterations if needed
        print("Fit complete.")
        print("\nFitted Gaussian Parameters:")
        fit_amplitude = fitted_gaussian.amplitude.value
        fit_mean = fitted_gaussian.mean.value * u.AA # Add units back
        fit_stddev = fitted_gaussian.stddev.value * u.AA
        fit_fwhm = fitted_gaussian.fwhm * u.AA # Calculate FWHM

        print(f"  Amplitude (Depth): {fit_amplitude:.4f}")
        print(f"  Mean (Center): {fit_mean:.4f}")
        print(f"  Stddev (Width): {fit_stddev:.4f}")
        print(f"  FWHM: {fit_fwhm:.4f}")

        # Get parameter uncertainties from covariance matrix (if fitter provides it)
        if hasattr(fitter, 'fit_info') and 'param_cov' in fitter.fit_info and fitter.fit_info['param_cov'] is not None:
            param_uncert = np.sqrt(np.diag(fitter.fit_info['param_cov']))
            print("\nParameter Uncertainties (approx):")
            print(f"  Amplitude Uncert: {param_uncert[0]:.4f}")
            print(f"  Mean Uncert: {param_uncert[1]:.4f} AA")
            print(f"  Stddev Uncert: {param_uncert[2]:.4f} AA")
        else:
             print("\nCould not retrieve parameter uncertainties from fitter.")

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 5))
        plt.plot(solar_line_spec.spectral_axis, solar_line_spec.flux, 'b.', label='Simulated Data (Norm Flux)')
        # Plot the final fitted profile (1 - fitted_gaussian)
        plt.plot(solar_line_spec.spectral_axis, 1.0 - fitted_gaussian(solar_line_spec.spectral_axis.value),
                 'r-', label=f'Gaussian Fit (Center={fit_mean:.3f})')
        plt.xlabel(f"Wavelength ({solar_line_spec.spectral_axis.unit})")
        plt.ylabel("Normalized Flux")
        plt.title("Gaussian Fit to Solar Absorption Line")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred during Gaussian fitting: {e}")
else:
    print("Skipping Solar Gaussian fit example: specutils unavailable.")

```

This Python script demonstrates fitting a Gaussian profile to a simulated solar absorption line using `astropy.modeling`. It first generates a continuum-normalized spectrum with a Gaussian absorption feature plus noise, representing typical solar line data. The script defines a `Gaussian1D` model from `astropy.modeling.models` with initial guesses for its parameters (amplitude/depth, mean/center, stddev/width), fitting the depth profile ($1 - F_{norm}$) for convenience. An instance of a non-linear least-squares fitter, `LevMarLSQFitter`, is created from `astropy.modeling.fitting`. The `fitter` object is then called with the initial model instance and the spectral data (wavelengths and corresponding depth values) to perform the fit. The function returns a *new* model instance (`fitted_gaussian`) containing the best-fit parameters. The script extracts these fitted parameters (amplitude, mean, standard deviation) and calculates the corresponding FWHM, printing the results. It also shows how to potentially access parameter uncertainties if the fitter calculates the covariance matrix. Finally, it visualizes the original data along with the best-fit Gaussian model overlaid, allowing assessment of the fit quality.

**12.5.2 Planetary: Simple Thermal Model Fitting (Asteroid)**
The infrared flux emitted by an asteroid depends on its size, albedo (reflectivity), distance, and surface temperature distribution, which is influenced by its rotation and thermal properties. Simple thermal models, like the Standard Thermal Model (STM) or the Near-Earth Asteroid Thermal Model (NEATM - Harris, 1998), relate these parameters to the observed thermal flux. Fitting these models to multi-band infrared observations (e.g., from WISE or Spitzer) allows estimation of asteroid diameter and albedo. This example simulates fitting a highly simplified thermal model (a greybody approximation proportional to $D^2 / \Delta^2$, ignoring detailed temperature distribution and wavelength dependence for simplicity) to simulated asteroid flux measurements at different distances using `scipy.optimize.curve_fit`, illustrating basic non-linear least-squares fitting outside the `astropy.modeling` framework.

```python
import numpy as np
from scipy.optimize import curve_fit # For basic non-linear least squares
import astropy.units as u
import matplotlib.pyplot as plt

# --- Simulate Asteroid Thermal Flux Data ---
# Assume observations at different heliocentric (r) and geocentric (Delta) distances
# Flux ~ Diameter^2 * Temperature^4 / Delta^2
# Temperature ~ 1 / sqrt(r) (approx equilibrium temp)
# So Flux ~ D^2 * (1/sqrt(r))^4 / Delta^2 = D^2 / (r^2 * Delta^2)
# Let's simulate flux proportional to D^2 / Delta^2 (ignoring r dependence and wavelength for simplicity)
true_diameter_km = 5.0 # km
# Simulate observations at various geocentric distances Delta (in AU)
delta_au = np.array([0.5, 0.8, 1.2, 1.8, 2.5]) * u.AU
# Simulate observed flux (arbitrary units, proportional to 1/Delta^2) + noise
flux_scale_factor = 100.0 # Relates D^2/Delta^2 to observed flux units
true_flux = flux_scale_factor * (true_diameter_km**2) / (delta_au.value**2)
# Add some measurement noise/scatter
flux_error = true_flux * 0.1 # 10% flux error
observed_flux = true_flux + np.random.normal(0, flux_error, size=len(delta_au))
print("Simulated asteroid thermal flux data created.")
print("Delta (AU):", delta_au.value)
print("Observed Flux:", observed_flux)

# --- Define the Simplified Thermal Model Function ---
# Model: Flux = constant * Diameter^2 / Delta^2
# We fit for the combined parameter: effective_size_param = sqrt(constant) * Diameter
# Or, fit for Diameter directly if constant is assumed/known
# Let's fit for Diameter, assuming constant = flux_scale_factor
def thermal_model_simplified(delta_values, diameter, scale=flux_scale_factor):
    """Simplified model: Flux = scale * D^2 / Delta^2"""
    # Ensure non-negative diameter during fit if needed
    if diameter < 0: return np.inf # Penalize negative diameter
    return scale * (diameter**2) / (delta_values**2)

# --- Perform Fit using scipy.optimize.curve_fit ---
print("\nFitting simplified thermal model using curve_fit...")
# curve_fit performs non-linear least squares minimization
# Inputs: model function, xdata, ydata, initial guess (p0), errors (sigma)
# xdata here is the geocentric distance Delta
x_data_fit = delta_au.value # Pass NumPy array without units to curve_fit
y_data_fit = observed_flux
y_sigma_fit = flux_error # Use simulated errors for weights

# Provide initial guess for the parameter (Diameter in km)
initial_guess = [3.0] # Guess Diameter = 3 km

try:
    # Perform the fit
    # popt: array of optimal parameters found
    # pcov: estimated covariance matrix of the parameters
    popt, pcov = curve_fit(thermal_model_simplified, x_data_fit, y_data_fit,
                           p0=initial_guess, sigma=y_sigma_fit,
                           absolute_sigma=True) # Use sigma as absolute errors

    fitted_diameter = popt[0]
    # Estimate uncertainty from covariance matrix
    fit_diameter_uncert = np.sqrt(np.diag(pcov))[0]

    print("Fit complete.")
    print(f"  Fitted Diameter: {fitted_diameter:.2f} +/- {fit_diameter_uncert:.2f} km")
    print(f"  (Input true diameter was: {true_diameter_km:.2f} km)")

    # --- Optional: Plotting ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_data_fit, y_data_fit, yerr=y_sigma_fit, fmt='o', label='Simulated Data', capsize=3)
    # Plot the best-fit model curve
    delta_fine = np.linspace(x_data_fit.min()*0.9, x_data_fit.max()*1.1, 100)
    plt.plot(delta_fine, thermal_model_simplified(delta_fine, fitted_diameter), 'r-',
             label=f'Best Fit Model (D={fitted_diameter:.2f} km)')
    plt.xlabel("Geocentric Distance Δ (AU)")
    plt.ylabel("Observed Thermal Flux (Arbitrary Units)")
    plt.title("Simplified Asteroid Thermal Model Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Log scale might be useful if flux range is large
    # plt.yscale('log')
    # plt.xscale('log')
    plt.show()

except Exception as e:
    print(f"An error occurred during model fitting: {e}")

```

This Python script demonstrates fitting a simplified physical model to planetary data, specifically modeling the thermal flux from an asteroid, using `scipy.optimize.curve_fit`. It simulates observed thermal flux measurements for an asteroid at several different geocentric distances ($\Delta$), assuming a simplified relationship where flux is proportional to $Diameter^2 / \Delta^2$ (ignoring dependencies on heliocentric distance, albedo, emissivity, and wavelength for illustration). A Python function `thermal_model_simplified` is defined to represent this model, taking distance $\Delta$ and the parameter to be fitted (Diameter) as input. The `scipy.optimize.curve_fit` function is then used to perform a non-linear least-squares fit of this model function to the simulated observed fluxes (y-data) as a function of geocentric distance (x-data), weighting the fit by the measurement uncertainties (`sigma`). It requires an initial guess (`p0`) for the parameter(s). The function returns the optimal parameter value (`popt`, containing the best-fit diameter) and its estimated covariance matrix (`pcov`), from which the parameter uncertainty is derived. The plot shows the simulated data points with error bars and the best-fit thermal model curve overlaid.

**12.5.3 Stellar: Isochrone Fitting to Cluster CMD (Conceptual)**
One of the most powerful techniques for determining the age, distance, and metallicity of a star cluster is **isochrone fitting**. An isochrone represents the predicted locus of stars in a Color-Magnitude Diagram (CMD) – typically plotting magnitude versus color (e.g., G vs. BP-RP) – that were all born at the same time (iso-) with the same initial chemical composition (metallicity) but have evolved to the current time (chrone). By comparing the observed CMD of a star cluster to theoretical isochrones calculated from stellar evolution models for different ages, metallicities, and distances (or reddening), the best-matching isochrone reveals the cluster's properties. Performing a full statistical fit (e.g., Bayesian MCMC or likelihood analysis) involves comparing the distribution of observed stars to the density predicted by isochrones, accounting for photometric errors, stellar multiplicity, and field star contamination, which is computationally complex (e.g., using specialized codes - Dotter, 2016; Griggio et al., 2022; Monteiro et al., 2021). This example provides a highly simplified conceptual illustration: it simulates CMD data for a cluster, loads a single theoretical isochrone (assuming age, metallicity), applies distance modulus and reddening, and visually compares the shifted isochrone track to the simulated cluster data, highlighting the core principle without performing a quantitative fit.

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
# Assume isochrone data can be loaded (e.g., from MIST, PARSEC websites/files)
# For this example, we simulate a simple isochrone track

# --- Simulate Cluster CMD Data ---
# Simulate members following a simplified isochrone track + scatter/errors
n_stars = 500
# True parameters for simulation
true_age_gyr = 1.0 # Gyr
true_dist_mod = 10.0 # Distance modulus = 5*log10(d_pc) - 5 => d=1kpc
true_ebv = 0.05 # Color excess E(B-V)
# Approximate extinction A_G ~ 2.7 * E(B-V), A_BP ~ 3.4*E(B-V), A_RP ~ 2.1*E(B-V) (Gaia DR3 approx)
A_G = 2.7 * true_ebv
A_BP = 3.4 * true_ebv
A_RP = 2.1 * true_ebv
A_BP_RP = A_BP - A_RP # Reddening in BP-RP color

# Simulate intrinsic main sequence + turnoff + giant branch based on color
intrinsic_bp_rp = np.random.uniform(-0.1, 1.8, n_stars) # Color range
# Simplified intrinsic absolute G magnitude relation (piecewise)
intrinsic_abs_g = np.piecewise(intrinsic_bp_rp,
                             [intrinsic_bp_rp < 0.4, # Early main sequence
                              (intrinsic_bp_rp >= 0.4) & (intrinsic_bp_rp < 0.8), # Turnoff region approx
                              intrinsic_bp_rp >= 0.8], # Giant branch approx
                             [lambda x: 4.0 + 3*x, # Early MS slope
                              lambda x: 4.0 + 3*0.4 + 8*(x-0.4), # Sharper turnoff/subgiant slope
                              lambda x: 4.0 + 3*0.4 + 8*0.4 - 2*(x-1.2)]) # Red giant branch slope
# Add scatter representing age spread / binary sequence / intrinsic dispersion
intrinsic_abs_g += np.random.normal(0, 0.15 + 0.1*np.abs(intrinsic_bp_rp - 0.6), n_stars)

# Apply distance modulus and extinction/reddening
observed_g = intrinsic_abs_g + true_dist_mod + A_G
observed_bp_rp = intrinsic_bp_rp + A_BP_RP
# Add photometric errors
g_err = 0.02
bp_rp_err = 0.03
observed_g += np.random.normal(0, g_err, n_stars)
observed_bp_rp += np.random.normal(0, bp_rp_err, n_stars)

cluster_cmd_data = Table({'Gmag': observed_g, 'BP_RP': observed_bp_rp})
print("Simulated cluster CMD data created.")

# --- Simulate/Load a Single Theoretical Isochrone ---
# In practice, load from MIST/PARSEC file for specific age/metallicity
print("Simulating a theoretical isochrone track...")
iso_intrinsic_bp_rp = np.linspace(-0.1, 1.8, 200) # Color points for track
# Use the same simplified relations as for data simulation (represents the 'true' model)
iso_intrinsic_abs_g = np.piecewise(iso_intrinsic_bp_rp,
                             [iso_intrinsic_bp_rp < 0.4,
                              (iso_intrinsic_bp_rp >= 0.4) & (iso_intrinsic_bp_rp < 0.8),
                              iso_intrinsic_bp_rp >= 0.8],
                             [lambda x: 4.0 + 3*x,
                              lambda x: 4.0 + 3*0.4 + 8*(x-0.4),
                              lambda x: 4.0 + 3*0.4 + 8*0.4 - 2*(x-1.2)])
isochrone_track = Table({'Gmag_abs': iso_intrinsic_abs_g, 'BP_RP_int': iso_intrinsic_bp_rp})
isochrone_age = true_age_gyr # Assume we know/selected the correct age for this track

# --- Apply Assumed Distance Modulus and Reddening to Isochrone ---
# These are the parameters one would fit for in a real analysis
assumed_dist_mod = true_dist_mod # Use true value for visual overlay here
assumed_ebv = true_ebv
# Recalculate assumed extinction/reddening
assumed_A_G = 2.7 * assumed_ebv
assumed_A_BP_RP = (3.4 - 2.1) * assumed_ebv

# Shift the isochrone track to observed CMD space
isochrone_track['Gmag_obs'] = isochrone_track['Gmag_abs'] + assumed_dist_mod + assumed_A_G
isochrone_track['BP_RP_obs'] = isochrone_track['BP_RP_int'] + assumed_A_BP_RP
print(f"Shifted isochrone using DM={assumed_dist_mod:.2f}, E(B-V)={assumed_ebv:.3f}")

# --- Visualize Cluster Data and Isochrone Track ---
print("\nPlotting observed cluster CMD and isochrone track...")
plt.figure(figsize=(7, 9))
# Plot observed cluster data points
plt.scatter(cluster_cmd_data['BP_RP'], cluster_cmd_data['Gmag'], s=5, alpha=0.5, label='Simulated Cluster Data')
# Plot the shifted theoretical isochrone track
plt.plot(isochrone_track['BP_RP_obs'], isochrone_track['Gmag_obs'], 'r-', lw=2,
         label=f'Isochrone (Age={isochrone_age} Gyr, Shifted)')
# Plot formatting
plt.xlabel("Gaia BP - RP Color")
plt.ylabel("Gaia G Magnitude")
plt.title("Cluster CMD with Isochrone Overlay (Conceptual)")
# Invert magnitude axis (brighter is up)
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, alpha=0.3)
# Set reasonable plot limits based on data
plt.xlim(np.percentile(cluster_cmd_data['BP_RP'], 1), np.percentile(cluster_cmd_data['BP_RP'], 99))
plt.ylim(np.percentile(cluster_cmd_data['Gmag'], 99) + 1, np.percentile(cluster_cmd_data['Gmag'], 1) - 1)
plt.show()

print("\nNOTE: This example shows only visual comparison.")
print("Quantitative isochrone fitting involves statistical comparison")
print("(e.g., likelihood, Bayesian methods) over grids of age, metallicity,")
print("distance, and reddening, accounting for errors and stellar density.")

```

This Python script provides a conceptual illustration of isochrone fitting, a fundamental technique for determining star cluster properties, without performing the complex statistical fit itself. It first simulates observational data for a star cluster's Color-Magnitude Diagram (CMD), plotting Gaia G magnitude against BP-RP color, incorporating scatter due to photometric errors and potentially age/binary spreads around a simplified underlying stellar evolutionary track. It then simulates loading a single theoretical isochrone track (representing stars of a specific age and metallicity) from stellar evolution models. The core concept demonstrated is the shifting of this theoretical isochrone in both color (due to interstellar reddening, parameterized by $E(B-V)$) and magnitude (due to distance modulus and extinction, $DM + A_G$). By applying assumed values for distance modulus and reddening (which would be free parameters in a real fit) to the theoretical track, it is overlaid onto the observed cluster CMD. The visual comparison between the shifted isochrone and the cluster data allows astronomers to qualitatively assess the match and estimate the cluster's properties; a quantitative fit would involve statistically comparing the data point distribution to isochrones across a range of parameters using likelihood or Bayesian methods.

**12.5.4 Exoplanetary: Transit Light Curve Fitting (`batman`, `emcee`)**
Fitting transit light curves with physical models is essential for determining the parameters of exoplanets, such as the planet-to-star radius ratio ($R_p/R_\star$), orbital inclination ($i$), and scaled semi-major axis ($a/R_\star$). The `batman` package efficiently calculates theoretical transit light curves, while MCMC samplers like `emcee` are widely used to perform Bayesian parameter estimation, exploring the posterior probability distribution of the transit parameters given the observed photometric data. This example demonstrates fitting a transit model generated by `batman` to a simulated TESS-like light curve using `emcee` to find the best-fit parameters and their uncertainties.

```python
import numpy as np
import matplotlib.pyplot as plt
# Requires batman-package: pip install batman-package
try:
    import batman
    batman_available = True
except ImportError:
    print("batman-package not found, skipping Exoplanet transit fit example.")
    batman_available = False
# Requires emcee: pip install emcee
try:
    import emcee
    emcee_available = True
except ImportError:
    print("emcee not found, skipping Exoplanet transit fit example.")
    emcee_available = False
# Requires corner for plotting posteriors: pip install corner
try:
    import corner
    corner_available = True
except ImportError:
    corner_available = False
import time as timer # Avoid conflict with astropy time

# --- Simulate Transit Light Curve Data ---
if batman_available and emcee_available:
    print("Simulating transit light curve data...")
    # Define true parameters
    params_true = batman.TransitParams()
    params_true.t0 = 1.5      # time of inferior conjunction (transit center) [days]
    params_true.per = 3.5     # orbital period [days]
    params_true.rp = 0.1      # planet radius (in units of stellar radii)
    params_true.a = 8.8       # semi-major axis (in units of stellar radii)
    params_true.inc = 87.0    # orbital inclination (in degrees)
    params_true.ecc = 0.      # eccentricity
    params_true.w = 90.       # longitude of periastron (in degrees)
    params_true.u = [0.1, 0.3]# limb darkening coefficients [u1, u2]
    params_true.limb_dark = "quadratic" # limb darkening model

    # Define observation times (e.g., TESS-like sampling)
    times = np.linspace(0, 10, 500) # 10 days, 500 points
    # Simulate noise
    data_err = 0.0005 # 500 ppm noise

    # Initialize batman model and generate noise-free light curve
    m_true = batman.TransitModel(params_true, times)
    flux_true = m_true.light_curve(params_true)
    # Add noise
    flux_observed = flux_true + np.random.normal(0., data_err, len(times))
    print("Simulated data created.")

    # --- Define Model and Log-Likelihood/Posterior for MCMC ---
    # Function to calculate transit model flux for given parameters theta
    # theta typically contains: t0, per, log_rp, log_a, inc (or impact parameter b)
    # Fit for log(Rp/R*) and log(a/R*) often better behaved
    def transit_model_func(theta, time_data):
        t0_fit, per_fit, log_rp_fit, log_a_fit, inc_fit = theta
        params_fit = batman.TransitParams()
        params_fit.t0 = t0_fit
        params_fit.per = per_fit
        params_fit.rp = np.exp(log_rp_fit)
        params_fit.a = np.exp(log_a_fit)
        params_fit.inc = inc_fit
        params_fit.ecc = 0. # Assume circular for fit
        params_fit.w = 90.
        params_fit.u = [0.1, 0.3] # Fix limb darkening for simplicity
        params_fit.limb_dark = "quadratic"
        m_fit = batman.TransitModel(params_fit, time_data)
        return m_fit.light_curve(params_fit)

    # Log-likelihood function (assuming Gaussian errors)
    # ln(L) = -0.5 * sum[ (data - model)^2 / error^2 + ln(2*pi*error^2) ]
    def log_likelihood(theta, times, flux, flux_err):
        model_flux = transit_model_func(theta, times)
        sigma2 = flux_err**2
        return -0.5 * np.sum((flux - model_flux)**2 / sigma2 + np.log(2 * np.pi * sigma2))

    # Log-prior function (define allowed ranges for parameters)
    def log_prior(theta):
        t0, per, log_rp, log_a, inc = theta
        # Example flat priors within reasonable ranges
        if (1.0 < t0 < 2.0 and 3.0 < per < 4.0 and
            np.log(0.01) < log_rp < np.log(0.3) and # Rp/R* between 0.01 and 0.3
            np.log(5.0) < log_a < np.log(15.0) and  # a/R* between 5 and 15
            80.0 < inc < 90.0):
            return 0.0 # Log(1) = 0 -> flat prior probability
        return -np.inf # Log(0) = -inf -> zero prior probability outside range

    # Log-posterior function = Log-Likelihood + Log-Prior
    def log_posterior(theta, times, flux, flux_err):
        lp = log_prior(theta)
        if not np.isfinite(lp): # Check if parameters are outside prior range
            return -np.inf
        return lp + log_likelihood(theta, times, flux, flux_err)

    # --- Run MCMC using emcee ---
    print("\nRunning MCMC using emcee...")
    # Set up MCMC parameters
    n_dim = 5 # Number of parameters we are fitting
    n_walkers = 32 # Number of walkers in the ensemble (should be > 2*n_dim)
    n_burn = 500 # Number of burn-in steps to discard
    n_steps = 2000 # Number of production steps per walker

    # Initial positions for walkers (small ball around initial guess)
    # Guess parameters slightly offset from true values
    initial_guess = np.array([params_true.t0 + 0.01, params_true.per - 0.02,
                              np.log(params_true.rp * 0.9), np.log(params_true.a * 1.1),
                              params_true.inc - 0.1])
    pos_init = initial_guess + 1e-4 * np.random.randn(n_walkers, n_dim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior,
                                    args=(times, flux_observed, data_err))

    # Run MCMC (including burn-in)
    start_mcmc = timer.time()
    sampler.run_mcmc(pos_init, n_steps + n_burn, progress=True)
    end_mcmc = timer.time()
    print(f"MCMC run complete in {end_mcmc - start_mcmc:.1f} seconds.")

    # --- Analyze MCMC Results ---
    # Discard burn-in steps and flatten the chain
    samples = sampler.get_chain(discard=n_burn, flat=True)
    # Convert log parameters back to original scale if needed
    samples[:, 2] = np.exp(samples[:, 2]) # Convert log_rp to rp
    samples[:, 3] = np.exp(samples[:, 3]) # Convert log_a to a

    print("\nPosterior Parameter Estimates (Median and 16th/84th percentiles):")
    param_names = ["t0", "per", "Rp/R*", "a/R*", "inc"]
    for i in range(n_dim):
        mcmc_median = np.percentile(samples[:, i], 50)
        mcmc_lower = np.percentile(samples[:, i], 16)
        mcmc_upper = np.percentile(samples[:, i], 84)
        print(f"  {param_names[i]}: {mcmc_median:.5f} +{mcmc_upper-mcmc_median:.5f} / -{mcmc_median-mcmc_lower:.5f}")

    # --- Optional: Corner Plot ---
    if corner_available:
        print("Generating corner plot...")
        # Provide true values for reference
        true_vals = [params_true.t0, params_true.per, params_true.rp, params_true.a, params_true.inc]
        fig = corner.corner(samples, labels=param_names, truths=true_vals,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.suptitle("MCMC Posterior Distributions (Exoplanet Transit Fit)")
        plt.show()

else:
    print("Skipping Exoplanet transit fit example: batman or emcee unavailable.")

```

This Python script demonstrates a Bayesian approach to fitting an exoplanet transit light curve using the `batman` package to generate the transit model and the `emcee` package for Markov Chain Monte Carlo (MCMC) sampling. It begins by simulating realistic transit data, generating a theoretical light curve using `batman` with known "true" planet parameters and adding noise. The core of the analysis involves defining Python functions for the `log_prior` (specifying allowed ranges for the parameters being fit: transit time $t_0$, period $P$, log of radius ratio $\ln(R_p/R_\star)$, log of scaled semi-major axis $\ln(a/R_\star)$, and inclination $i$), and the `log_likelihood` (calculating the probability of the data given the model based on Gaussian noise assumptions). These are combined into a `log_posterior` function. An `emcee.EnsembleSampler` is initialized with multiple "walkers," the number of parameters, and the log-posterior function. The sampler is run for a number of steps, allowing the walkers to explore the parameter space. After discarding an initial "burn-in" phase, the remaining samples represent the posterior probability distribution. The script analyzes these samples to derive median parameter estimates and credible intervals (representing uncertainties) and uses the `corner` package to visualize the 1D and 2D marginalized posterior distributions, showing the results of the MCMC fit and parameter correlations.

**12.5.5 Galactic: Multi-component Gaussian Fitting (HI Line)**
Radio observations of neutral hydrogen (HI) 21-cm emission line profiles from within the Milky Way or nearby galaxies often reveal complex structures resulting from multiple gas clouds along the line of sight moving at different velocities. Decomposing these blended profiles into individual components is crucial for studying Galactic structure and gas kinematics. This example demonstrates fitting a complex HI line profile (simulated as the sum of multiple Gaussian components) using `astropy.modeling`. It defines a compound model consisting of several `Gaussian1D` models plus a baseline and fits this composite model to the data to extract the parameters (center velocity, width, amplitude) of each individual kinematic component.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Galactic HI fit example.")
    class Spectrum1D: pass # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

# --- Simulate Blended HI Emission Line Profile ---
if specutils_available:
    # Define velocity axis (km/s)
    velocity_axis = np.linspace(-150, 150, 300) * u.km / u.s
    # Simulate baseline + noise
    baseline_level = 5.0 # mJy
    noise_level = 0.5 # mJy RMS
    flux_data = np.random.normal(baseline_level, noise_level, size=velocity_axis.shape)
    flux_unit = u.mJy
    # Add multiple Gaussian components representing gas clouds
    # Component 1: Broad, low velocity
    amp1 = 10.0; center1 = -80.0; sigma1 = 25.0
    flux_data += amp1 * np.exp(-0.5 * ((velocity_axis.value - center1) / sigma1)**2)
    # Component 2: Narrower, intermediate velocity
    amp2 = 20.0; center2 = -10.0; sigma2 = 8.0
    flux_data += amp2 * np.exp(-0.5 * ((velocity_axis.value - center2) / sigma2)**2)
    # Component 3: Weak, high positive velocity
    amp3 = 8.0; center3 = 90.0; sigma3 = 12.0
    flux_data += amp3 * np.exp(-0.5 * ((velocity_axis.value - center3) / sigma3)**2)
    flux_data *= flux_unit # Apply units

    # Create Spectrum1D object
    hi_spec = Spectrum1D(flux=flux_data, spectral_axis=velocity_axis)
    print("Simulated blended HI line profile created.")

    # --- Define Compound Model and Initial Guesses ---
    print("Defining compound model (Baseline + 3 Gaussians)...")
    # Provide initial guesses for each component based on visual inspection or prior knowledge
    # Baseline (Constant or Polynomial)
    cont_guess = models.Const1D(amplitude=np.median(flux_data).value) # Use median as baseline guess
    # Gaussian components (provide unique names for parameters if desired)
    g1_init = models.Gaussian1D(amplitude=8.0, mean=-90.0, stddev=20.0, name='Comp1')
    g2_init = models.Gaussian1D(amplitude=18.0, mean=-5.0, stddev=10.0, name='Comp2')
    g3_init = models.Gaussian1D(amplitude=6.0, mean=85.0, stddev=15.0, name='Comp3')
    # Combine models using '+' operator
    full_model_init = cont_guess + g1_init + g2_init + g3_init
    print("Initial model defined.")
    # Optional: Add bounds to parameters (e.g., amplitude > 0, stddev > 0)
    full_model_init.amplitude_1.min = 0
    full_model_init.stddev_1.min = 0
    full_model_init.amplitude_2.min = 0
    full_model_init.stddev_2.min = 0
    full_model_init.amplitude_3.min = 0
    full_model_init.stddev_3.min = 0

    # --- Fit the Compound Model ---
    print("Fitting the compound model...")
    # Choose a fitter
    fitter = fitting.LevMarLSQFitter()
    # Fit the model to the data (use .value for numerical fitting)
    try:
        fitted_model = fitter(full_model_init, hi_spec.spectral_axis.value, hi_spec.flux.value,
                              weights=1.0/noise_level**2, # Use weights based on noise
                              maxiter=1000) # Allow more iterations for complex fits
        print("Fit complete.")

        # --- Extract Fitted Parameters for Each Component ---
        print("\nFitted Parameters:")
        print(f"  Baseline Level: {fitted_model.amplitude_0.value:.2f} {flux_unit}")
        print("  Component 1:")
        print(f"    Amplitude: {fitted_model.amplitude_1.value:.2f} {flux_unit}")
        print(f"    Center Velocity: {fitted_model.mean_1.value:.2f} {velocity_axis.unit}")
        print(f"    Stddev (Velocity): {fitted_model.stddev_1.value:.2f} {velocity_axis.unit}")
        print("  Component 2:")
        print(f"    Amplitude: {fitted_model.amplitude_2.value:.2f} {flux_unit}")
        print(f"    Center Velocity: {fitted_model.mean_2.value:.2f} {velocity_axis.unit}")
        print(f"    Stddev (Velocity): {fitted_model.stddev_2.value:.2f} {velocity_axis.unit}")
        print("  Component 3:")
        print(f"    Amplitude: {fitted_model.amplitude_3.value:.2f} {flux_unit}")
        print(f"    Center Velocity: {fitted_model.mean_3.value:.2f} {velocity_axis.unit}")
        print(f"    Stddev (Velocity): {fitted_model.stddev_3.value:.2f} {velocity_axis.unit}")

        # --- Optional: Plotting ---
        plt.figure(figsize=(10, 6))
        plt.plot(hi_spec.spectral_axis, hi_spec.flux, label='Simulated HI Spectrum', drawstyle='steps-mid')
        # Plot the full fitted model
        plt.plot(hi_spec.spectral_axis, fitted_model(hi_spec.spectral_axis.value)*flux_unit,
                 'r-', label='Full Fit')
        # Plot individual components (add baseline to each Gaussian)
        baseline = fitted_model[0](hi_spec.spectral_axis.value) * flux_unit
        plt.plot(hi_spec.spectral_axis, baseline + fitted_model[1](hi_spec.spectral_axis.value)*flux_unit,
                 'g:', label='Fit Comp 1')
        plt.plot(hi_spec.spectral_axis, baseline + fitted_model[2](hi_spec.spectral_axis.value)*flux_unit,
                 'm:', label='Fit Comp 2')
        plt.plot(hi_spec.spectral_axis, baseline + fitted_model[3](hi_spec.spectral_axis.value)*flux_unit,
                 'c:', label='Fit Comp 3')

        plt.xlabel(f"Velocity ({hi_spec.spectral_axis.unit})")
        plt.ylabel(f"Flux Density ({hi_spec.flux.unit})")
        plt.title("Multi-component Gaussian Fit to HI Line Profile")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred during multi-component fitting: {e}")
else:
    print("Skipping Galactic HI fit example: specutils unavailable.")

```

This Python script demonstrates how to decompose a complex, blended spectral line profile, typical of Galactic HI 21-cm emission, into multiple Gaussian components using `astropy.modeling`. It simulates an HI spectrum exhibiting three overlapping Gaussian emission features superimposed on a flat baseline with noise. The core of the analysis involves constructing a **compound model** by summing individual `astropy.modeling` model instances: a `Const1D` for the baseline and three separate `Gaussian1D` models (given unique names like 'Comp1', 'Comp2', 'Comp3' for clarity) representing the distinct kinematic components. Initial guesses for the parameters of each component (amplitude, center velocity, width) are provided. A non-linear least-squares fitter (`LevMarLSQFitter`) is then used to optimize the parameters of this entire compound model simultaneously against the observed spectral data. The script extracts the best-fit parameters for the baseline and for each individual Gaussian component from the resulting `fitted_model` object. The visualization plots the original data, the overall best-fit compound model, and the individual fitted Gaussian components (plus baseline), clearly illustrating how the blended profile has been decomposed into its constituent parts.

**12.5.6 Extragalactic: Sérsic Profile Fitting (Galaxy Surface Brightness)**
Analyzing the structure of galaxies often involves characterizing their surface brightness profiles – how their brightness decreases with radius from the center. The Sérsic profile (Sérsic, 1963) is a widely used empirical function that provides a remarkably good fit to the light profiles of diverse galaxy types (ellipticals, bulges, some disks). Its functional form is:
$I(R) = I_e \exp\left( -b_n \left[ \left(\frac{R}{R_e}\right)^{1/n} - 1 \right] \right)$
where $I(R)$ is the intensity at radius $R$, $I_e$ is the intensity at the effective radius $R_e$ (which encloses half the total light), $n$ is the Sérsic index controlling the profile shape (n=4 for de Vaucouleurs profile typical of ellipticals, n=1 for exponential profile typical of disks), and $b_n$ is a constant related to $n$ (approximated as $b_n \approx 1.9992n - 0.3271$). Fitting a Sérsic model to a galaxy's observed radial surface brightness profile allows quantitative measurement of its size ($R_e$), concentration ($n$), and central surface brightness ($I_e$). This example demonstrates fitting a 1D Sérsic profile model (`astropy.modeling.models.Sersic1D`) to a simulated galaxy radial profile using `astropy.modeling.fitting`.

```python
import numpy as np
# Requires astropy: pip install astropy
try:
    from astropy.modeling import models, fitting
    astropy_modeling_available = True
except ImportError:
    print("astropy.modeling not found, skipping Extragalactic Sersic fit example.")
    astropy_modeling_available = False
import astropy.units as u
import matplotlib.pyplot as plt

# --- Simulate Galaxy Surface Brightness Profile Data ---
if astropy_modeling_available:
    # Define radial bins (e.g., in arcsec)
    radius = np.linspace(0.1, 30.0, 40) * u.arcsec
    # True parameters for simulation
    true_amplitude = 20.0 # Intensity at R=0 (related to I_e) [mag/arcsec^2 or flux units]
    true_r_eff = 5.0 * u.arcsec # Effective radius
    true_n = 2.5 # Sersic index

    # Use Sersic1D model to generate true profile
    # Amplitude in Sersic1D corresponds to I(0) if using default relation
    # Or directly I_e if specified. Let's use I(0) = amplitude.
    sersic_true_model = models.Sersic1D(amplitude=true_amplitude, r_eff=true_r_eff, n=true_n)
    true_surface_brightness = sersic_true_model(radius)
    # Add noise (e.g., increasing uncertainty at larger radii/fainter levels)
    sb_error = 0.5 + 0.1 * (radius / true_r_eff).value**0.5 # Example error model
    observed_sb = true_surface_brightness + np.random.normal(0, sb_error, size=radius.shape)
    observed_sb_unit = u.Unit("mag(AB) / arcsec2") # Example units (use magnitudes for realism)
    # Convert flux to magnitude (higher flux -> smaller mag)
    # Assume some zeropoint relation for magnitude
    mag_zp = 25.0
    observed_sb_mag = mag_zp - 2.5 * np.log10(np.maximum(observed_sb.value, 1e-3)) # Avoid log(0)
    sb_error_mag = (2.5 / np.log(10)) * (sb_error / np.maximum(observed_sb.value, 1e-3)) # Approx error prop

    print("Simulated galaxy surface brightness profile data created.")

    # --- Define Sersic Model and Initial Guesses ---
    print("Defining Sersic1D model for fitting...")
    # Provide initial guesses for amplitude (in mag/arcsec^2 -> convert from flux guess), r_eff, n
    # Guess from data? Median brightness, approx radius where flux drops significantly?
    amp_guess_mag = np.min(observed_sb_mag) # Brightest point as amplitude guess (mag)
    r_eff_guess = 8.0 * u.arcsec
    n_guess = 2.0
    # Need to fit magnitude data - Sersic1D gives intensity.
    # Option 1: Fit Intensity. Convert mag data back to flux.
    # Option 2: Create a custom magnitude Sersic model.
    # Let's try Option 1: Convert data back to flux scale for fitting
    observed_flux_fit = 10**(-0.4 * (observed_sb_mag - mag_zp))
    flux_error_fit = observed_flux_fit * (sb_error_mag * np.log(10) / 2.5)
    # Guess amplitude in flux units now
    amp_guess_flux = 10**(-0.4 * (amp_guess_mag - mag_zp))

    sersic_init = models.Sersic1D(amplitude=amp_guess_flux, r_eff=r_eff_guess, n=n_guess)
    # Add bounds to parameters
    sersic_init.amplitude.bounds = (0, None) # Flux amplitude must be positive
    sersic_init.r_eff.bounds = (0.1, 100.0) # Reasonable range for Reff
    sersic_init.n.bounds = (0.2, 8.0) # Reasonable range for n

    # --- Fit the Sersic Model ---
    print("Fitting Sersic1D model...")
    fitter = fitting.LevMarLSQFitter()
    # Fit to the flux data, using radius values
    try:
        fitted_sersic = fitter(sersic_init, radius.value, observed_flux_fit,
                               weights=1.0/flux_error_fit**2, # Weight by inverse variance
                               maxiter=1000)
        print("Fit complete.")

        # --- Extract Fitted Parameters ---
        print("\nFitted Sersic Parameters:")
        fit_amplitude_flux = fitted_sersic.amplitude.value
        fit_r_eff = fitted_sersic.r_eff.value * u.arcsec # Add units back
        fit_n = fitted_sersic.n.value
        # Convert fitted amplitude back to mag/arcsec^2 for comparison
        fit_amplitude_mag = mag_zp - 2.5 * np.log10(fit_amplitude_flux)

        print(f"  Amplitude (I(0) approx, mag/arcsec^2): {fit_amplitude_mag:.3f}")
        print(f"  Effective Radius (Re): {fit_r_eff:.3f}")
        print(f"  Sersic Index (n): {fit_n:.3f}")
        print(f"  (Input true values approx: Re={true_r_eff:.3f}, n={true_n:.3f})")

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 6))
        # Plot data in magnitudes
        plt.errorbar(radius.value, observed_sb_mag, yerr=sb_error_mag, fmt='o', label='Simulated Data')
        # Plot fitted model converted back to magnitudes
        radius_fine = np.linspace(radius.value.min(), radius.value.max(), 200)
        fitted_flux_fine = fitted_sersic(radius_fine)
        fitted_mag_fine = mag_zp - 2.5 * np.log10(np.maximum(fitted_flux_fine, 1e-5)) # Avoid log(0)
        plt.plot(radius_fine, fitted_mag_fine, 'r-', label=f'Sersic Fit (n={fit_n:.2f}, Re={fit_r_eff:.2f})')

        plt.xlabel(f"Radius ({radius.unit})")
        plt.ylabel(f"Surface Brightness ({observed_sb_unit})")
        plt.title("Sersic Profile Fit to Galaxy Surface Brightness")
        # Invert magnitude axis
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Log scale for radius often used
        # plt.xscale('log')
        plt.show()

    except Exception as e:
        print(f"An error occurred during Sersic fitting: {e}")
else:
    print("Skipping Extragalactic Sersic fit example: astropy.modeling unavailable.")

```

This Python script demonstrates fitting the widely used Sérsic profile to a simulated galaxy surface brightness profile using `astropy.modeling`. It first generates mock data representing the surface brightness (initially in flux units, then converted to magnitudes for realism) as a function of radius from the galaxy center, based on a known input Sérsic profile ($R_e$, $n$, amplitude) plus noise. The core of the analysis involves defining a `Sersic1D` model instance from `astropy.modeling.models` with initial parameter guesses. Since the model predicts intensity (flux) while the simulated data is in magnitudes, the observed magnitudes are converted back to flux units for the fitting process. Appropriate bounds are set on the parameters ($R_e$, $n$, amplitude) to ensure physical results. A non-linear least-squares fitter (`LevMarLSQFitter`) is then used to optimize the Sérsic model parameters (amplitude, $R_e$, $n$) to best match the simulated flux profile data, using the measurement errors to weight the fit. The script extracts and prints the best-fit parameters, providing quantitative measures of the galaxy's size ($R_e$) and concentration ($n$). The visualization plots the original surface brightness data (in magnitudes) along with the best-fit Sérsic model (converted back to magnitudes) overlaid, allowing assessment of the fit quality.

**12.5.7 Cosmology: Hubble Law Fitting (Supernovae)**
The Hubble-Lemaître Law describes the linear relationship between the distance to a galaxy and its recession velocity (or redshift) due to the expansion of the Universe: $v = H_0 d$. Type Ia supernovae (SNe Ia) serve as standardizable candles, allowing measurement of their distances ($d$) independent of redshift ($z \approx v/c$). Plotting distance modulus ($\mu$, related to $d$) versus redshift $z$ (or $\log z$) for a sample of SNe Ia yields the Hubble diagram. Fitting a linear model to this diagram provides an estimate of the Hubble constant ($H_0$), a fundamental cosmological parameter. This example simulates basic Hubble diagram data (distance modulus vs. redshift) for nearby SNe Ia and performs a simple linear least-squares fit using `scipy.stats.linregress` to estimate $H_0$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # For basic linear regression
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM # For context/comparison

# --- Simulate Nearby Supernova Hubble Diagram Data ---
# Low redshift range where linear Hubble law is good approximation: v = H0 * d
# v ~ c*z
# Distance modulus mu = 5*log10(d_Mpc) + 25
# d_Mpc ~ c*z / H0 => mu = 5*log10(c*z / H0) + 25
# mu = 5*log10(z) + 5*log10(c/H0) + 25
# So, mu vs log10(z) should be linear with slope 5. Intercept relates to H0.
true_H0 = 70.0 # km/s/Mpc
n_sne = 40
# Simulate redshifts (logarithmically spaced often better, but use linear here)
z_sne = np.random.uniform(0.01, 0.1, n_sne) # Nearby redshift range
# Calculate distance modulus using astropy cosmology
cosmo_true = FlatLambdaCDM(H0=true_H0, Om0=0.3)
true_dist_mod = cosmo_true.distmod(z_sne).value
# Add scatter/uncertainty to distance modulus measurements
dist_mod_err = 0.15 # Typical uncertainty in magnitudes
observed_dist_mod = true_dist_mod + np.random.normal(0, dist_mod_err, n_sne)
print(f"Simulated Hubble diagram data for {n_sne} nearby SNe Ia created.")

# --- Perform Linear Fit (mu vs log10(z)) ---
# Fit: mu = slope * log10(z) + intercept
x_data = np.log10(z_sne) # Independent variable
y_data = observed_dist_mod # Dependent variable
# Use scipy.stats.linregress for simple linear fit
print("Performing linear regression on Hubble diagram (mu vs log10(z))...")
try:
    # linregress returns: slope, intercept, r_value, p_value, stderr
    slope, intercept, r_value, p_value, std_err_slope = stats.linregress(x_data, y_data)
    print("Linear fit complete.")
    print(f"  Fitted Slope: {slope:.3f} +/- {std_err_slope:.3f} (Expected approx 5)")
    print(f"  Fitted Intercept: {intercept:.3f}")

    # --- Estimate H0 from Intercept ---
    # Intercept = 5*log10(c/H0) + 25  (where c is in km/s if H0 is km/s/Mpc)
    # Need speed of light in km/s
    c_kms = const.c.to(u.km/u.s).value
    # Solve for H0: 5*log10(H0) = 5*log10(c_kms) + 25 - intercept
    # log10(H0) = log10(c_kms) + 5 - intercept/5
    log10_H0_fit = np.log10(c_kms) + 5.0 - intercept / 5.0
    H0_fit = 10**log10_H0_fit
    # Basic error propagation for H0 (ignoring intercept uncertainty for simplicity)
    # Requires more careful error prop using covariance if available
    print(f"\nEstimated Hubble Constant (H0):")
    print(f"  H0 = {H0_fit:.2f} km/s/Mpc")
    print(f"  (Input true H0 was: {true_H0:.2f} km/s/Mpc)")
    # Note: Accuracy depends heavily on data quality, redshift range, and cosmology assumptions

    # --- Optional: Plotting ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(z_sne, observed_dist_mod, yerr=dist_mod_err, fmt='o', label='Simulated SNe Ia Data', capsize=3, alpha=0.7)
    # Plot the best-fit line (convert z to log10(z) for line)
    z_fine = np.logspace(np.log10(z_sne.min()), np.log10(z_sne.max()), 100)
    mu_fit_line = slope * np.log10(z_fine) + intercept
    plt.plot(z_fine, mu_fit_line, 'r-', label=f'Linear Fit (H0 ≈ {H0_fit:.1f})')
    plt.xscale('log') # Plot redshift on log scale
    plt.xlabel("Redshift (z)")
    plt.ylabel("Distance Modulus (μ)")
    plt.title("Hubble Diagram Fit (Nearby SNe Ia)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print(f"An error occurred during Hubble Law fitting: {e}")

```

This Python script demonstrates a simplified analysis of the Hubble-Lemaître Law using simulated Type Ia Supernova (SNe Ia) data. It generates mock data representing the measured distance moduli ($\mu$) and redshifts ($z$) for a sample of nearby SNe Ia, incorporating typical observational scatter. Recognizing the theoretical linear relationship between distance modulus and the logarithm of redshift ($\mu \approx 5 \log_{10}(z) + \text{Constant}$) in the low-redshift limit, the script uses `scipy.stats.linregress` to perform a linear least-squares fit to the simulated $\mu$ versus $\log_{10}(z)$ data. The fit yields the slope (which should be close to 5) and the intercept. The script then shows how the Hubble constant ($H_0$) can be estimated from the fitted intercept value using the definition of distance modulus and the relationship between velocity, redshift, and distance ($v \approx cz = H_0 d$). The final plot displays the simulated SNe Ia data points on a Hubble diagram (distance modulus vs. log redshift) along with the best-fit linear relationship, visually representing the expansion of the Universe and the estimation of $H_0$.

---

**References**

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project. Its `astropy.modeling` sub-package (Sections 12.1, 12.4) provides the core framework for defining and fitting models demonstrated in several examples (12.5.1, 12.5.5, 12.5.6, 12.5.7).

Buchner, J. (2021). Nested sampling methods. *Statistics and Computing, 31*(5), 70. https://doi.org/10.1007/s11222-021-10042-z
*   *Summary:* This paper provides a detailed overview of nested sampling algorithms (e.g., implemented in `dynesty`), an important alternative to MCMC for Bayesian inference and model comparison, relevant to the discussion of advanced sampling techniques (Section 12.3).

Foreman-Mackey, D., Farr, W. M., Sinha, M., Archibald, A. M., Hogg, D. W., Sanders, J. S., Zuntz, J., Williams, P. K. G., Nelson, A. R., de Val-Borro, M., Erhardt, T., Pasham, D. R., & Pla, O. (2021). exoplanet: Gradient-based probabilistic inference for exoplanet data & other astronomical time series. *Journal of Open Source Software, 6*(62), 3285. https://doi.org/10.21105/joss.03285
*   *Summary:* Introduces the `exoplanet` Python package, which combines probabilistic modeling (often Bayesian, Section 12.2.4) with efficient gradient-based samplers (like HMC) for analyzing time-series data, particularly exoplanet light curves and RVs (related to Example 12.5.4).

Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific, 125*(925), 306–312. https://doi.org/10.1086/670067 *(Note: Foundational software paper, pre-2020)*
*   *Summary:* The foundational paper describing the `emcee` package, which implements the affine-invariant ensemble MCMC sampler. Although pre-2020, `emcee` remains one of the most widely used MCMC tools in astronomy (Section 12.3) and is used in Example 12.5.4.

García-Benito, R., & Jurado, E. (2023). Optimization of background subtraction in 1D astronomical spectra using orthogonal polynomials. *Astronomy and Computing, 44*, 100735. https://doi.org/10.1016/j.ascom.2023.100735
*   *Summary:* Discusses using polynomial models (relevant to `astropy.modeling`, Section 12.1) for fitting backgrounds/continua in spectra. The fitting principles are analogous to those used for fitting line profiles or other models (Section 12.2).

Griggio, M., Torniamenti, S., Girardi, L., Rubele, S., Pastorelli, G., Bekki, K., Bressan, A., Chen, Y., Clementini, G., Groenewegen, M. A. T., Kerber, L., Marigo, P., Montalto, M., Nanni, A., Ripepi, V., Trabucchi, M., & van Loon, J. T. (2022). StarHorse results for Gaia EDR3: Estimating distances and extinctions for 1.47 billion stars using parallax-based distances for training. *Astronomy & Astrophysics, 663*, A16. https://doi.org/10.1051/0004-6361/202243375
*   *Summary:* This paper uses Bayesian methods (related to Section 12.2.4) to estimate stellar parameters (distance, extinction) by comparing Gaia data to stellar models (conceptually related to isochrone fitting, Example 12.5.3).

Monteiro, F. R., Dias, W. S., Monteiro, H., & Moitinho, A. (2021). Maximum likelihood isochrone fitting and the Gaia DR2 open cluster population. *Monthly Notices of the Royal Astronomical Society, 501*(4), 5442–5457. https://doi.org/10.1093/mnras/staa3823
*   *Summary:* Applies Maximum Likelihood Estimation (MLE, Section 12.2.3) for isochrone fitting to open clusters using Gaia data, providing a specific example of applying MLE to estimate cluster parameters (age, distance, etc.), related to Example 12.5.3.

Rezaei, Z., Johnson, B. D., Leja, J., & Conroy, C. (2022). Systematic effects in the modeling of galaxy spectral energy distributions: Line-of-sight dust, emission lines, and parameter degeneracies. *The Astrophysical Journal, 941*(1), 16. https://doi.org/10.3847/1538-4357/ac9c13
*   *Summary:* Discusses challenges and systematics in modeling galaxy SEDs, often involving complex models and Bayesian parameter estimation techniques (Sections 12.2.4, 12.3) to handle degeneracies between parameters like dust, age, and metallicity.

Scolnic, D., Brout, D., Carr, A., Riess, A. G., Davis, T. M., Dwomoh, A., Jones, D. O., Ali, N., Clocchiatti, A., Filippenko, A. V., Foley, R. J., Hicken, M., Hinton, S. R., Kessler, R., Lidman, C., Möller, A., Nugent, P. E., Popovic, B., Setiawan, A. K., … Wiseman, P. (2022). Measuring the Hubble Constant with Type Ia Supernovae Observed by the Dark Energy Survey Photometric Calibration System. *The Astrophysical Journal, 938*(2), 113. https://doi.org/10.3847/1538-4357/ac8e7a
*   *Summary:* This cosmological study relies on fitting light curve models (`sncosmo` related, see Example 8.6.7) to SNe Ia data and then fitting cosmological models (like the Hubble Law, Example 12.5.7) to the derived distances, often using sophisticated statistical frameworks (Section 12.2).

Speagle, J. S. (2020). DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences. *Monthly Notices of the Royal Astronomical Society, 493*(3), 3132–3158. https://doi.org/10.1093/mnras/staa278
*   *Summary:* The primary paper describing the `dynesty` Python package, which implements dynamic nested sampling. This is a powerful alternative to MCMC for Bayesian inference and evidence calculation (Section 12.3).

