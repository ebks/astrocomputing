
---
# Chapter 7
# Spectroscopic Analysis: Feature Measurement and Physical Interpretation
---

This chapter delves into the quantitative analysis of astronomical spectra following their initial reduction and calibration, focusing on the extraction of scientifically meaningful information encoded within spectral features. It addresses the critical methodologies employed to measure the properties of emission lines, absorption lines, and continuum characteristics, which serve as powerful diagnostics of the physical conditions, chemical composition, kinematics, and evolution of celestial objects. The discussion commences with techniques for modeling and normalizing the spectral continuum, a fundamental step required to isolate distinct spectral features for accurate measurement. Subsequently, methods for identifying spectral lines, both visually and algorithmically, against reference line lists are presented. The core of the chapter is dedicated to the detailed measurement of key line properties: determining precise line centroids to infer velocities or redshifts; calculating equivalent widths as indicators of line strength related to abundance or column density; quantifying line widths (e.g., Full Width at Half Maximum) to probe physical broadening mechanisms or instrumental resolution; and performing detailed line profile fitting using mathematical models (e.g., Gaussian, Voigt) to characterize line shapes and potentially deblend overlapping features. Techniques for measuring broader spectral indices, often used in stellar population and galaxy studies, are also outlined. Finally, the chapter explores common methods for determining the redshift of extragalactic objects, primarily through cross-correlation techniques or by fitting identified spectral features. Throughout the discussion, practical implementation using the Python scientific ecosystem, particularly leveraging the capabilities of `specutils` and `astropy.modeling`, is emphasized, with illustrative examples demonstrating the application of these analysis techniques across diverse astrophysical domains, thereby bridging the gap between calibrated data and physical interpretation.

---

**7.1 Continuum Fitting and Normalization (`specutils.fitting`)**

The spectrum of an astronomical object typically consists of discrete emission or absorption features superimposed on an underlying **continuum** radiation component. This continuum represents the broadband emission from the source (e.g., thermal emission from a stellar photosphere, synchrotron radiation from an active galactic nucleus (AGN), integrated light from unresolved stars in a galaxy) potentially modified by frequency-dependent absorption or scattering processes. Before detailed analysis of specific spectral lines can be performed accurately, it is often necessary to model and potentially remove or normalize by this continuum level (Reetz, 2023). **Continuum fitting** aims to estimate the shape of this underlying continuum across the observed wavelength range, while **continuum normalization** involves dividing the observed spectrum by the fitted continuum model. Normalization transforms the spectrum so that the continuum level is approximately unity, making the depths or heights of spectral features (relative to the continuum) easier to measure and compare, independent of the object's overall brightness or instrumental throughput effects not perfectly removed by flat-fielding (Section 4.6).

The process of continuum fitting requires careful consideration, as the "true" continuum is often complex and unobservable directly where strong lines are present. The key steps involve:
1.  **Region Selection/Masking:** Identify regions of the spectrum that are believed to be relatively free from strong emission or absorption lines and thus representative of the underlying continuum. This often involves prior knowledge of the expected spectral features for the object type or iterative masking based on outlier rejection relative to an initial smooth fit. Masking strong absorption or emission lines is crucial to prevent them from biasing the continuum fit downwards or upwards, respectively. Regions affected by poor sky subtraction residuals or detector artifacts should also be masked.
2.  **Model Selection:** Choose a mathematical function to represent the continuum shape across the selected regions. Common choices include:
    *   **Polynomials (`astropy.modeling.models.Polynomial1D`, `Chebyshev1D`, `Legendre1D`):** Low-to-moderate order polynomials (e.g., degree 3 to 10) are frequently used for fitting relatively smooth continuum shapes over limited wavelength ranges. Orthogonal polynomials like Chebyshev or Legendre are often preferred for numerical stability, especially for higher orders (García-Benito & Jurado, 2023). The order must be chosen carefully – too low may not capture real continuum curvature, while too high can lead to overfitting noise or trying to fit broad spectral features as continuum.
    *   **Splines (`astropy.modeling.models.Spline1D`):** Piecewise spline functions offer more flexibility than global polynomials for modeling complex or irregular continuum shapes. They involve fitting smooth polynomial segments between defined "knots" or breakpoints along the wavelength axis. The number and placement of knots control the flexibility of the fit. Splines can adapt well to local variations but require careful knot placement or smoothing parameters to avoid overfitting.
    *   **Running Median/Mean Filters:** Applying a wide median or mean filter to the spectrum (after masking strong lines) can provide a simple estimate of the continuum, particularly effective at removing high-frequency noise while following broad continuum trends. The window size of the filter is the key parameter.
    *   **Physical Models:** In some cases, simplified physical models (e.g., blackbody function - `astropy.modeling.models.BlackBody1D`, power law - `astropy.modeling.models.PowerLaw1D`) might be appropriate if the continuum emission mechanism is well understood, although these are often used for broader SED fitting rather than local continuum normalization within a single spectrum.
3.  **Fitting Algorithm:** Fit the chosen model function to the flux values in the selected/unmasked continuum regions of the spectrum. Weighted least-squares fitting is typically used, where points might be weighted by their inverse variance (uncertainty squared) to give more influence to higher signal-to-noise regions. Iterative fitting with outlier rejection (sigma clipping) is highly recommended: fit the model, calculate residuals, reject points deviating significantly (e.g., > 3-sigma) from the fit (which might be previously unmasked faint lines or noise spikes), and refit the model to the remaining points. This iterative process helps ensure the fit is robustly tracking the true continuum level.
4.  **Evaluation and Normalization:** Evaluate the fitted continuum model function $C(\lambda)$ across the *entire* wavelength range of the spectrum. Visually inspect the fit overlaid on the spectrum to ensure it provides a reasonable representation of the continuum, particularly checking that it passes through appropriate points between lines and doesn't dip into absorption features or rise over emission features. The **continuum-normalized spectrum** $F_{norm}(\lambda)$ is then obtained by dividing the original observed flux $F_{obs}(\lambda)$ by the fitted continuum model $C(\lambda)$:
    $F_{norm}(\lambda) = \frac{F_{obs}(\lambda)}{C(\lambda)}$
    The resulting $F_{norm}(\lambda)$ should fluctuate around a value of 1.0 in continuum regions. Uncertainties must be propagated correctly during this division: $\sigma_{norm}(\lambda) \approx \frac{F_{norm}(\lambda)}{\mathrm{SNR}(\lambda)}$, where $\mathrm{SNR}(\lambda)$ is the signal-to-noise ratio of the original spectrum, assuming the uncertainty in the continuum fit itself is negligible (often a reasonable approximation if the fit is based on many high-SNR points, but not always valid).

The **`specutils.fitting`** module provides convenient functions specifically designed for continuum fitting and normalization of `Spectrum1D` objects. Functions like `fit_generic_continuum` allow fitting various `astropy.modeling` models (polynomials, splines, etc.) to spectral data, often incorporating options for masking spectral windows or performing iterative sigma clipping during the fit. Helper functions may also assist in identifying continuum regions automatically based on local noise properties. Once a continuum model is fitted (returned as an `astropy.modeling` model instance), the spectrum can be easily normalized by simple division using the model evaluated at the spectrum's spectral axis coordinates. Proper continuum normalization is essential for accurate measurement of equivalent widths (Section 7.3.2) and for comparing line profiles between different objects or observations.

The following Python example demonstrates fitting a polynomial continuum to a simulated spectrum using `specutils.fitting.fit_generic_continuum` and then normalizing the spectrum by dividing by the fitted model. It highlights the importance of excluding potential line regions from the fit using the `exclude_regions` parameter.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_generic_continuum
    specutils_available = True
except ImportError:
    print("specutils not found, skipping continuum fitting example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_generic_continuum(spectrum, model=None, exclude_regions=None): return None # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling.models import Polynomial1D, Chebyshev1D # Example models
import matplotlib.pyplot as plt

# --- Simulate Spectrum with Continuum and Lines ---
if specutils_available:
    # Define wavelength axis
    wavelengths = np.linspace(6000, 7000, 500) * u.AA
    # Simulate a sloped continuum + Gaussian emission line + noise
    true_continuum_model = Polynomial1D(degree=2, c0=100, c1=0.02, c2=-1e-6) # Quadratic continuum
    true_continuum = true_continuum_model(wavelengths.value)
    emission_line = 50 * np.exp(-0.5 * ((wavelengths.value - 6563) / 10.0)**2) # H-alpha emission
    noise = np.random.normal(0, 5.0, size=wavelengths.shape)
    flux_observed = (true_continuum + emission_line + noise) * u.Jy # Example flux units

    # Create the Spectrum1D object
    observed_spectrum = Spectrum1D(flux=flux_observed, spectral_axis=wavelengths)
    print("Simulated spectrum created.")

    # --- Define Regions to Exclude from Continuum Fit ---
    # Exclude the region around the known emission line (e.g., H-alpha at 6563 A)
    # Create a SpectralRegion object defining the window(s) to mask
    # Define region slightly wider than the line itself
    halpha_region = SpectralRegion(6563*u.AA - 30*u.AA, 6563*u.AA + 30*u.AA)
    exclude_regions_list = [halpha_region]
    print(f"Defined exclusion region: {exclude_regions_list[0]}")

    # --- Fit the Continuum using specutils ---
    print("Fitting continuum using fit_generic_continuum...")
    # Choose a model to fit (e.g., Chebyshev polynomial for stability)
    # Degree needs careful selection based on expected continuum shape
    fit_degree = 2
    continuum_model_fit = Chebyshev1D(degree=fit_degree)

    # Use fit_generic_continuum, providing the spectrum, model, and exclude regions
    # This function internally handles masking and fitting
    try:
        fitted_continuum_model = fit_generic_continuum(
            observed_spectrum,
            model=continuum_model_fit,
            exclude_regions=exclude_regions_list
            # Can add 'weights' argument if uncertainty is available
            # Can add 'fitter' for more control (e.g., with outlier rejection)
        )
        print(f"Continuum fitted with {type(fitted_continuum_model).__name__} (degree={fit_degree}).")

        # Evaluate the fitted continuum model over the full wavelength range
        continuum_fit_values = fitted_continuum_model(observed_spectrum.spectral_axis)

        # --- Normalize the Spectrum ---
        print("Normalizing spectrum by the fitted continuum...")
        # Divide the observed spectrum by the fitted continuum values
        # specutils handles units correctly during division
        normalized_spectrum = observed_spectrum / continuum_fit_values

        print("Continuum normalization complete.")
        # Check continuum level near 1.0 away from the line
        print(f"  Normalized flux near 6400A: {normalized_spectrum.flux[np.argmin(np.abs(normalized_spectrum.spectral_axis.value - 6400))]:.3f}")
        print(f"  Normalized flux near 6800A: {normalized_spectrum.flux[np.argmin(np.abs(normalized_spectrum.spectral_axis.value - 6800))]:.3f}")

        # --- Optional: Plotting ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Plot original spectrum and fitted continuum
        axes[0].plot(observed_spectrum.spectral_axis, observed_spectrum.flux, label='Observed Spectrum')
        axes[0].plot(observed_spectrum.spectral_axis, continuum_fit_values, label=f'Fitted Continuum (Chebyshev{fit_degree})', color='red', linestyle='--')
        axes[0].set_ylabel(f"Flux ({observed_spectrum.flux.unit})")
        axes[0].set_title("Continuum Fitting")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Plot normalized spectrum
        axes[1].plot(normalized_spectrum.spectral_axis, normalized_spectrum.flux, label='Normalized Spectrum')
        axes[1].axhline(1.0, color='grey', linestyle=':') # Show continuum level = 1
        axes[1].set_xlabel(f"Wavelength ({normalized_spectrum.spectral_axis.unit})")
        axes[1].set_ylabel("Normalized Flux")
        axes[1].set_title("Continuum Normalized Spectrum")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.5, 1.5 + np.max(normalized_spectrum.flux.value[50:-50]) - 1) # Adjust ylim for better view

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Error: specutils or astropy.modeling might be missing.")
    except Exception as e:
        print(f"An unexpected error occurred during continuum fitting: {e}")
else:
    print("Skipping continuum fitting example: specutils unavailable.")

```

This Python script demonstrates the process of continuum fitting and normalization applied to a simulated astronomical spectrum using the `specutils` library. It begins by creating a `Spectrum1D` object containing simulated flux data with a defined continuum shape and an emission line, along with its corresponding wavelength axis. Crucially, it defines a `SpectralRegion` object to specify the wavelength range occupied by the emission line, which should be excluded from the continuum fitting process to avoid biasing the fit. The core fitting operation is performed by `specutils.fitting.fit_generic_continuum`, which takes the observed spectrum, a chosen `astropy.modeling` model (here, a `Chebyshev1D` polynomial), and the list of `exclude_regions` as input. This function robustly fits the model to the spectrum while ignoring the specified exclusion regions. The fitted continuum model is then evaluated across the full spectral range, and the original observed spectrum is divided by this fitted continuum using simple division (which `specutils` handles correctly with units). The resulting `normalized_spectrum` object has its continuum level approximately at unity, making the emission line profile relative to the continuum clearly visible and ready for quantitative measurement. The plots effectively visualize the original spectrum with the overlaid continuum fit and the final normalized spectrum.

**7.2 Spectral Feature Identification**

Once a spectrum has been reduced, calibrated, and potentially continuum-normalized, the next step in analysis is often the identification of specific spectral features – emission lines or absorption lines – that correspond to known atomic or molecular transitions. This identification process is crucial for interpreting the physical nature of the observed object. For example, recognizing specific patterns of absorption lines (like the Balmer series of hydrogen) can reveal the spectral type and temperature of a star, while identifying specific emission lines (like [O III] 5007 Å or H$\alpha$ 6563 Å) can indicate the presence of ionized gas in nebulae or active galactic nuclei (AGN) and be used to measure redshifts for galaxies (Osterbrock & Ferland, 2006).

The identification process involves comparing the observed features in the spectrum with reference **line lists**. These lists contain accurately measured laboratory wavelengths ($\lambda_{rest}$) for transitions of various elements and ions under specific physical conditions. Numerous astronomical line lists are available, often compiled in databases or accessible through tools like `astroquery` or included within spectral analysis packages. Examples include lists for common nebular emission lines, stellar photospheric absorption lines (e.g., from NIST Atomic Spectra Database, VALD - Piskunov et al., 1995; Ryabchikova et al., 2015), or molecular transitions relevant for planetary atmospheres or the interstellar medium.

Feature identification can be performed through several methods:
*   **Visual Inspection:** An experienced astronomer can often visually identify prominent lines or patterns in a plotted spectrum by comparing their observed wavelengths ($\lambda_{obs}$) with known rest wavelengths ($\lambda_{rest}$) from relevant line lists, especially if the object type and approximate redshift (if applicable) are known beforehand. Plotting tools that allow interactive zooming, cursor position reading (wavelength, flux), and overlaying reference line markers are invaluable for this process. This method relies heavily on expertise and can be subjective or time-consuming for complex spectra.
*   **Automated Line Finding and Matching:** Algorithms can be employed to automatically detect significant peaks (for emission lines) or troughs (for absorption lines) in the spectrum that exceed a certain signal-to-noise threshold or depth relative to the continuum/noise level. The measured central wavelengths ($\lambda_{obs}$) of these detected features can then be automatically compared against reference line lists.
    *   If the object's approximate redshift ($z$) is known, the expected observed wavelength for a line with rest wavelength $\lambda_{rest}$ is $\lambda_{exp} = \lambda_{rest}(1+z)$. Detected lines near $\lambda_{exp}$ within some tolerance are potential identifications.
    *   If the redshift is unknown, pattern matching techniques can be used. Ratios of wavelengths or separations between multiple detected lines can be compared to ratios expected from specific ions or common patterns (e.g., Balmer series ratios, doublet separations) in reference lists to find potential matches and simultaneously estimate the redshift.
    *   **Template Cross-Correlation (primarily for redshift):** While often used for redshift determination (Section 7.5), cross-correlating the observed spectrum with template spectra containing known features can also implicitly identify lines based on the alignment that maximizes the correlation.
*   **Model Fitting:** Simultaneously fitting multiple spectral features (e.g., a series of lines from the same element) with appropriate profile models (Section 7.3.4) where the relative line positions are constrained by laboratory values can provide robust identifications and measurements, especially in blended spectra.

Software packages like `specutils` provide tools that can assist with identification, such as functions for finding peaks (`find_lines_threshold`, `find_lines_derivative`) and potentially tools for comparing detected features to line lists, although robust automated matching often requires more specialized algorithms or integration with external databases. Successful identification requires accurate wavelength calibration (Section 4.5), knowledge of potential line blending, and consideration of the object's expected physical conditions (temperature, density, ionization state) which determine which transitions are likely to be prominent. Misidentification of lines can lead to significant errors in derived physical parameters or velocities/redshifts.

**7.3 Line Property Measurement (`specutils.analysis`)**

Once spectral lines have been identified, quantifying their properties provides crucial diagnostic information about the emitting or absorbing gas. Key properties include the line's central position (which yields velocity or redshift), its total strength (related to abundance or column density), and its width (probing temperature, turbulence, rotation, or instrumental effects). The **`specutils.analysis`** module offers functions for measuring several of these fundamental properties directly from `Spectrum1D` objects.

*   **7.3.1 Centroid Determination (Velocity/Redshift)**
    Determining the precise central wavelength ($\lambda_{cen}$) of a spectral line is fundamental for measuring the line-of-sight velocity ($v_{los}$) of the source relative to the observer via the Doppler effect, or its cosmological redshift ($z$). The Doppler shift relates the observed central wavelength ($\lambda_{obs} = \lambda_{cen}$) to the rest wavelength ($\lambda_{rest}$) of the transition in the source's frame:
    $\frac{\lambda_{obs} - \lambda_{rest}}{\lambda_{rest}} = \frac{\Delta \lambda}{\lambda_{rest}} \approx \frac{v_{los}}{c}$ (for non-relativistic velocities $v_{los} \ll c$)
    For cosmological sources, the redshift $z$ is defined as:
    $z = \frac{\lambda_{obs} - \lambda_{rest}}{\lambda_{rest}}$ or $1+z = \frac{\lambda_{obs}}{\lambda_{rest}}$
    Accurate determination of $\lambda_{cen}$ is therefore critical. Several methods can be used:
    *   **Weighted Mean (Centroid):** Calculate the flux-weighted average wavelength across the line profile:
        $\lambda_{cen} = \frac{\int \lambda F(\lambda) d\lambda}{\int F(\lambda) d\lambda}$
        where $F(\lambda)$ is the continuum-subtracted flux (for emission lines) or $1 - F_{norm}(\lambda)$ (for absorption lines, where $F_{norm}$ is the normalized flux), and the integral is performed over the wavelength range of the line. `specutils.analysis.centroid` implements this, potentially using different weighting schemes. Simple centroids can be biased by noise or asymmetry.
    *   **Gaussian Fitting:** Fit a Gaussian profile model (Section 7.3.4) to the line. The center parameter ($\mu$) of the fitted Gaussian provides a robust estimate of $\lambda_{cen}$, particularly for symmetric lines, and the uncertainty on the fitted center provides an estimate of the centroid measurement error.
    *   **Other Profile Fitting:** Fitting more complex profiles (Lorentzian, Voigt) also yields the line center as a fit parameter.
    *   **Minimum/Maximum Finding:** For very sharp, high-SNR lines, simply finding the wavelength of the peak pixel (emission) or minimum pixel (absorption) can provide a quick estimate, but lacks sub-pixel precision and robustness.
    `specutils.analysis.line_flux` (when used carefully on continuum-subtracted data) and related functions can assist, but dedicated profile fitting (Section 7.3.4) is often preferred for highest precision centroiding. The uncertainty in the centroid measurement depends on the line's signal-to-noise ratio, its width, and the spectral resolution/sampling.

*   **7.3.2 Equivalent Width Calculation**
    The **equivalent width (EW)** of a spectral line provides a measure of its total strength relative to the continuum level, independent of the absolute flux calibration (as long as the continuum is correctly normalized). It represents the width of a hypothetical rectangle, extending from zero flux (for absorption) or zero flux above continuum (for emission) up to the continuum level, that has the same integrated area as the line profile itself.
    For an absorption line in a continuum-normalized spectrum $F_{norm}(\lambda)$, the equivalent width is defined as:
    $EW = \int_{\text{line}} (1 - F_{norm}(\lambda)) d\lambda$
    For an emission line with flux $F_{em}(\lambda)$ above a continuum level $C(\lambda)$, it can be defined as:
    $EW_{em} = \int_{\text{line}} \frac{F_{em}(\lambda)}{C(\lambda)} d\lambda = \int_{\text{line}} \frac{F_{obs}(\lambda) - C(\lambda)}{C(\lambda)} d\lambda$
    The integration is performed over the wavelength range spanned by the line profile. The EW has units of wavelength (e.g., Å, nm, mÅ). Physically, the EW of an absorption line is related to the number of absorbing atoms or ions along the line of sight (column density) and the transition's oscillator strength, often analyzed using curve-of-growth analysis (Spitzer, 1978). For emission lines, it relates the line luminosity to the underlying continuum luminosity.
    EW can be calculated using:
    *   **Direct Summation:** Numerically integrating the definition above using the continuum-normalized flux values (or continuum-subtracted flux divided by continuum for emission) over the pixels spanning the line profile. The wavelength interval $d\lambda$ corresponds to the width of each wavelength bin ($\Delta\lambda_{pix}$).
        $EW \approx \sum_{i \in \text{line}} (1 - F_{norm, i}) \Delta\lambda_{pix, i}$
        `specutils.analysis.equivalent_width` implements this direct summation method, requiring the continuum level to be implicitly or explicitly handled (e.g., by operating on a normalized spectrum).
    *   **Profile Fitting:** Fit an appropriate profile model (e.g., Gaussian, Voigt) to the line. The integrated area of the fitted profile (analytically calculable from the fit parameters like amplitude and width) directly gives the equivalent width (if the input spectrum was normalized or the continuum level used in the fit). For a Gaussian fit $A \exp(-(\lambda-\mu)^2 / (2\sigma^2))$ to an absorption line in a normalized spectrum, $EW = A \sigma \sqrt{2\pi}$. This method can be more robust against noise than direct summation.
    Accurate EW measurement requires precise continuum placement/normalization and defining the integration limits carefully to include the full line profile wings without excessive noise contribution.

*   **7.3.3 Full Width at Half Maximum (FWHM) Measurement**
    The **Full Width at Half Maximum (FWHM)** characterizes the width of a spectral line profile. It is defined as the full width of the line, measured in wavelength (or velocity) units, at a level corresponding to half the peak intensity (above continuum for emission, below continuum for absorption). The FWHM provides insights into the physical processes broadening the line profile:
    *   **Natural Broadening:** Intrinsic width due to the uncertainty principle (usually negligible).
    *   **Thermal Broadening:** Doppler shifts due to the random thermal motion of atoms/ions. Width depends on temperature ($T$) and particle mass ($m$): $FWHM_{thermal} \propto \sqrt{T/m}$.
    *   **Turbulent Broadening:** Doppler shifts from macroscopic, non-thermal turbulent motions within the gas.
    *   **Rotational Broadening:** Doppler shifts across the surface of a rotating object (e.g., a star) can significantly broaden lines.
    *   **Pressure/Stark Broadening:** Perturbations of energy levels due to collisions or electric fields (important in dense stellar atmospheres).
    *   **Instrumental Broadening:** The finite spectral resolution of the spectrograph itself broadens all observed lines. The instrumental profile (Line Spread Function, LSF) needs to be known or measured (e.g., from narrow arc lines) to disentangle intrinsic source broadening from instrumental effects, often through deconvolution or by fitting models convolved with the LSF.
    FWHM can be measured by:
    *   **Direct Measurement:** Find the peak (or minimum) flux level ($F_{peak}$ or $F_{min}$) relative to the continuum ($F_{cont}$). Determine the half-maximum/minimum level ($F_{half} = F_{cont} \pm (F_{peak/min} - F_{cont})/2$). Find the two wavelengths ($\lambda_1, \lambda_2$) on either side of the line center where the flux equals $F_{half}$. The FWHM is then $|\lambda_2 - \lambda_1|$. This requires interpolation between data points for accuracy. `specutils.analysis.fwhm` provides functionality for this direct measurement.
    *   **Profile Fitting:** Fit a profile model (Gaussian, Voigt, etc.) to the line. The FWHM can be calculated directly from the fitted width parameter ($\sigma$ for Gaussian, related parameters for Lorentzian/Voigt). For a Gaussian, $FWHM = 2\sqrt{2\ln 2} \sigma \approx 2.355 \sigma$. This is generally more robust against noise.
    Measuring FWHM accurately requires good SNR and spectral sampling (sufficient pixels across the line profile).

*   **7.3.4 Line Profile Fitting (`astropy.modeling`)**
    Fitting mathematical models directly to the observed line profile provides the most detailed characterization, yielding parameters like the precise center ($\lambda_{cen}$), amplitude or depth ($A$), width ($\sigma$ or related parameters), and potentially parameters describing profile shape (e.g., Gaussian vs. Lorentzian component in a Voigt profile). This is particularly powerful for analyzing line shapes to probe physical conditions, measuring parameters robustly in noisy data, and **deblending** closely spaced or overlapping lines by simultaneously fitting multiple components.
    1.  **Model Selection:** Choose an appropriate profile function based on the expected physics and observed shape:
        *   **Gaussian (`Gaussian1D`):** Often used for lines broadened primarily by thermal/turbulent Doppler effects or dominated by instrumental broadening if the LSF is approximately Gaussian. Defined by amplitude $A$, mean $\mu$ ($\lambda_{cen}$), and standard deviation $\sigma$.
        *   **Lorentzian (`Lorentz1D`):** Describes natural broadening or pressure broadening. Defined by amplitude $A$, center $x_0$ ($\lambda_{cen}$), and FWHM $\gamma$. Has broader wings than a Gaussian.
        *   **Voigt (`Voigt1D`):** A convolution of a Gaussian and a Lorentzian profile. Physically represents lines subject to both Doppler (Gaussian) and damping/pressure (Lorentzian) broadening, common in stellar atmospheres or damped Lyman-alpha systems. Defined by amplitude $A$, center $x_0$ ($\lambda_{cen}$), Gaussian width $\sigma$, and Lorentzian width $\gamma$. Requires numerical evaluation.
        *   **Custom Models:** More complex or asymmetric profiles can be built using `astropy.modeling.custom_model` or by combining standard models (e.g., multiple Gaussians for complex emission features).
    2.  **Continuum Handling:** The model must be fitted to the line *plus* the underlying continuum. Either fit the line profile model added to a continuum model (e.g., `Polynomial1D + Gaussian1D`), or fit the profile model directly to the continuum-subtracted (or normalized) spectrum. Fitting simultaneously with the continuum is often more robust if the continuum level near the line is uncertain.
    3.  **Fitting Procedure:** Use a non-linear least-squares fitting algorithm (e.g., `astropy.modeling.fitting.LevMarLSQFitter` or `TRFLSQFitter`) to optimize the parameters of the chosen model (line parameters + continuum parameters if included) to best match the observed flux data points across the line profile. Providing good initial guesses for the parameters (center, amplitude, width) is often crucial for the fitter to converge correctly. Weighting the fit by inverse variance is recommended.
    4.  **Parameter Extraction and Uncertainties:** The best-fit parameters ($\mu, \sigma, A, \gamma$, etc.) and their statistical uncertainties (often estimated from the covariance matrix returned by the fitter) provide the quantitative measurements of the line properties. Derived quantities like FWHM or EW can be calculated from these fitted parameters, including propagation of uncertainties.
    5.  **Goodness-of-Fit:** Assess the quality of the fit using metrics like reduced chi-squared ($\chi^2_\nu$) or by visually inspecting the residuals (data - model). Poor fits may indicate an inappropriate model choice, underestimated uncertainties, or unaccounted-for line blending.
    Fitting line profiles using `astropy.modeling` integrated with `specutils` provides a powerful and flexible framework for detailed quantitative spectroscopic analysis.

**7.4 Spectroscopic Index Measurement**

While detailed line profile analysis provides rich information, sometimes a quicker characterization of spectral features, particularly broader absorption bands common in stellar or galaxy spectra, is desired. **Spectroscopic indices** are designed for this purpose. They quantify the strength of a specific spectral feature by comparing the flux within a defined central "feature" bandpass to the flux in one or more adjacent "pseudo-continuum" bandpasses located on either side of the feature (Worthey et al., 1994; Trager et al., 1998). These indices are often defined relative to standard systems, most notably the **Lick/IDS system**, developed to study stellar populations in integrated light from galaxies.

A typical index definition involves:
*   **Feature Bandpass:** A central wavelength range $(\lambda_{F1}, \lambda_{F2})$ encompassing the absorption line or band of interest.
*   **Continuum Bandpasses:** One or two wavelength regions, typically on the blue side $(\lambda_{C1B}, \lambda_{C2B})$ and/or red side $(\lambda_{CR1}, \lambda_{CR2})$ of the feature bandpass, chosen to be relatively free of strong lines and representative of the local continuum level.
*   **Index Calculation:** The index is calculated based on the average flux density within these bandpasses. Common definitions include:
    *   **Magnitude-based indices (e.g., Lick system):** The index is expressed in magnitudes, representing a color difference:
        $I_{mag} = -2.5 \log_{10} \left( \frac{\int_{\lambda_{F1}}^{\lambda_{F2}} F(\lambda) d\lambda / (\lambda_{F2}-\lambda_{F1})}{\int_{\text{Continuum}} F_C(\lambda) d\lambda / \Delta\lambda_C} \right)$
        where $F_C(\lambda)$ is the estimated continuum flux (often interpolated linearly in $F_\lambda$ between the average fluxes in the blue and red continuum bandpasses) integrated over the feature bandpass $\Delta\lambda_C = (\lambda_{F2}-\lambda_{F1})$.
    *   **Equivalent Width-like indices:** Some indices are defined similarly to equivalent widths, representing the area lost due to absorption relative to the pseudo-continuum:
        $I_{EW} = \int_{\lambda_{F1}}^{\lambda_{F2}} \left( 1 - \frac{F(\lambda)}{F_C(\lambda)} \right) d\lambda$
    *   **Ratio indices:** Simpler ratios of flux in the feature band to flux in continuum bands.

Notable examples include:
*   **Lick Indices:** A set of ~25 indices defined in the optical range (e.g., Mg$_2$, Fe5270, Fe5335, H$\beta$, G4300) primarily measuring strengths of absorption features sensitive to stellar age, metallicity, and abundance ratios in integrated stellar populations (Trager et al., 1998).
*   **D4000 Break (D$_n$(4000)):** Measures the strength of the discontinuity in galaxy spectra around 4000 Å, caused by an accumulation of absorption lines (primarily Ca II H&K) and opacity effects in cooler stellar atmospheres. It is a strong indicator of the average age of the stellar population (older populations have stronger breaks) (Bruzual, 1983; Balogh et al., 1999). Defined as the ratio of average flux density in a red pseudo-continuum band (e.g., 4000-4100 Å) to that in a blue band (e.g., 3850-3950 Å).

Calculating these indices requires a spectrum with accurate relative flux calibration (though not necessarily absolute) and precise wavelength calibration. The spectrum should ideally be smoothed to match the resolution of the original Lick/IDS system if direct comparison is intended. Dedicated functions or workflows exist in astronomical software packages (sometimes external to core libraries like `specutils`) for calculating standard indices based on their specific bandpass definitions. Spectroscopic indices provide valuable, albeit simplified, integrated information about stellar populations and galaxy properties, complementary to detailed line fitting analysis.

**7.5 Redshift Determination Techniques**

Determining the **redshift ($z$)** of extragalactic objects (galaxies, quasars) is one of the most fundamental measurements in cosmology and extragalactic astronomy. Redshift quantifies the amount by which the object's light has been stretched to longer wavelengths due to the expansion of the Universe (Hubble's Law) plus any contribution from peculiar velocities. Measuring redshift allows estimation of the object's distance, enabling studies of large-scale structure, galaxy evolution over cosmic time, and cosmological parameter estimation. Several techniques are used to measure redshift from spectra (Bolton et al., 2012; Comparat et al., 2023).

1.  **Emission/Absorption Line Fitting:** If prominent spectral features (either emission lines from ionized gas in galaxies/AGN or absorption lines from stars/interstellar medium/intergalactic medium) can be reliably identified (Section 7.2), their observed central wavelengths ($\lambda_{obs}$) can be measured accurately (Section 7.3.1). Comparing these to the known rest wavelengths ($\lambda_{rest}$) of the identified transitions directly yields the redshift for each line:
    $z_i = \frac{\lambda_{obs, i} - \lambda_{rest, i}}{\lambda_{rest, i}}$
    Measuring multiple lines is crucial. A consistent redshift value derived from several different identified lines provides a robust redshift determination. The final redshift is typically taken as the uncertainty-weighted average of the values obtained from individual lines. This method requires clear identification of spectral features and accurate wavelength calibration. It is the most precise method when strong, identifiable lines are present.
2.  **Cross-Correlation:** This technique is particularly powerful when individual spectral lines are weak or blended, or when dealing with spectra dominated by broad absorption features (like typical galaxy spectra dominated by starlight). It works by comparing the observed spectrum with template spectra of objects with known redshifts (e.g., template spectra representing different galaxy types, stellar types, or quasars). The core idea is to find the redshift $z$ at which a chosen template spectrum $T(\lambda_{rest})$, when redshifted to $\lambda = \lambda_{rest}(1+z)$ and potentially broadened to match the observed spectrum's resolution, provides the best match to the observed spectrum $O(\lambda)$. The matching is typically quantified by calculating the cross-correlation function (CCF) between the observed spectrum and the redshifted template as a function of trial redshift $z$:
    $CCF(z) = \int O(\lambda) T\left(\frac{\lambda}{1+z}\right) d\lambda$
    (Often performed in log-wavelength space where redshift becomes a simple linear shift, allowing use of Fast Fourier Transforms (FFTs) for efficiency). The redshift $z$ corresponding to the peak of the CCF represents the best-fit redshift where the spectral features (absorption lines, continuum breaks) in the template optimally align with those in the observed spectrum.
    *   **Steps:**
        *   Prepare observed spectrum (e.g., continuum subtract or normalize, resample to log-wavelength grid).
        *   Select appropriate template spectrum(s) based on expected object type. Multiple templates are often used.
        *   For each template, calculate the CCF over a plausible range of redshifts.
        *   Identify the peak(s) in the CCF for each template.
        *   Determine the best redshift based on the highest CCF peak, potentially considering the quality of the peak (e.g., height relative to noise, width) and consistency across different templates.
    *   **Libraries:** Python packages like `specutils` offer basic cross-correlation functionality, while dedicated redshift pipelines used by surveys (e.g., SDSS pipeline, DESI pipeline based on `redrock` - Bailey et al., 2023) implement highly optimized and robust cross-correlation algorithms using large libraries of templates and sophisticated peak finding and quality assessment logic.
    Cross-correlation is less precise than line fitting for spectra with strong emission lines but is highly effective for absorption-line dominated spectra and provides a robust method for large surveys where automated redshift determination is required for millions of objects. The accuracy depends on the quality of the spectrum (SNR), the suitability of the templates, and the wavelength range covered.

The choice between line fitting and cross-correlation depends on the nature of the spectrum. Often, both methods are used, with line fitting providing high precision when possible, and cross-correlation providing robustness and handling spectra lacking strong emission features. Accurate redshift measurement is foundational for much of modern extragalactic astronomy and cosmology.

**7.6 Examples in Practice (Python): Spectral Analysis Tasks**

The following examples demonstrate the practical application of spectroscopic analysis techniques discussed in this chapter using Python libraries, primarily `specutils` and `astropy.modeling`. Each example targets a specific astronomical scenario, showcasing how to measure line properties (centroids/velocities, equivalent widths, FWHM, profile fits) and determine redshifts. These examples aim to provide concrete illustrations of the workflows involved in extracting physical information from calibrated spectra across different scientific domains.

**7.6.1 Solar: Measuring Doppler Shifts using `specutils`**
Solar spectroscopy often aims to measure plasma motions on the Sun by detecting small Doppler shifts in the central wavelengths of photospheric or chromospheric absorption lines relative to their known rest wavelengths. These shifts reveal phenomena like solar rotation, convective flows (granulation), wave propagation, or outflows associated with active regions. This example demonstrates how to load a solar spectrum, fit a Gaussian profile to a specific absorption line using `specutils`'s interface to `astropy.modeling`, extract the fitted line center, and calculate the corresponding Doppler velocity relative to the line's known rest wavelength. Accurate continuum normalization or subtraction around the line is implicitly assumed for precise fitting.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_lines # Function to fit models to lines
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Solar Doppler example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_lines(spectrum, model, window=None): return model # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models # For Gaussian1D model
import astropy.constants as const # For speed of light c
import matplotlib.pyplot as plt

# --- Simulate Solar Spectrum Segment around a Line ---
if specutils_available:
    # Define wavelength axis (e.g., around Fe I line at 5434.5 A)
    line_rest_wavelength = 5434.527 * u.AA
    doppler_shift_kms = 1.5 * u.km / u.s # Introduce a small velocity shift
    # Calculate observed wavelength
    line_observed_wavelength = line_rest_wavelength * (1 + doppler_shift_kms / const.c)

    wavelengths = np.linspace(line_observed_wavelength.value - 1.0, line_observed_wavelength.value + 1.0, 200) * u.AA
    # Simulate continuum normalized spectrum with absorption line + noise
    # Gaussian profile: Amplitude is depth (negative for absorption in norm spec)
    line_depth = 0.6 # 60% depth relative to continuum
    line_sigma = 0.08 * u.AA # Line width (standard deviation)
    # Model = 1 - Gaussian
    true_profile = 1.0 - line_depth * np.exp(-0.5 * ((wavelengths - line_observed_wavelength) / line_sigma)**2)
    noise = np.random.normal(0, 0.02, size=wavelengths.shape) # Noise on normalized flux
    flux_normalized = true_profile + noise

    # Create Spectrum1D object (continuum normalized)
    solar_spec_norm = Spectrum1D(flux=flux_normalized, spectral_axis=wavelengths)
    print("Simulated normalized solar spectrum segment created.")

    # --- Fit Gaussian Profile to the Absorption Line ---
    print(f"Fitting Gaussian profile to line near {line_rest_wavelength:.2f}...")
    # Define the Gaussian model to fit (absorption line)
    # Amplitude is negative, tied to continuum=1. Provide initial guesses.
    # Use Gaussian1D for normalized data: 1 + Gaussian1D() where amplitude < 0
    # Or fit Gaussian1D to (1 - flux_normalized)
    # Let's fit Gaussian1D to the absorption depth profile: 1 - flux
    absorption_depth = 1.0 - solar_spec_norm.flux
    # Need to handle potential negative values if noise dips below 0 in normalized flux
    absorption_depth = np.maximum(absorption_depth.value, 0) # Ensure non-negative depth

    # Initial guesses for parameters: amplitude, mean (center), stddev
    amp_guess = line_depth # Positive amplitude for depth profile
    mean_guess = line_observed_wavelength # Use expected observed center
    stddev_guess = line_sigma

    gauss_init = models.Gaussian1D(amplitude=amp_guess, mean=mean_guess, stddev=stddev_guess)
    # Fit the model using specutils.fitting.fit_lines
    # Need to provide the spectrum of the depth profile
    depth_spectrum = Spectrum1D(flux=absorption_depth, spectral_axis=solar_spec_norm.spectral_axis)
    # Define a spectral window around the line for fitting
    fit_window = SpectralRegion(mean_guess - 3*stddev_guess, mean_guess + 3*stddev_guess)

    try:
        # fit_lines takes the spectrum object and the model instance
        fitted_model = fit_lines(depth_spectrum, gauss_init, window=fit_window)
        print("Gaussian profile fitted.")
        print("Fitted Parameters:")
        print(f"  Amplitude (Depth): {fitted_model.amplitude.value:.3f}")
        print(f"  Mean (Observed Wavelength): {fitted_model.mean.quantity:.4f}")
        print(f"  Standard Deviation: {fitted_model.stddev.quantity:.4f}")

        # Extract the fitted central wavelength
        lambda_center_fitted = fitted_model.mean.quantity # Includes units

        # --- Calculate Doppler Velocity ---
        print("\nCalculating Doppler velocity...")
        # v = c * (lambda_obs - lambda_rest) / lambda_rest
        doppler_velocity = const.c * (lambda_center_fitted - line_rest_wavelength) / line_rest_wavelength
        # Convert velocity to km/s
        doppler_velocity_kms = doppler_velocity.to(u.km / u.s)

        print(f"  Rest Wavelength: {line_rest_wavelength:.4f}")
        print(f"  Fitted Observed Wavelength: {lambda_center_fitted:.4f}")
        print(f"  Calculated Doppler Velocity: {doppler_velocity_kms:.2f}")
        print(f"  (Input simulated velocity was: {doppler_shift_kms:.2f})")

        # --- Optional: Plot Fit ---
        plt.figure(figsize=(8, 5))
        plt.plot(solar_spec_norm.spectral_axis, solar_spec_norm.flux, label='Simulated Data (Norm Flux)', drawstyle='steps-mid')
        # Plot the fitted profile (remember we fitted 1 - flux)
        plt.plot(solar_spec_norm.spectral_axis, 1.0 - fitted_model(solar_spec_norm.spectral_axis.value),
                 label=f'Gaussian Fit (Center={lambda_center_fitted:.3f})', color='red', linestyle='--')
        plt.xlabel(f"Wavelength ({solar_spec_norm.spectral_axis.unit})")
        plt.ylabel("Normalized Flux")
        plt.title(f"Solar Line Profile Fit (Doppler Shift: {doppler_velocity_kms:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred during fitting or velocity calculation: {e}")
else:
    print("Skipping Solar Doppler example: specutils unavailable.")

```

This Python script demonstrates the measurement of Doppler shifts in solar spectra, a key technique for studying plasma motions on the Sun. It simulates a segment of a continuum-normalized solar spectrum containing an absorption line with a known, small Doppler shift introduced relative to its rest wavelength. The core analysis uses `specutils.fitting.fit_lines` in conjunction with an `astropy.modeling.models.Gaussian1D` model to fit the observed absorption line profile (specifically, fitting the depth profile $1 - F_{norm}$). The function returns the best-fit parameters for the Gaussian model, including the precise central wavelength ($\lambda_{center\_fitted}$) of the observed line. This fitted central wavelength is then compared to the known rest wavelength ($\lambda_{rest}$) of the spectral line using the Doppler formula ($v = c (\lambda_{obs} - \lambda_{rest}) / \lambda_{rest}$) to calculate the line-of-sight velocity of the solar plasma. The resulting Doppler velocity, converted to km/s, provides the quantitative measure of the plasma motion. The plot visually compares the simulated data with the fitted Gaussian profile.

**7.6.2 Planetary: Equivalent Width Measurement of Methane Bands**
Spectra of gas giant planets like Jupiter or Saturn, or moons with atmospheres like Titan, often exhibit broad absorption bands due to molecules like methane (CH₄) or ammonia (NH₃). The equivalent width (EW) of these bands provides information about the abundance and vertical distribution of these gases in the planet's atmosphere. This example simulates measuring the equivalent width of a broad methane absorption band in a planetary spectrum. It assumes the spectrum is continuum-normalized and uses `specutils.analysis.equivalent_width` to calculate the EW by direct integration over the specified band region.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.analysis import equivalent_width
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Planetary EW example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def equivalent_width(spectrum, continuum=None, regions=None): return np.nan*u.AA # Dummy
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt

# --- Simulate Planetary Spectrum (Continuum Normalized) ---
if specutils_available:
    # Define wavelength axis (e.g., near-infrared)
    wavelengths = np.linspace(8000, 9500, 600) * u.AA
    # Simulate continuum = 1 + noise
    flux_normalized = np.random.normal(1.0, 0.015, size=wavelengths.shape)
    # Add a broad absorption band (e.g., methane)
    band_center = 8900 * u.AA
    band_width = 150 * u.AA # FWHM approx
    band_depth = 0.4 # 40% absorption at center
    # Use Gaussian shape for simplicity, real bands are more complex (multiple lines)
    flux_normalized *= (1.0 - band_depth * np.exp(-0.5 * ((wavelengths - band_center) / (band_width / 2.355))**2))
    # Ensure flux is not negative
    flux_normalized = np.maximum(flux_normalized, 0)

    # Create Spectrum1D object
    planetary_spec_norm = Spectrum1D(flux=flux_normalized, spectral_axis=wavelengths)
    print("Simulated normalized planetary spectrum with absorption band created.")

    # --- Define Spectral Region for EW Measurement ---
    # Define the region over which to integrate the absorption band
    # Should encompass the full band, avoid continuum regions if possible
    # Typically defined based on visual inspection or known band limits
    ew_region_start = band_center - 1.5 * band_width # Integrate over +/- 1.5 * FWHM approx
    ew_region_end = band_center + 1.5 * band_width
    ew_spectral_region = SpectralRegion(ew_region_start, ew_region_end)
    print(f"Defined EW integration region: {ew_spectral_region}")

    # --- Calculate Equivalent Width using specutils ---
    print("Calculating equivalent width...")
    # Use specutils.analysis.equivalent_width
    # Assumes spectrum is continuum-normalized (continuum=1)
    # Provide the spectral region over which to calculate the EW
    try:
        # equivalent_width implicitly assumes continuum=1 if not provided for normalized spectrum
        eq_width = equivalent_width(planetary_spec_norm, regions=ew_spectral_region)

        print("\nPlanetary Example: Equivalent Width Calculation Complete.")
        # EW result will have wavelength units (e.g., Angstrom)
        print(f"  Equivalent Width of the band: {eq_width:.2f}")
        # Note: EW of absorption is positive by convention in this calculation

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 5))
        plt.plot(planetary_spec_norm.spectral_axis, planetary_spec_norm.flux, label='Normalized Spectrum')
        # Shade the region used for EW calculation
        plt.axvspan(ew_region_start.value, ew_region_end.value, color='lightcoral', alpha=0.3, label='EW Integration Region')
        plt.axhline(1.0, color='grey', linestyle=':', label='Continuum Level')
        plt.xlabel(f"Wavelength ({planetary_spec_norm.spectral_axis.unit})")
        plt.ylabel("Normalized Flux")
        plt.title(f"Planetary Absorption Band (EW = {eq_width:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(np.min(flux_normalized)-0.05, 1.1)
        plt.show()

    except Exception as e:
        print(f"An error occurred during equivalent width calculation: {e}")
else:
    print("Skipping Planetary EW example: specutils unavailable.")

```

This Python script focuses on measuring the equivalent width (EW) of a broad molecular absorption band, characteristic of planetary atmosphere spectra, using `specutils`. It simulates a continuum-normalized spectrum featuring a prominent absorption band (mimicking, for instance, methane absorption). A `SpectralRegion` object is defined to specify the wavelength range encompassing the absorption band over which the EW calculation should be performed. The core measurement is achieved using the `specutils.analysis.equivalent_width` function. Since the input spectrum is already continuum-normalized (continuum level is 1.0), the function directly integrates the depth of the absorption ($1 - F_{norm}$) over the specified `SpectralRegion`. The resulting `eq_width` value quantitatively represents the total strength of the absorption band in wavelength units (e.g., Angstroms), providing a key metric for assessing the abundance of the absorbing molecule in the planet's atmosphere. The plot visualizes the normalized spectrum and highlights the integration region used for the EW calculation.

**7.6.3 Stellar: Gaussian Profile Fitting to Balmer Lines**
Hydrogen Balmer lines (H$\alpha$, H$\beta$, H$\gamma$, etc.) are prominent absorption features in the spectra of many stars (like A-type or F-type stars) and are sensitive diagnostics of stellar temperature, surface gravity, and atmospheric conditions. Accurately characterizing their profiles, often by fitting mathematical models, is a common task. This example demonstrates fitting a Gaussian profile to a simulated H$\beta$ absorption line in a stellar spectrum using `specutils.fitting.fit_lines` and `astropy.modeling`. It extracts key parameters from the fit, such as the line center, depth (amplitude), and width (sigma/FWHM).

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_lines, estimate_line_parameters
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Stellar Balmer line example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_lines(spectrum, model, window=None): return model # Dummy
    def estimate_line_parameters(spectrum, models=None): return models.Gaussian1D() # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models # For Gaussian1D model
import matplotlib.pyplot as plt

# --- Simulate Stellar Spectrum (Continuum Normalized) ---
if specutils_available:
    # Wavelength axis around H-beta (4861 A)
    hbeta_rest = 4861.33 * u.AA
    wavelengths = np.linspace(hbeta_rest.value - 50, hbeta_rest.value + 50, 300) * u.AA
    # Simulate normalized flux = 1 + noise - Absorption line
    flux_normalized = np.random.normal(1.0, 0.01, size=wavelengths.shape)
    # Add H-beta absorption profile (Gaussian for simplicity)
    line_depth = 0.7 # 70% depth
    line_sigma = 5.0 * u.AA # Line width (broad for Balmer)
    flux_normalized -= line_depth * np.exp(-0.5 * ((wavelengths - hbeta_rest) / line_sigma)**2)
    # Ensure non-negative
    flux_normalized = np.maximum(flux_normalized, 0)

    # Create Spectrum1D object
    star_spec_norm = Spectrum1D(flux=flux_normalized, spectral_axis=wavelengths)
    print("Simulated normalized stellar spectrum with H-beta line created.")

    # --- Fit Gaussian Profile to H-beta ---
    print("Fitting Gaussian profile to H-beta line...")
    # Define the model: Continuum (1.0) + Gaussian (negative amplitude for absorption)
    # Note: fit_lines fits the model directly to the flux data.
    # We need a model representing Continuum + Line.
    # Model = Const1D(amplitude=1.0, fixed={'amplitude': True}) + Gaussian1D(...)
    # Initial guesses: Amplitude (negative), mean (rest wavelength), stddev
    amp_guess = -line_depth # Negative for absorption relative to continuum=1
    mean_guess = hbeta_rest
    stddev_guess = line_sigma
    # Need to define the compound model to fit
    line_model_init = models.Gaussian1D(amplitude=amp_guess, mean=mean_guess, stddev=stddev_guess)
    # Fit only the line profile to continuum-subtracted data is often easier
    # Fit Gaussian1D to (1 - flux_normalized) as in Solar example

    # Let's try fitting the absorption depth profile: 1 - flux
    absorption_depth = (1.0 - star_spec_norm.flux.value) # Dimensionless depth
    depth_spectrum = Spectrum1D(flux=absorption_depth, spectral_axis=star_spec_norm.spectral_axis)

    # Use estimate_line_parameters for better guesses if needed (optional)
    # gauss_guess = estimate_line_parameters(depth_spectrum, models.Gaussian1D())
    # Use our manual guesses here:
    gauss_init = models.Gaussian1D(amplitude=line_depth, mean=mean_guess, stddev=stddev_guess) # Positive amplitude for depth

    # Define fitting window
    fit_window = SpectralRegion(hbeta_rest - 3*line_sigma, hbeta_rest + 3*line_sigma)

    try:
        # Fit the Gaussian model to the depth profile
        hbeta_fit_model = fit_lines(depth_spectrum, gauss_init, window=fit_window)

        print("Gaussian profile fitted to H-beta absorption.")
        print("Fitted Parameters:")
        print(f"  Amplitude (Depth): {hbeta_fit_model.amplitude.value:.3f}") # Should be positive depth
        print(f"  Mean (Center Wavelength): {hbeta_fit_model.mean.quantity:.3f}")
        print(f"  Standard Deviation (Width): {hbeta_fit_model.stddev.quantity:.3f}")

        # Calculate FWHM from fitted sigma
        fwhm = hbeta_fit_model.fwhm # astropy models have fwhm property
        print(f"  Derived FWHM: {fwhm:.3f}")

        # Calculate Equivalent Width from fitted parameters (Integral of depth profile)
        # EW = Amplitude * StdDev * sqrt(2*pi)
        ew_fit = hbeta_fit_model.amplitude.value * hbeta_fit_model.stddev.quantity * np.sqrt(2 * np.pi)
        print(f"  Derived Equivalent Width (from fit): {ew_fit:.3f}")

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 5))
        plt.plot(star_spec_norm.spectral_axis, star_spec_norm.flux, label='Normalized Spectrum', drawstyle='steps-mid')
        # Plot the fitted absorption profile (1 - fitted_gaussian_depth)
        plt.plot(star_spec_norm.spectral_axis, 1.0 - hbeta_fit_model(star_spec_norm.spectral_axis.value),
                 label=f'Gaussian Fit (FWHM={fwhm:.2f})', color='red', linestyle='--')
        plt.xlabel(f"Wavelength ({star_spec_norm.spectral_axis.unit})")
        plt.ylabel("Normalized Flux")
        plt.title(f"Stellar H-beta Line Profile Fit")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred during H-beta fitting: {e}")
else:
    print("Skipping Stellar Balmer line example: specutils unavailable.")

```

This Python script focuses on characterizing a stellar absorption line, specifically the H$\beta$ Balmer line, by fitting a Gaussian profile using `specutils` and `astropy.modeling`. It simulates a continuum-normalized stellar spectrum featuring a prominent H$\beta$ absorption line. To simplify fitting, it works with the absorption depth profile ($1 - F_{norm}$). An initial guess for the Gaussian parameters (amplitude representing depth, mean representing center wavelength, standard deviation representing width) is provided. The `specutils.fitting.fit_lines` function is employed, along with the initial `Gaussian1D` model and a `SpectralRegion` defining the wavelength window around H$\beta$, to perform the least-squares fit to the depth profile. The script extracts the best-fit parameters (amplitude/depth, mean/center, stddev/width) from the returned fitted model. It then demonstrates calculating derived properties like the Full Width at Half Maximum (FWHM) and the Equivalent Width (EW) directly from these fitted Gaussian parameters, providing a comprehensive quantitative characterization of the Balmer line profile. The plot visually confirms the quality of the fit by overlaying the model on the original normalized spectrum.

**7.6.4 Exoplanetary: Radial Velocity Measurement via Cross-Correlation**
Detecting exoplanets via the radial velocity (RV) method relies on measuring tiny, periodic Doppler shifts in the spectrum of the host star caused by the gravitational pull of the orbiting planet(s). The cross-correlation technique is a standard method for measuring these subtle shifts. It involves cross-correlating the observed stellar spectrum against a high-resolution template spectrum (either synthetic, or an observed spectrum of the same star or a similar star with known, stable velocity), typically after continuum removal and resampling to log-wavelength space. The position of the peak in the resulting cross-correlation function (CCF) indicates the velocity shift between the observed spectrum and the template. This example conceptually demonstrates measuring an RV shift using `specutils.analysis.correlation.crosscorrelate`.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectrumCollection
    from specutils.manipulation import LinearInterpolatedResampler, FluxConservingResampler
    from specutils.analysis.correlation import cross_correlate
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Exoplanetary RV example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectrumCollection: pass
    def cross_correlate(spec1, spec2): return (np.arange(10)-5)*u.km/u.s, np.exp(-0.5*((np.arange(10)-5)/2)**2) # Dummy CCF
    specutils_available = False # Set flag
import astropy.units as u
import astropy.constants as const
from astropy.modeling import models # For simulation
import matplotlib.pyplot as plt

# --- Simulate Observed and Template Stellar Spectra ---
if specutils_available:
    # Define high-resolution wavelength axis (log-linear often preferred for CCF)
    # Create linear axis first, then convert for simulation
    lambda_min, lambda_max, n_pix = 5000, 5100, 2000
    wavelengths_lin = np.linspace(lambda_min, lambda_max, n_pix) * u.AA
    # log_wavelengths = np.log(wavelengths_lin.value) # Example log-lambda grid

    # Simulate a simple template spectrum (e.g., continuum + absorption lines)
    template_flux = np.ones(n_pix)
    # Add absorption lines at specific rest wavelengths
    lines_rest = [5015.0, 5045.0, 5070.0, 5095.0] * u.AA
    line_depth = 0.4
    line_sigma = 0.1 * u.AA # Narrow lines for high-res
    for line_wav in lines_rest:
        template_flux -= line_depth * np.exp(-0.5 * ((wavelengths_lin - line_wav) / line_sigma)**2)
    template_spec = Spectrum1D(flux=template_flux, spectral_axis=wavelengths_lin)
    print("Simulated template spectrum created.")

    # Simulate observed spectrum with a Doppler shift (RV) + noise
    rv_shift = 25.5 * u.m / u.s # Realistic small RV shift in m/s
    observed_wavelengths = wavelengths_lin * (1 + rv_shift / const.c)
    # Observed flux needs to be evaluated at the *shifted* wavelengths corresponding to the template grid
    # Easier to shift the template and evaluate on original grid
    shifted_template_flux = np.ones(n_pix)
    for line_wav in lines_rest:
        shifted_line_wav = line_wav * (1 + rv_shift / const.c)
        shifted_template_flux -= line_depth * np.exp(-0.5 * ((wavelengths_lin - shifted_line_wav) / line_sigma)**2)
    # Add noise
    observed_flux = shifted_template_flux + np.random.normal(0, 0.02, size=n_pix)
    observed_spec = Spectrum1D(flux=observed_flux, spectral_axis=wavelengths_lin)
    print(f"Simulated observed spectrum with RV shift = {rv_shift:.1f} created.")

    # --- Perform Cross-Correlation using specutils ---
    # Often requires continuum subtraction/normalization and resampling to log-lambda first
    # For simplicity, assume spectra are suitably prepared (e.g., continuum normalized)
    # And use linear wavelength grid here (though log is better for RV precision)
    print("Performing cross-correlation...")

    try:
        # Use specutils.analysis.correlation.cross_correlate
        # It correlates spectrum1 against spectrum2
        # Requires spectra to be on the same spectral axis grid (or handles resampling)
        # Returns lag (velocity shift) and cross-correlation function (CCF) value
        # Ensure spectra are on same grid (already true in this simulation)
        # The function expects Spectrum1D objects
        lag_velocity, ccf_values = cross_correlate(observed_spec, template_spec)

        print("Cross-correlation complete.")

        # --- Find Peak of CCF to determine RV shift ---
        print("Finding peak of CCF...")
        # Find the index where the CCF value is maximum
        peak_index = np.argmax(ccf_values)
        # The velocity corresponding to this peak is the measured RV shift
        measured_rv = lag_velocity[peak_index]
        # Fit a parabola/Gaussian to CCF peak for sub-pixel precision (optional)
        # ... peak fitting code ...

        print("\nExoplanetary Example: RV Measurement via CCF Complete.")
        print(f"  Measured RV shift (peak of CCF): {measured_rv.to(u.m/u.s):.2f}")
        print(f"  (Input simulated RV shift was: {rv_shift:.2f})")

        # --- Optional: Plotting ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        # Plot spectra (offset for clarity)
        axes[0].plot(template_spec.spectral_axis, template_spec.flux, label='Template Spectrum')
        axes[0].plot(observed_spec.spectral_axis, observed_spec.flux + 0.1, label='Observed Spectrum (Shifted + Noise)')
        axes[0].set_xlabel(f"Wavelength ({template_spec.spectral_axis.unit})")
        axes[0].set_ylabel("Normalized Flux (offset)")
        axes[0].set_title("Template and Observed Spectra")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Plot CCF and peak
        axes[1].plot(lag_velocity.to(u.km/u.s), ccf_values, label='Cross-Correlation Function (CCF)')
        axes[1].axvline(measured_rv.to(u.km/u.s).value, color='red', linestyle='--', label=f'Measured RV = {measured_rv.to(u.m/u.s):.1f}')
        axes[1].set_xlabel(f"Velocity Lag ({u.km/u.s})")
        axes[1].set_ylabel("CCF Amplitude")
        axes[1].set_title("Cross-Correlation Function")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during cross-correlation or RV measurement: {e}")
else:
    print("Skipping Exoplanetary RV example: specutils unavailable.")

```

This Python script demonstrates the fundamental technique of measuring stellar radial velocity (RV) shifts, often used for exoplanet detection, via cross-correlation using `specutils`. It simulates a high-resolution template spectrum and an "observed" spectrum that includes a small, known RV shift plus noise. The core analysis step employs `specutils.analysis.correlation.cross_correlate`, which computes the cross-correlation function (CCF) between the observed spectrum and the template spectrum as a function of velocity lag. This function effectively slides the template spectrum across the observed spectrum in velocity space and measures their similarity at each lag. The script then identifies the velocity lag at which the CCF reaches its maximum value; this lag corresponds to the measured Doppler shift (RV) of the observed spectrum relative to the template. The result quantitatively measures the star's line-of-sight velocity shift, which, when monitored over time, can reveal the presence of orbiting planets. The plots visualize the input spectra and the resulting CCF with its peak indicating the measured RV.

**7.6.5 Galactic: Emission Line Flux and Velocity Measurement**
Studying ionized gas in Galactic nebulae (e.g., HII regions, planetary nebulae) often involves measuring the properties of prominent optical emission lines like H$\alpha$, H$\beta$, [O III], [N II], or [S II]. The flux of these lines relates to the amount of ionized gas and its physical conditions (temperature, density, ionization state), while their central wavelengths reveal the gas kinematics (bulk velocity, velocity dispersion/turbulence). This example simulates measuring the integrated flux and centroid velocity of an emission line (e.g., [O III] 5007 Å) in a spectrum extracted from such a nebula. It uses `specutils` functions to fit a Gaussian profile to the emission line, from which the total flux (related to the integral of the fit) and the precise center (yielding velocity) are derived.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_lines
    from specutils.analysis import line_flux # Can use for flux integration
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Galactic emission line example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_lines(spectrum, model, window=None): return model # Dummy
    def line_flux(spectrum, regions=None): return 0*u.erg/u.s/u.cm**2 # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models # For Gaussian1D model
import astropy.constants as const
import matplotlib.pyplot as plt

# --- Simulate Nebular Emission Line Spectrum ---
if specutils_available:
    # Wavelength axis around [O III] 5007 A
    oiii_rest = 5006.84 * u.AA
    nebula_velocity = -50.5 * u.km / u.s # Example bulk velocity
    # Calculate observed wavelength
    oiii_observed = oiii_rest * (1 + nebula_velocity / const.c)

    wavelengths = np.linspace(oiii_observed.value - 15, oiii_observed.value + 15, 250) * u.AA
    # Simulate flat continuum + emission line + noise
    # Assume flux units are calibrated (e.g., erg/s/cm2/A)
    flux_unit = u.erg / u.s / u.cm**2 / u.AA
    continuum_level = 1e-17 * flux_unit
    flux_data = np.random.normal(continuum_level.value, 0.1e-17, size=wavelengths.shape)
    # Add [O III] emission profile (Gaussian)
    line_amplitude = 15e-17 # Peak flux above continuum
    line_sigma = 1.5 * u.AA # Line width (velocity dispersion + instrumental)
    flux_data += line_amplitude * np.exp(-0.5 * ((wavelengths - oiii_observed) / line_sigma)**2)
    flux_data *= flux_unit # Apply units

    # Create Spectrum1D object
    nebula_spec = Spectrum1D(flux=flux_data, spectral_axis=wavelengths)
    print("Simulated nebular emission line spectrum created.")

    # --- Fit Gaussian Profile to the Emission Line ---
    print(f"Fitting Gaussian profile to emission line near {oiii_rest:.2f}...")
    # Define the model: Continuum (Flat/Polynomial) + Gaussian Emission Line
    # Fit continuum + line simultaneously
    amp_guess = line_amplitude
    mean_guess = oiii_observed
    stddev_guess = line_sigma
    cont_guess = continuum_level
    # Compound model
    fit_model_init = models.Const1D(amplitude=cont_guess.value) + \
                     models.Gaussian1D(amplitude=amp_guess.value, mean=mean_guess.value, stddev=stddev_guess.value) # Use values for init if model needs floats

    # Define fitting window
    fit_window = SpectralRegion(oiii_observed - 3*line_sigma, oiii_observed + 3*line_sigma)

    try:
        # Use fit_lines to fit the compound model
        fitted_compound_model = fit_lines(nebula_spec, fit_model_init, window=fit_window)

        # Extract the fitted Gaussian component for line properties
        # fitted_compound_model might be the compound model itself
        # Access submodels by index or name if defined
        if isinstance(fitted_compound_model, models.CompoundModel) and len(fitted_compound_model) == 2:
             fitted_gaussian = fitted_compound_model[1] # Assume Gaussian is second submodel
             fitted_continuum = fitted_compound_model[0].amplitude.quantity
        else: # Assume fit_lines returned just the Gaussian if continuum was fixed/subtracted
             fitted_gaussian = fitted_compound_model
             fitted_continuum = cont_guess # Use initial guess if only line fitted

        print("Gaussian profile fitted to emission line.")
        print("Fitted Parameters (Gaussian Component):")
        # Add units back to fitted parameters if needed
        fit_amp = fitted_gaussian.amplitude.value * flux_unit
        fit_mean = fitted_gaussian.mean.value * u.AA
        fit_stddev = fitted_gaussian.stddev.value * u.AA

        print(f"  Amplitude (Peak Flux above continuum): {fit_amp:.2E}")
        print(f"  Mean (Observed Wavelength): {fit_mean:.3f}")
        print(f"  Standard Deviation (Width): {fit_stddev:.3f}")

        # --- Calculate Integrated Line Flux ---
        # Flux = Amplitude * StdDev * sqrt(2*pi) -- Flux integrated over wavelength
        integrated_flux = (fit_amp * fit_stddev * np.sqrt(2 * np.pi))
        # Units should be Flux Density * Wavelength = Flux
        integrated_flux = integrated_flux.to(u.erg / u.s / u.cm**2) # Convert to common flux units

        print(f"\nCalculated Integrated Line Flux: {integrated_flux:.3E}")

        # --- Calculate Centroid Velocity ---
        lambda_center_fitted = fit_mean
        velocity = const.c * (lambda_center_fitted - oiii_rest) / oiii_rest
        velocity_kms = velocity.to(u.km / u.s)

        print(f"\nCalculated Centroid Velocity:")
        print(f"  Rest Wavelength: {oiii_rest:.3f}")
        print(f"  Fitted Center: {lambda_center_fitted:.3f}")
        print(f"  Velocity: {velocity_kms:.2f}")
        print(f"  (Input simulated velocity was: {nebula_velocity:.2f})")

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 5))
        plt.plot(nebula_spec.spectral_axis, nebula_spec.flux, label='Nebula Spectrum', drawstyle='steps-mid')
        # Plot the full fitted model (Continuum + Gaussian)
        plt.plot(nebula_spec.spectral_axis, fitted_compound_model(nebula_spec.spectral_axis.value)*flux_unit,
                 label=f'Fit (Velocity={velocity_kms:.1f})', color='red', linestyle='--')
        plt.xlabel(f"Wavelength ({nebula_spec.spectral_axis.unit})")
        plt.ylabel(f"Flux ({nebula_spec.flux.unit})")
        plt.title(f"Galactic Emission Line Fit ([O III] 5007)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An error occurred during emission line fitting: {e}")
else:
    print("Skipping Galactic emission line example: specutils unavailable.")

```

This Python script demonstrates the analysis of a prominent emission line, such as [O III] 5007 Å, commonly observed in Galactic nebulae. It simulates a spectrum containing the emission line superimposed on a relatively flat continuum with noise, assuming the spectrum is already flux-calibrated. The core analysis involves fitting a model consisting of a constant continuum plus a Gaussian profile (`Const1D + Gaussian1D`) to the emission line using `specutils.fitting.fit_lines`. This simultaneous fit determines the continuum level, the line's peak amplitude above the continuum, its precise central wavelength ($\lambda_{center\_fitted}$), and its width ($\sigma$). From these fitted parameters, the script calculates two key physical quantities: the integrated line flux (obtained by integrating the fitted Gaussian profile, proportional to Amplitude $\times$ Width) which relates to the emission measure of the gas, and the line-of-sight velocity of the nebula derived from the Doppler shift between the fitted center $\lambda_{center\_fitted}$ and the line's rest wavelength $\lambda_{rest}$. The plot shows the original spectrum and the combined continuum+Gaussian fit, illustrating how the model captures the line properties.

**7.6.6 Extragalactic: Redshift Determination via Emission Line Fitting**
For many galaxies, especially star-forming galaxies or those hosting active galactic nuclei (AGN), the optical spectrum exhibits strong, narrow emission lines (e.g., H$\alpha$, [O III], [N II], [S II]). Identifying these lines and accurately measuring their observed wavelengths provides the most precise method for determining the galaxy's redshift, $z = (\lambda_{obs} - \lambda_{rest}) / \lambda_{rest}$. This example simulates fitting Gaussian profiles to multiple identified emission lines in a galaxy spectrum to determine a robust redshift. It involves fitting each line individually (or potentially simultaneously) and calculating the redshift implied by each line's measured center, then combining these estimates.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_lines
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Extragalactic redshift example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_lines(spectrum, model, window=None): return model # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models, fitting # Need fitter object explicitly sometimes
import astropy.constants as const
import matplotlib.pyplot as plt

# --- Simulate Galaxy Spectrum at a known Redshift ---
if specutils_available:
    true_redshift = 0.15 # Example galaxy redshift
    # Define rest wavelengths of common emission lines
    lines_rest = {
        'Hbeta': 4861.33 * u.AA,
        '[OIII]5007': 5006.84 * u.AA,
        'Halpha': 6562.80 * u.AA,
        '[NII]6583': 6583.45 * u.AA
    }
    # Calculate observed wavelengths
    lines_observed = {name: wav * (1 + true_redshift) for name, wav in lines_rest.items()}

    # Define wavelength axis covering these lines
    wavelengths = np.linspace(lines_observed['Hbeta'].value - 100, lines_observed['[NII]6583'].value + 100, 800) * u.AA
    # Simulate galaxy continuum (e.g., relatively flat + noise)
    flux_unit = u.Unit("1e-17 erg / (s cm2 Angstrom)") # Example flux density units
    continuum_level = 2.0
    flux_data = np.random.normal(continuum_level, 0.2, size=wavelengths.shape)
    # Add emission lines (Gaussians)
    line_amplitudes = {'Hbeta': 5.0, '[OIII]5007': 15.0, 'Halpha': 20.0, '[NII]6583': 8.0} # Relative strengths
    line_sigma = 3.0 * u.AA # Assume same width for simplicity (instrumental broadening dominates)
    for name, obs_wav in lines_observed.items():
        flux_data += line_amplitudes[name] * np.exp(-0.5 * ((wavelengths - obs_wav) / line_sigma)**2)
    flux_data *= flux_unit

    # Create Spectrum1D object
    galaxy_spec = Spectrum1D(flux=flux_data, spectral_axis=wavelengths)
    print(f"Simulated galaxy spectrum created at z={true_redshift:.3f}.")

    # --- Fit Emission Lines to Determine Redshift ---
    print("Fitting emission lines to determine redshift...")
    measured_redshifts = []
    fitted_lines = {}

    # Fit each identified line individually
    # (Alternatively, could do a combined fit constraining redshift to be the same)
    for name, rest_wav in lines_rest.items():
        print(f"\nFitting line: {name} (Rest: {rest_wav:.2f})")
        obs_wav_expected = lines_observed[name] # Use calculated observed wavelength as initial guess
        # Define fitting window around the expected observed wavelength
        fit_window = SpectralRegion(obs_wav_expected - 5*line_sigma, obs_wav_expected + 5*line_sigma)
        # Define model: Flat continuum + Gaussian
        # Estimate initial parameters from data within window (or use guesses)
        spec_in_window = galaxy_spec.spectral_region_extract(fit_window)
        # Ensure cont_guess calculation avoids potential masked NaNs if mask exists
        cont_guess_val = np.nanmedian(spec_in_window.flux.value)
        if np.isnan(cont_guess_val): cont_guess_val = continuum_level # Fallback
        cont_guess = cont_guess_val * flux_unit
        amp_guess_val = np.nanmax(spec_in_window.flux.value) - cont_guess_val # Peak above median
        if np.isnan(amp_guess_val) or amp_guess_val <= 0: amp_guess_val = np.std(spec_in_window.flux.value)*3 # Fallback guess
        amp_guess = amp_guess_val * flux_unit

        stddev_guess = line_sigma
        mean_guess = obs_wav_expected

        # Initial compound model
        fit_model_init = models.Const1D(amplitude=cont_guess.value) + \
                         models.Gaussian1D(amplitude=amp_guess.value, mean=mean_guess.value, stddev=stddev_guess.value) # Use values for init

        # Perform the fit using a robust fitter
        try:
            fitter = fitting.LevMarLSQFitter() # Or TRFLSQFitter

            # Use fitter directly as fit_lines might have issues with compound models or units
            # Provide weights if uncertainty is known
            fitted_model_manual = fitter(fit_model_init, galaxy_spec.spectral_axis.value, galaxy_spec.flux.value,
                                         weights = 1.0 / (0.2e-17**2) if 'ivar' not in galaxy_spec.meta else 1.0/galaxy_spec.uncertainty.array**2, # Example weights
                                         maxiter=200) # Increase max iterations if needed


            # Extract Gaussian parameters from the fitted compound model
            if isinstance(fitted_model_manual, models.CompoundModel) and len(fitted_model_manual)==2:
                 fitted_gaussian = fitted_model_manual[1]
            else: # Handle case if only Gaussian was fitted somehow
                 fitted_gaussian = fitted_model_manual

            lambda_center_fitted = fitted_gaussian.mean.value * u.AA # Assign units

            # Check if fit converged and parameters are reasonable (e.g., positive width)
            if fitter.fit_info['ierr'] not in [1, 2, 3, 4] or fitted_gaussian.stddev.value <= 0:
                 print(f"  Fit did not converge well for line {name}. Skipping.")
                 continue

            fitted_lines[name] = lambda_center_fitted # Store fitted wavelength

            # Calculate redshift for this line
            z_line = (lambda_center_fitted - rest_wav) / rest_wav
            measured_redshifts.append(z_line.value) # Store dimensionless redshift value
            print(f"  Fitted Center: {lambda_center_fitted:.3f}")
            print(f"  Implied Redshift (z): {z_line:.5f}")

        except Exception as fit_err:
            print(f"  Fit failed for line {name}: {fit_err}")


    # --- Combine Redshifts for Final Estimate ---
    if measured_redshifts:
        # Calculate robust mean/median redshift from measurements
        final_redshift = np.nanmedian(measured_redshifts) # Use median for robustness
        redshift_stddev = np.nanstd(measured_redshifts)
        print(f"\nCombined Redshift Estimate:")
        print(f"  Median z = {final_redshift:.5f}")
        print(f"  StdDev z = {redshift_stddev:.5f}")
        print(f"  (Input true redshift was: {true_redshift:.5f})")
    else:
        print("\nNo lines successfully fitted. Cannot determine redshift.")
        final_redshift = np.nan

    # --- Optional: Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(galaxy_spec.spectral_axis, galaxy_spec.flux, label='Galaxy Spectrum', drawstyle='steps-mid', alpha=0.7)
    # Mark fitted line centers
    for name, fitted_wav in fitted_lines.items():
        plt.axvline(fitted_wav.value, color='red', linestyle='--', alpha=0.8, label=f'{name} Fit={fitted_wav:.2f}')
    plt.xlabel(f"Observed Wavelength ({galaxy_spec.spectral_axis.unit})")
    plt.ylabel(f"Flux Density ({galaxy_spec.flux.unit})")
    plt.title(f"Galaxy Spectrum - Redshift Measurement (z={final_redshift:.4f})")
    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, alpha=0.3)
    plt.show()

else:
    print("Skipping Extragalactic redshift example: specutils unavailable.")

```

This Python script demonstrates how to determine the redshift of a galaxy by identifying and fitting multiple prominent emission lines in its spectrum. It simulates a galaxy spectrum at a known redshift, including several common emission lines (H$\beta$, [O III] 5007, H$\alpha$, [N II] 6583) shifted to their observed wavelengths. The core analysis loop iterates through each expected emission line. For each line, it defines a spectral window around its anticipated observed wavelength and fits a model (here, a Gaussian plus a constant continuum) to the spectral data within that window using `astropy.modeling` and a least-squares fitter (like `LevMarLSQFitter`). From the fitted Gaussian component, the precise observed central wavelength ($\lambda_{obs}$) is extracted. This observed wavelength is then compared to the known rest wavelength ($\lambda_{rest}$) of the corresponding transition to calculate the redshift $z = (\lambda_{obs} - \lambda_{rest}) / \lambda_{rest}$ implied by that specific line. After fitting multiple lines, the script calculates a robust final redshift estimate (e.g., the median) from the individual line measurements, providing a reliable determination of the galaxy's redshift. The plot shows the spectrum with vertical lines indicating the fitted centers of the emission lines used for the redshift calculation.

**7.6.7 Cosmology: Lyman-alpha Absorption Line Property Measurement**
Spectra of distant quasars (QSOs) often show a "forest" of absorption lines blueward of the quasar's Lyman-alpha (Ly$\alpha$) emission line (rest wavelength $\lambda_{rest} \approx 1215.67$ Å). These absorption lines arise from neutral hydrogen gas in intervening clouds and galaxies along the line of sight at various redshifts lower than the quasar's. Studying the properties of these Ly$\alpha$ forest absorbers (their redshift distribution, column densities, line widths) provides crucial information about the intergalactic medium (IGM) and the large-scale structure of the universe. This example simulates measuring the equivalent width and fitting the profile of a single Ly$\alpha$ absorption line within a simulated QSO spectrum segment, assuming the quasar continuum has been estimated and normalized.

```python
import numpy as np
# Requires specutils and astropy: pip install specutils astropy
try:
    from specutils import Spectrum1D, SpectralRegion
    from specutils.fitting import fit_lines
    from specutils.analysis import equivalent_width
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Cosmology Ly-alpha example.")
    # Define dummy classes if needed
    class Spectrum1D: pass
    class SpectralRegion: pass
    def fit_lines(spectrum, model, window=None): return model # Dummy
    def equivalent_width(spectrum, continuum=None, regions=None): return 0*u.AA # Dummy
    specutils_available = False # Set flag
import astropy.units as u
from astropy.modeling import models # For Voigt1D model
import matplotlib.pyplot as plt

# --- Simulate QSO Spectrum Segment (Continuum Normalized) ---
if specutils_available:
    # Assume observing blueward of QSO Ly-alpha emission
    # Wavelength axis in Angstrom
    wavelengths = np.linspace(4000, 4100, 400) * u.AA
    # Simulate normalized continuum = 1 + noise
    flux_normalized = np.random.normal(1.0, 0.03, size=wavelengths.shape)
    # Add a Lyman-alpha absorption line from an intervening cloud
    # Line center corresponds to Ly-alpha at the cloud's redshift z_abs
    lya_rest = 1215.67 * u.AA
    z_abs = 2.3 # Example absorber redshift
    line_center_obs = lya_rest * (1 + z_abs) # Expected observed wavelength
    # Ly-alpha lines often have Voigt profiles (Gaussian core + Lorentzian damping wings)
    # Simulate with a Voigt profile
    line_amplitude_voigt = 0.8 # Depth
    line_sigma_gauss = 0.5 * u.AA # Gaussian component width (thermal/turbulent/instrumental)
    line_gamma_lorentz = 0.2 * u.AA # Lorentzian component width (damping parameter)
    # Astropy Voigt1D uses FWHM parameters
    fwhm_g_voigt = line_sigma_gauss * 2 * np.sqrt(2 * np.log(2))
    fwhm_l_voigt = line_gamma_lorentz * 2 # FWHM = 2*gamma for Lorentzian

    # Create the Voigt model for simulation (need to subtract from 1)
    voigt_sim_model = models.Voigt1D(amplitude_L=line_amplitude_voigt, x_0=line_center_obs,
                                     fwhm_L=fwhm_l_voigt, fwhm_G=fwhm_g_voigt)
    # Evaluate model, normalize peak to line_amplitude_voigt approx depth (tricky)
    # Evaluate profile and scale to desired depth
    profile_shape = voigt_sim_model(wavelengths)
    if np.max(profile_shape) > 0: # Avoid division by zero if flat
         profile_shape_norm = profile_shape / np.max(profile_shape) # Normalize peak to 1
         flux_normalized -= line_amplitude_voigt * profile_shape_norm # Subtract scaled profile
    flux_normalized = np.maximum(flux_normalized, 0) # Ensure non-negative

    # Create Spectrum1D object
    qso_spec_norm = Spectrum1D(flux=flux_normalized, spectral_axis=wavelengths)
    print(f"Simulated normalized QSO spectrum segment with Ly-alpha absorber at z={z_abs:.2f} created.")

    # --- Measure Equivalent Width ---
    print("\nCalculating Equivalent Width...")
    # Define spectral region around the absorption line
    # Width should encompass significant part of profile (e.g., +/- 10 sigma_G approx)
    ew_width_factor = 10
    ew_region = SpectralRegion(line_center_obs - ew_width_factor * line_sigma_gauss,
                               line_center_obs + ew_width_factor * line_sigma_gauss)
    try:
        # Use equivalent_width on the normalized spectrum
        eq_width = equivalent_width(qso_spec_norm, regions=ew_region) # Assumes continuum=1
        print(f"  Equivalent Width (observed frame): {eq_width:.3f}")
        # Rest frame EW is often the physically relevant quantity
        ew_rest = eq_width / (1 + z_abs) # Approximate correction
        print(f"  Approx. Rest Frame EW: {ew_rest:.3f}")

    except Exception as ew_err:
        print(f"  Could not calculate EW: {ew_err}")
        eq_width = np.nan * u.AA

    # --- Fit Voigt Profile (Conceptual - Requires good initial guesses) ---
    print("\nFitting Voigt Profile (Conceptual)...")
    # Define Voigt model - initial guesses are critical for Voigt fits
    # Fit the depth profile (1 - flux_normalized)
    depth_profile = (1.0 - qso_spec_norm.flux.value) # Dimensionless
    depth_spec = Spectrum1D(flux=depth_profile, spectral_axis=qso_spec_norm.spectral_axis)
    # Guess parameters (amplitude might need adjustment from depth)
    # Use simulated values as guesses here for demonstration
    amp_l_guess = line_amplitude_voigt # Initial guess for amplitude_L
    x_0_guess = line_center_obs
    fwhm_l_guess = fwhm_l_voigt
    fwhm_g_guess = fwhm_g_voigt

    voigt_init = models.Voigt1D(amplitude_L=amp_l_guess, x_0=x_0_guess,
                                fwhm_L=fwhm_l_guess, fwhm_G=fwhm_g_guess)
    # Define fitting window
    fit_window_voigt = SpectralRegion(line_center_obs - ew_width_factor * line_sigma_gauss, # Use wider window for Voigt
                                      line_center_obs + ew_width_factor * line_sigma_gauss)

    try:
        # Perform the fit (often needs robust fitter and bounds)
        # fitter = fitting.LevMarLSQFitter()
        # voigt_fit_model = fitter(voigt_init, wavelengths.value, depth_profile) # Manual fit example
        # Or use fit_lines if it supports Voigt well
        # Note: Fitting Voigt profiles robustly can be challenging.
        # Specialized codes (e.g., VPFIT) are often used for Ly-alpha forest analysis.
        print("  (Skipping actual Voigt fit execution - complex and requires careful setup)")
        # Conceptual: Extract parameters if fit succeeds
        # fitted_center = voigt_fit_model.x_0.value
        # fitted_fwhm_G = voigt_fit_model.fwhm_G.value
        # fitted_fwhm_L = voigt_fit_model.fwhm_L.value
        # Column density (N_HI) and Doppler parameter (b) can be derived from Voigt parameters.

    except Exception as fit_err:
        print(f"  Conceptual Voigt fit encountered potential error: {fit_err}")


    # --- Optional: Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(qso_spec_norm.spectral_axis, qso_spec_norm.flux, label='Normalized QSO Spectrum', drawstyle='steps-mid')
    # Mark the absorption line region
    plt.axvspan(ew_region.lower.value, ew_region.upper.value, color='lightgray', alpha=0.4, label='Ly-alpha Absorber Region')
    plt.axhline(1.0, color='grey', linestyle=':')
    plt.xlabel(f"Observed Wavelength ({qso_spec_norm.spectral_axis.unit})")
    plt.ylabel("Normalized Flux")
    plt.title(f"Simulated Ly-alpha Absorption Line (z={z_abs:.2f}, EW={eq_width:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(np.min(flux_normalized)-0.1, 1.1)
    plt.show()

else:
    print("Skipping Cosmology Ly-alpha example: specutils unavailable.")

```

This final script addresses the analysis of Lyman-alpha (Ly$\alpha$) forest absorption lines observed in quasar spectra, which are crucial probes of the intergalactic medium (IGM) for cosmology. It simulates a segment of a continuum-normalized QSO spectrum containing a single Ly$\alpha$ absorption line originating from an intervening gas cloud at a specific redshift ($z_{abs}$). The script first demonstrates measuring the equivalent width (EW) of the absorption line using `specutils.analysis.equivalent_width` applied to the normalized spectrum within a defined spectral region encompassing the line. The rest-frame EW, often used to estimate the neutral hydrogen column density ($N_{HI}$), is approximated by dividing the observed EW by $(1+z_{abs})$. Secondly, the script conceptually outlines the process of fitting a Voigt profile (`astropy.modeling.models.Voigt1D`) to the absorption line. While the actual fitting is noted as complex and often requires specialized codes, the example sets up the model with initial parameter guesses (center wavelength, Gaussian width related to temperature/turbulence/instrument, Lorentzian width related to damping wings/column density). A successful Voigt fit would yield detailed parameters characterizing the absorber, providing deeper physical insights into the IGM cloud properties than EW alone. The plot visualizes the simulated spectrum and the region of the absorption line analyzed.

---

**References**

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project. The `specutils` package and `astropy.modeling` are the core libraries providing the data structures (`Spectrum1D`), analysis functions (`equivalent_width`, `centroid`, etc.), and model fitting capabilities central to this chapter (Sections 7.1, 7.3, 7.5, 7.6).

Bailey, S., Abareshi, B., Abidi, A., Abolfathi, B., Aerts, J., Aguilera-Gomez, C., Ahlen, S., Alam, S., Alexander, D. M., Alfarsy, R., Allen, L., Prieto, C. A., Alves-Oliveira, N., Anand, A., Armengaud, E., Ata, M., Avilés, A., Avon, M., Brooks, D., … Zou, H. (2023). The Data Release 1 of the Dark Energy Spectroscopic Instrument. *The Astrophysical Journal, 960*(1), 75. https://doi.org/10.3847/1538-4357/acff2f
*   *Summary:* Describes the first major data release and the reduction/analysis pipeline (including `redrock`) for the DESI survey. This exemplifies the application of automated redshift determination techniques (Section 7.5), particularly cross-correlation, on massive datasets.

Comparat, J., Dwelly, T., Plesha, A., Yeche, C., Georgakakis, A., Mountrichas, G., Newman, J. A., Cooper, M., Fan, X., Fotopoulou, S., Hsu, C.-h., Nandra, K., Salim, S., Salvato, M., Seth, A., Schulze, A., Stern, D., Wolf, J., & Davis, T. M. (2023). The extended Baryon Oscillation Spectroscopic Survey: Overview and measurements of the 140k quasars from the Sloan Digital Sky Survey IV Data Release 16. *Monthly Notices of the Royal Astronomical Society, 519*(1), 485–501. https://doi.org/10.1093/mnras/stac3465
*   *Summary:* Presents analysis of a large quasar sample, relying heavily on accurate redshifts (Section 7.5) and analysis of spectral features like broad emission lines (Sections 7.2, 7.3) measured from survey spectra.

Crawford, S. M., Earl, N., Lim, P. L., Deil, C., Tollerud, E. J., Morris, B. M., Bray, E., Conseil, S., Donath, A., Fowler, J., Ginsburg, A., Kulumani, S., Pascual, S., Perren, G., Sipőcz, B., Weaver, B. A., Williams, R., Teuben, P., & Astropy Collaboration. (2023). specutils: A Python package for spectral analysis. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.8040075
*   *Summary:* This Zenodo record archives a version of `specutils`. It provides the core data structures and functions (`Spectrum1D`, `fit_lines`, `equivalent_width`, `centroid`, `cross_correlate`, etc.) used throughout this chapter for spectral manipulation and analysis (Sections 7.1, 7.3, 7.5, 7.6). Essential software reference.

García-Benito, R., & Jurado, E. (2023). Optimization of background subtraction in 1D astronomical spectra using orthogonal polynomials. *Astronomy and Computing, 44*, 100735. https://doi.org/10.1016/j.ascom.2023.100735
*   *Summary:* Focuses on using orthogonal polynomials for background/continuum fitting in spectra. This is directly relevant to the techniques discussed for continuum fitting and normalization in Section 7.1.

Hinkle, K. H., Stauffer, J. R., Plavchan, P. P., & Wallace, L. (2021). Infrared Astronomical Spectroscopy with High Spectral Resolution. *Publications of the Astronomical Society of the Pacific, 133*(1027), 092001. https://doi.org/10.1088/1538-3873/ac1a3a
*   *Summary:* Reviews high-resolution infrared spectroscopy. Analysis in this regime often requires careful line identification (Section 7.2) and precise profile fitting (Section 7.3.4) to measure features affected by telluric lines or molecular blends.

Ji, X., Frebel, A., Chiti, A., Simon, J. D., Jerkstrand, A., Lin, D., Thompson, I. B., Aguilera-Gómez, C., Casey, A. R., Gomez, F. A., Han, J., Ji, A. P., Kim, D., Marengo, M., McConnachie, A. W., Stringfellow, G. S., & Yoon, J. (2023). Chemical abundances of the stars in the Tucana II ultra-faint dwarf galaxy. *The Astronomical Journal, 165*(1), 26. https://doi.org/10.3847/1538-3881/aca4a5
*   *Summary:* Performs stellar abundance analysis using high-resolution spectra. This science application heavily relies on the accurate measurement of equivalent widths (Section 7.3.2) or detailed line synthesis/fitting (related to Section 7.3.4) of stellar absorption lines.

Prochaska, J. X., Hennawi, J. F., Westfall, K. B., Cooke, R., Wang, F., Hsyu, T., & Emg, D. (2020). PypeIt: The Python Spectroscopic Data Reduction Pipeline. *Journal of Open Source Software, 5*(54), 2308. https://doi.org/10.21105/joss.02308
*   *Summary:* Introduces the PypeIt pipeline. Beyond reduction, it includes modules for basic spectral analysis tasks like line detection and fitting (Sections 7.2, 7.3), illustrating the integration of analysis steps within modern pipelines.

Reetz, K. (2023). Determining the continuum for the identification of stellar absorption lines in digitised objective prism spectra. *Astronomische Nachrichten, 344*(1-3), e20230013. https://doi.org/10.1002/asna.20230013
*   *Summary:* Specifically addresses continuum determination challenges. This highlights the practical importance and difficulty of accurate continuum fitting (Section 7.1), which is a necessary prerequisite for most line measurement techniques (Section 7.3).

Ryabchikova, T., Piskunov, N., Kurucz, R. L., Stempels, H. C., Heiter, U., Pakhomov, Y., & Barklem, P. S. (2015). A major upgrade of the VALD database. *Physica Scripta, 90*(5), 054005. https://doi.org/10.1088/0031-8949/90/5/054005 *(Note: Pre-2020, but describes essential line database)*
*   *Summary:* Describes a major update to the Vienna Atomic Line Database (VALD), a critical resource for atomic transition data (rest wavelengths, oscillator strengths, etc.) used extensively for line identification (Section 7.2) and theoretical modeling in stellar spectroscopy. Foundational reference.

Sandford, N. R., Maseda, M. V., Chevallard, J., Tacchella, S., Arribas, S., Charlton, J. C., Curtis-Lake, E., Egami, E., Endsley, R., Hainline, K., Johnson, B. D., Robertson, B. E., Shivaei, I., Stacey, H., Stark, D. P., Williams, C. C., Boyett, K. N. K., Bunker, A. J., Charlot, S., … Willott, C. J. (2024). Emission line ratios from the JWST NIRSpec G395H spectrum of GN-z11. *arXiv preprint arXiv:2401.16955*. https://doi.org/10.48550/arXiv.2401.16955
*   *Summary:* Analyzes emission lines from a high-redshift galaxy with JWST. This study relies on accurate measurement of line fluxes, potentially through profile fitting (Section 7.3.4), to derive line ratios which are key physical diagnostics, illustrating cutting-edge spectral analysis.

