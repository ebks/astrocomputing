---

# Chapter 4

# Basic Spectroscopic Reduction: Extraction and Calibration

---
![imagem](imagem.png)

*This chapter focuses on the fundamental computational procedures required to transform raw, two-dimensional data acquired by astronomical spectrographs into calibrated, one-dimensional spectra suitable for scientific analysis. It begins by introducing the operational principles and data characteristics of common spectrograph designs, including long-slit, echelle, multi-object, and integral field unit instruments, highlighting how each configuration disperses light and encodes spectral information onto a detector array. Subsequently, the standard methods for representing spectroscopic data within computational frameworks are discussed, emphasizing the utility of specialized Python libraries like `specutils` for handling flux, spectral coordinates, uncertainties, and metadata in a structured manner. The critical process of accurately tracing the path of the dispersed spectrum (or multiple spectra/orders) across the two-dimensional detector is then detailed, covering algorithms for identifying the spectral trace and modeling its geometry. Following tracing, the chapter elaborates on methods for spectral extraction, explaining techniques ranging from simple summation to statistically optimal algorithms designed to maximize the signal-to-noise ratio by weighting pixel contributions based on the spatial profile and noise characteristics. The crucial step of wavelength calibration is thoroughly examined, encompassing the identification of known spectral features in calibration sources (e.g., arc lamps, sky lines) and the mathematical fitting of a dispersion solution to map detector pixels to physical wavelengths. Furthermore, the necessity and methods for spectroscopic flat-fielding are presented, addressing the correction for wavelength-dependent variations in instrument throughput and detector sensitivity. Finally, the chapter addresses common techniques for subtracting the contaminating background sky emission spectrum, a particularly important step for ground-based observations. Illustrative Python examples utilizing relevant libraries conclude the chapter, demonstrating practical application of these core reduction steps across diverse astronomical scenarios.*

---

**4.1 Fundamentals of Spectrographs**

Astronomical spectrographs are instruments designed to disperse the light collected by a telescope according to its wavelength, allowing detailed analysis of the spectral energy distribution of celestial objects. This spectral information is encoded in features like emission lines, absorption lines, and the shape of the continuum, revealing crucial information about the object's chemical composition, temperature, density, velocity, magnetic fields, and evolutionary state (Kitchin, 2013; Ayres, 2022). While diverse in design, spectrographs fundamentally rely on a dispersive element – typically a diffraction grating or prism – to separate incoming light into its constituent wavelengths, which are then imaged onto a two-dimensional detector (usually a CCD or similar array detector, see Chapter 2). The raw data product from most spectrographs is therefore a 2D image where one axis represents spatial position along the entrance aperture (slit or fiber) and the other axis represents dispersion (wavelength).

Several common spectrograph configurations exist, each suited for different scientific applications:
*   **Long-slit Spectrographs:** These instruments employ a narrow rectangular slit as the entrance aperture. Light entering the slit is collimated, dispersed by a grating, and then re-imaged onto the detector. The resulting 2D frame captures spatial information along the length of the slit in one dimension (e.g., the Y-axis) and spectral information (dispersion) along the other dimension (e.g., the X-axis). This allows simultaneous measurement of spectra from different spatial locations along the slit as it is placed across an extended object (e.g., a galaxy, nebula) or multiple objects falling along the slit. The spectral resolution ($R = \lambda / \Delta\lambda$, where $\Delta\lambda$ is the smallest resolvable wavelength interval) depends on the slit width, the grating properties (groove density), and the collimator/camera focal lengths. A narrower slit yields higher spectral resolution but lower throughput (less light enters).
*   **Echelle Spectrographs:** Designed to achieve very high spectral resolution ($R > 20,000 - 100,000+$), echelle spectrographs use a coarsely ruled diffraction grating (an echelle grating) operated at a high angle of incidence. This grating produces multiple, overlapping high-dispersion spectral orders. A second dispersive element (a cross-disperser, typically a prism or a low-dispersion grating) oriented perpendicular to the main dispersion direction is used to separate these overlapping orders spatially on the 2D detector. The raw data frame displays a characteristic pattern of curved or tilted spectral orders stacked vertically, with wavelength increasing along each order and order number changing vertically (e.g., Hinkle et al., 2021). This allows coverage of a wide wavelength range at high resolution simultaneously, ideal for detailed stellar spectroscopy, exoplanet radial velocity measurements, or quasar absorption line studies. Reduction requires carefully tracing and extracting each individual order.
*   **Multi-Object Spectrographs (MOS):** To increase observational efficiency for surveys or studies of stellar/galaxy populations, MOS instruments allow simultaneous acquisition of spectra from many (~10s to 1000s) individual objects within the telescope's field of view. This is typically achieved using either:
    *   *Slit Masks:* Custom-milled plates or laser-cut masks placed at the focal plane with small slitlets positioned precisely at the locations of target objects. Light from each slitlet is dispersed and imaged onto different regions of the detector.
    *   *Fiber Positioners:* Robotic systems that precisely position the ends of optical fibers onto target locations in the focal plane. Light from each fiber is then fed to one or more fixed spectrograph units. Examples include SDSS spectrographs, DESI, 4MOST, WEAVE (e.g., de Jong et al., 2019; Dalton et al., 2022).
    The raw 2D data frame contains multiple individual spectra (potentially hundreds) arranged across the detector, often densely packed. Data reduction involves identifying, tracing, and extracting each individual spectrum, requiring sophisticated pipeline processing. Dedicated fibers are often positioned on blank sky regions to facilitate sky subtraction.
*   **Integral Field Unit (IFU) Spectrographs:** IFUs enable **integral field spectroscopy (IFS)**, obtaining a spectrum for *every* spatial position within a contiguous two-dimensional field of view simultaneously. This produces a 3D data cube with two spatial dimensions (X, Y) and one spectral dimension ($\lambda$) (Allington-Smith & Content, 2002; Haynes et al., 2023). Different technologies are used to reformat the 2D field into one or more long pseudo-slits at the entrance of a conventional spectrograph:
    *   *Lenslet Arrays:* A grid of small lenses samples the focal plane, with each lenslet focusing light onto a corresponding optical fiber or directly creating a pupil image that is then dispersed.
    *   *Fiber Bundles:* A densely packed bundle of optical fibers collects light from the focal plane, with the fibers rearranged into one or more linear slits feeding the spectrograph(s).
    *   *Image Slicers:* Utilize stacks of small mirrors to slice the 2D field into multiple thin strips, which are then optically rearranged side-by-side to form a long input slit for the spectrograph.
    The raw 2D detector frame contains reformatted spectra from all spatial elements, often interleaved in complex patterns. Data reduction involves extracting the spectrum corresponding to each spatial element (spaxel) and reconstructing the 3D data cube. IFS is extremely powerful for studying the internal kinematics, chemical composition, and physical conditions within extended objects like galaxies, nebulae, and supernova remnants (e.g., MUSE science based on Bacon et al., 2010).

Regardless of the specific type, the output 2D raw frame ($I_{raw}(p_x, p_y)$ where $p_x, p_y$ are pixel coordinates) encodes both spatial and spectral information, contaminated by instrumental signatures (Chapter 3) and the detector response. The fundamental goal of spectroscopic reduction is to isolate the spectrum corresponding to the astronomical source(s) of interest, calibrate its flux and wavelength scales accurately, and remove instrumental and atmospheric artifacts. Key parameters characterizing a spectrograph's performance include its **wavelength coverage** (the range of wavelengths observed), **spectral resolution** ($R$), **dispersion** (the change in wavelength per pixel, e.g., Å/pixel), and **throughput** (overall efficiency).

**4.2 Representing Spectroscopic Data (`specutils`)**

Effective computational analysis of spectroscopic data requires standardized data structures that encapsulate not only the measured flux values but also the corresponding spectral axis coordinates (wavelength, frequency, or energy), associated uncertainties, data quality flags (masks), and essential metadata. While a simple 1D spectrum might initially seem representable by two NumPy arrays (one for wavelength, one for flux), this approach lacks robustness and fails to adequately handle uncertainties, units, and metadata critical for scientific analysis and interoperability. Managing these distinct but related pieces of information separately increases the risk of errors, such as applying operations without considering units, losing track of data quality, or mismatching flux values with their corresponding spectral coordinates. A more integrated approach enhances reliability and simplifies complex analysis workflows.

The **`specutils`** package, an Astropy-affiliated library, provides a dedicated object-oriented framework in Python for representing and manipulating astronomical spectra (Astropy Collaboration et al., 2022; Crawford et al., 2023). It aims to provide a common data model and functionalities for spectral analysis, analogous to how `CCDData` serves imaging data. The core class is `specutils.Spectrum1D`. A `Spectrum1D` object acts as a container holding:
*   **`flux`:** An `astropy.units.Quantity` array representing the measured flux (or signal) values. Using `Quantity` ensures that physical units (e.g., erg s⁻¹ cm⁻² Å⁻¹, Jy, counts) are explicitly associated with the data and handled correctly during operations, preventing unit conversion errors.
*   **`spectral_axis`:** An `astropy.units.Quantity` array defining the corresponding coordinates along the spectral dimension (wavelength, frequency, energy, or sometimes pixel number if uncalibrated). It must have the same shape as the `flux` array along the spectral dimension. `specutils` includes tools for handling different spectral coordinate systems and conversions (e.g., wavelength to frequency or velocity).
*   **`uncertainty`:** An optional uncertainty object (conforming to `astropy.nddata.NDUncertainty`) representing the uncertainty associated with each flux value. Common types include `StdDevUncertainty` (standard deviation) or `VarianceUncertainty`. Proper uncertainty propagation is crucial for quantitative spectral analysis (e.g., line fitting, model comparison), and storing it alongside the flux ensures consistency.
*   **`mask`:** An optional boolean NumPy array indicating bad or unreliable data points (e.g., pixels affected by cosmic rays, strong sky line residuals, detector defects). Masked points are typically ignored in calculations performed by `specutils` functions.
*   **`meta`:** A dictionary-like object (often an `astropy.io.fits.Header`) containing metadata associated with the spectrum (e.g., observation details, instrument setup, processing history, WCS information if derived from a higher-dimensional product). This preserves essential contextual information.
*   **`wcs`:** An optional `astropy.wcs.WCS` object, particularly relevant if the spectrum was extracted from a higher-dimensional dataset (like an IFU cube or multi-order echelle frame) and retains spatial or other coordinate information beyond the primary spectral axis.

Using `Spectrum1D` offers several advantages over managing separate arrays: it ensures data integrity by keeping related components together; leverages `astropy.units` for robust unit handling; provides a standard location for metadata; enhances interoperability between different analysis tools expecting this format; and enables specialized `specutils` operations (resampling, continuum fitting, line analysis) that correctly operate on the combined flux, spectral axis, uncertainty, and mask information. For higher-dimensional spectroscopic data, such as IFU cubes or time-series spectra, `specutils` conceptually extends this model, often involving collections of `Spectrum1D` objects or integration with related libraries like `NDCube`. Loading data from FITS files into `Spectrum1D` structures often involves careful mapping of FITS header keywords (especially WCS) to the object attributes, facilitated by helper functions within `specutils` and `astropy.io`. Adopting standardized structures like `Spectrum1D` promotes cleaner, more robust, and maintainable spectroscopic analysis workflows.

**4.3 Tracing Spectral Orders/Fibers**

In many spectroscopic configurations – particularly echelle spectrographs, multi-object fiber spectrographs (MOS), and some IFU designs – the raw 2D detector frame contains multiple distinct spectral traces that need to be accurately located before individual spectra can be extracted. For echelle data, these are the different, often curved or tilted, spectral orders produced by the combination of the echelle grating and cross-disperser. For MOS or fiber-fed IFUs, these are the individual spectra corresponding to each object or sky fiber projected onto the detector, often arranged in complex patterns determined by the fiber routing and spectrograph optics. The process of precisely identifying the central path of each spectrum across the detector pixels is known as **spectral tracing**. Accurate tracing is fundamental, as errors in locating the spectrum's center will lead to incorrect flux extraction (collecting flux from adjacent orders/fibers or missing flux from the target), biased background subtraction, and potentially significant wavelength calibration errors if the dispersion solution is assumed to be constant along incorrect spatial positions (Piskunov & Valenti, 2002; Bolton & Schlegel, 2010).

Tracing is typically performed using dedicated calibration exposures where the spectra are strongly illuminated and their positions are clearly defined, such as:
*   **Continuum Lamp Flats:** Exposures taken with a bright broadband continuum lamp (e.g., a halogen lamp) illuminating the spectrograph entrance aperture(s). These produce relatively uniform illumination along each spectral trace, making the spatial profile at each wavelength easy to locate via centroiding or profile fitting. These are often the preferred frames for tracing.
*   **Arc Lamp Exposures:** Exposures using calibration lamps that produce sharp emission lines (e.g., ThAr, NeAr). While the illumination is non-uniform spectrally (only bright at line locations), the positions of these bright lines across the detector can be accurately measured and used to map the trace geometry. This can be useful if continuum flats are unavailable or have low signal in certain regions.
*   **Bright Object Exposures:** Sometimes, exposures of a bright star (if observing point sources) or a uniformly illuminated source like the twilight sky or a dome flat screen can be used. However, continuum lamps generally offer better stability and signal levels optimized for tracing across the full detector range.

The goal is to determine the spatial coordinate (e.g., the Y-pixel row) of the center of the spectral trace as a function of the dispersion coordinate (e.g., the X-pixel column) for each individual spectrum or order present on the detector. Common algorithmic approaches include:

1.  **Centroiding:** For each column (or row, depending on dispersion direction) along the dispersion axis, calculate the flux-weighted centroid of the signal in the spatial direction within a predefined window expected to contain the trace. This yields a set of $(p_x, y_{centroid})$ points along the trace for each spectral column $p_x$.
    $y_{centroid}(p_x) = \frac{\sum_{p_y} I(p_x, p_y) \times p_y}{\sum_{p_y} I(p_x, p_y)}$
    where $I(p_x, p_y)$ is the flux in pixel $(p_x, p_y)$ and the sum is over the spatial window in $p_y$. Simple centroiding can be sensitive to noise or asymmetric profiles, especially in low signal-to-noise regions. Variations include fitting a Gaussian profile to the spatial slice at each $p_x$ and using the Gaussian center as $y_{centroid}$.
2.  **Profile Fitting:** Assume a functional model for the spatial profile of the spectrum perpendicular to the dispersion direction (e.g., a Gaussian, Moffat, or an empirically determined profile shape). Fit this profile model to the pixel data in each column (or row) within the trace window, allowing the central position ($y_{center}$) to be a free parameter in the fit. This method can provide more robust center estimates than simple centroiding, particularly in noisy data or when dealing with non-Gaussian profiles, but requires a reasonable model for the profile shape.
3.  **Cross-Correlation:** If a template spatial profile is known or can be reliably estimated from bright parts of the trace, this template can be cross-correlated with the data along the spatial direction in each column. The location of the peak in the cross-correlation function indicates the best-fit center of the trace at that column. This can be effective if the profile shape is stable along the trace.

Once a set of discrete $(p_x, y_{center})$ points defining the path of a trace has been determined, a mathematical function is typically fitted to these points using a least-squares approach. This provides a continuous representation of the trace's geometry across the entire dispersion range, smoothing over noise in the individual center measurements and allowing interpolation/extrapolation to all pixel columns. Low-order polynomials (e.g., quadratic or cubic) or more flexible spline functions are commonly employed for this fit:

$y_{trace}(p_x) = P(p_x) = c_0 + c_1 p_x + c_2 p_x^2 + ...$

The resulting function, $y_{trace}(p_x)$, defines the central row coordinate for the spectrum at each column coordinate $p_x$ and is essential input for the subsequent spectral extraction step (Section 4.4).

Challenges in spectral tracing include handling **low signal-to-noise** regions where the trace is faint, accurately modeling the **curvature or tilt** often present in spectral traces due to optical effects, and dealing with **crowding** in densely packed MOS or IFU data where spectra may overlap. Automated data reduction pipelines (e.g., PypeIt - Prochaska et al., 2020) incorporate sophisticated algorithms to perform robust tracing, often involving iterative refinement, identification of multiple traces simultaneously, and using information from instrument models or multiple calibration frames. Accurate tracing is a critical prerequisite for obtaining high-quality extracted spectra.

**4.4 Optimal Spectral Extraction Algorithms**

Once the geometric path of each spectral trace across the 2D detector has been accurately determined (Section 4.3), the next critical step is **spectral extraction**. This process involves summing the flux along the spatial direction (perpendicular to the dispersion axis) at each position along the dispersion axis to collapse the 2D spectral trace into a 1D spectrum representing flux versus dispersion coordinate (pixel or wavelength). The goal is to perform this summation in a way that maximizes the signal-to-noise ratio (SNR) of the resulting 1D spectrum while accurately preserving the relative flux information and minimizing contamination from background noise or neighboring sources (Horne, 1986; Marsh, 1989). The choice of extraction algorithm significantly impacts the quality of the final spectrum, especially for faint targets or in noisy data.

The simplest extraction method is **boxcar extraction** or **simple summation**. This involves defining a fixed spatial window (aperture) of width $2w+1$ pixels centered on the spectral trace (using the trace function $y_{trace}(x)$ from Section 4.3). For each column $x$ along the dispersion axis, the flux values of all pixels within this window are summed after subtracting the local background $B(x, y)$:

$F_{boxcar}(x) = \sum_{y=y_{trace}(x) - w}^{y_{trace}(x) + w} (I(x, y) - B(x, y))$

While straightforward to implement, boxcar extraction is generally suboptimal in terms of SNR. It assigns equal weight ($w=1$) to every pixel within the aperture. This means noisy pixels at the faint wings of the spatial profile, which contribute very little astrophysical signal, add just as much noise (particularly read noise) to the sum as the bright pixels near the trace center containing most of the signal. Consequently, the SNR of the extracted spectrum is often lower than achievable with more sophisticated methods. Furthermore, the result is sensitive to the choice of aperture width $w$: too narrow loses source flux, while too wide includes excessive background noise. Simple summation is also highly susceptible to unmasked cosmic rays or bad pixels falling within the aperture.

**Optimal extraction**, based on the principles derived by Horne (1986) and Marsh (1989), provides a statistically rigorous method to maximize the SNR by employing variance-weighting. It recognizes that pixels should contribute to the sum based on how much signal they are expected to contain relative to their noise level. Pixels near the peak of the spatial profile, where the signal is strong, should receive higher weight, while pixels in the noisy wings should be down-weighted. The optimal weighting scheme minimizes the variance (and thus maximizes the SNR) of the final extracted flux estimate.

Assuming the spatial profile of the spectrum perpendicular to the dispersion direction at column $x$ can be modeled by a normalized profile function $P(y; x)$ (where $y$ is the spatial coordinate relative to the trace center $y_{trace}(x)$, and $\int P(y; x) dy = 1$), and the variance (noise squared) of the signal in pixel $(x, y)$ is accurately known as $\sigma^2(x, y)$, the optimally extracted flux $F_{opt}(x)$ is calculated via a weighted sum:

$F_{opt}(x) = \frac{\sum_{y} w(x, y) (I(x, y) - B(x, y))}{\sum_{y} w(x, y) P(y; x)}$

where the optimal weights $w(x, y)$ are inversely proportional to the variance and directly proportional to the square of the expected signal profile at that pixel:

$w(x, y) = \frac{P(y; x)^2}{\sigma^2(x, y)}$

This weighting scheme ensures that pixels with a higher expected signal-to-variance ratio contribute more significantly to the final extracted flux estimate. The denominator in the $F_{opt}(x)$ equation ensures correct normalization, preserving the total flux.

Implementing optimal extraction requires several key inputs:
1.  **Accurate Trace:** The trace center $y_{trace}(x)$ must be precisely known from the tracing step.
2.  **Spatial Profile Model $P(y; x)$:** This profile describes the distribution of light perpendicular to the dispersion. It can be determined empirically by fitting functions (e.g., Gaussian, Moffat) to the data in bright spectral regions, or by using a non-parametric smoothed version of the data itself. The profile might vary slowly with wavelength $x$ due to changes in focus or optical aberrations, which should ideally be accounted for.
3.  **Variance Map $\sigma^2(x, y)$:** An accurate pixel-by-pixel estimate of the noise variance is critical. This requires knowledge of the detector gain ($g$, in $e^-/\mathrm{ADU}$) and read noise ($\sigma_{read}$, in $e^-$). The variance typically includes contributions from read noise and Poisson noise from the source signal $S_{source}$, background $B$, and dark current $D$ (all converted to electrons):
    $\sigma^2_{e^-}(x, y) \approx \sigma_{read}^2 + S_{source}(x, y) + B_{e^-}(x, y) + D_{e^-}(x, y)$.
    Since $S_{source}$ depends on the flux being estimated, an iterative approach is often used: an initial flux estimate (e.g., from boxcar extraction) is used to calculate initial variances, optimal extraction is performed, and the resulting flux estimate is used to refine the variances for a subsequent extraction iteration.
4.  **Background Estimate $B(x, y)$:** A reliable estimate of the local sky or background level at each pixel $(x, y)$ is needed for subtraction before the weighted summation. This is often derived from adjacent spatial regions (Section 4.7).

**Advantages of Optimal Extraction:** It significantly improves SNR compared to boxcar methods, especially for faint sources or in read-noise dominated regimes. It is inherently robust against cosmic rays and bad pixels; these pixels typically have high flux or are flagged in masks, leading to very high estimated variance $\sigma^2(x, y)$ and thus extremely low weight $w(x, y)$, effectively excluding them from the sum without explicit masking in the algorithm itself (though prior masking is still good practice). It is also less sensitive to the exact boundaries of the extraction region compared to boxcar summation.

**Challenges:** Optimal extraction relies heavily on the accuracy of the spatial profile model and the variance map. Errors in either of these inputs can degrade the optimality and potentially introduce systematic biases. For instance, mismatches between the assumed profile and the true profile can lead to flux loss or biased results. Handling complex situations like blended spectra in crowded MOS fields requires more advanced profile fitting techniques that simultaneously model multiple overlapping sources (e.g., Bolton & Schlegel, 2010). Modern reduction pipelines like PypeIt (Prochaska et al., 2020) and ESOReflex workflows implement sophisticated optimal extraction routines, often including iterative refinement and robust profile/variance estimation tailored to specific instruments. The output of optimal extraction includes the 1D flux spectrum and, crucially, the propagated variance spectrum representing the uncertainty on the extracted flux at each wavelength.

**4.5 Wavelength Calibration**

After extracting a 1D spectrum ($F(p)$ vs. pixel index $p$), the crucial step of **wavelength calibration** is required to transform the instrumental pixel axis into a physical spectral coordinate axis, typically wavelength ($\lambda$) in units like Angstroms (Å) or nanometers (nm). This establishes the dispersion solution, $\lambda(p)$, which is the mathematical function relating pixel index $p$ to wavelength $\lambda$. Accurate wavelength calibration is fundamental for identifying spectral features, measuring Doppler shifts (velocities), comparing spectra with models or other observations, and performing virtually any quantitative spectroscopic analysis (Kerber et al., 2008; Murphy et al., 2007). Without a reliable wavelength scale, a spectrum remains merely a sequence of intensity values versus detector position, devoid of direct physical meaning regarding energy or atomic/molecular transitions.

The process relies on observing a calibration source whose spectrum contains features (usually sharp emission lines) occurring at wavelengths that are known with high precision from laboratory measurements or established astronomical standards. Common calibration sources include:
*   **Arc Lamps:** These are the workhorse for wavelength calibration in optical and near-infrared spectroscopy. Gas discharge lamps containing specific elements or mixtures (e.g., Thorium-Argon (ThAr), Neon-Argon (NeAr), Helium-Neon-Argon (HeNeAr), Copper-Argon (CuAr)) produce a rich spectrum composed of numerous narrow emission lines when ionized by an electric current. The wavelengths of these lines have been meticulously measured in laboratories to high accuracy and are compiled into standard reference line lists. ThAr lamps are particularly popular for high-resolution optical spectrographs due to their dense and relatively uniform distribution of lines across a wide wavelength range. The choice of lamp depends on the spectrograph's operational wavelength range. Arc lamp exposures are typically taken as separate calibration frames immediately before or after science exposures, or sometimes interleaved, to minimize the impact of potential instrument flexure or thermal drifts that could shift the spectrum on the detector between the calibration and science observations.
*   **Night Sky Emission Lines:** The Earth's upper atmosphere naturally emits light at specific, discrete wavelengths, creating a background 'sky spectrum' that is superimposed on ground-based astronomical observations. This airglow spectrum contains numerous emission lines, particularly from hydroxyl (OH) radicals in the near-infrared, as well as lines from O₂, Na, and atomic oxygen ([OI] 5577 Å, 6300 Å, 6363 Å) in the optical (Rousselot et al., 2000; Sullivan & Simcoe, 2012). If the wavelengths of these sky lines are known accurately (e.g., from compilations like Osterbrock et al., 1996 or specialized databases) and the lines are sufficiently strong, narrow, and unblended in the observed spectrum, they can serve as an *internal*, simultaneous wavelength reference. This is extremely valuable because it calibrates the wavelength scale at the exact time and telescope pointing as the science observation, automatically accounting for any instrumental shifts. Using sky lines is standard practice in near-infrared spectroscopy where stable, broadband arc lamps can be scarce and OH emission is strong (e.g., Maraston et al., 2020). However, sky line intensities can vary significantly over time, and blending can be an issue at lower spectral resolutions.

The wavelength calibration procedure typically involves two main stages: identifying the known lines in the observed calibration spectrum (whether from an arc lamp or the night sky) and fitting a mathematical model (the dispersion function) to relate their measured pixel positions to their known laboratory wavelengths.

*   **4.5.1 Line Identification**
    This critical step involves establishing a reliable correspondence between features detected in the observed calibration spectrum and lines in a known reference wavelength list.
    1.  **Extract Calibration Spectrum:** The 1D spectrum of the calibration source (arc lamp or sky) must first be extracted from its 2D frame using the same tracing (Section 4.3) and extraction (Section 4.4) procedures applied to the science target. This ensures the calibration spectrum experiences the same geometric distortions and extraction effects as the science spectrum.
    2.  **Detect Peaks:** Emission lines in the extracted calibration spectrum, $F_{calib}(p)$, are located. This usually involves algorithms that search for local maxima significantly exceeding the surrounding background noise level. Once peaks are detected, their precise centroid positions ($p_i$) along the pixel axis are measured, often using techniques like Gaussian fitting, quadratic interpolation, or weighted centroiding to achieve sub-pixel accuracy. The accuracy of these centroid measurements directly impacts the final wavelength solution precision.
    3.  **Initial Guess & Matching Strategy:** An initial, approximate dispersion solution ($\lambda_{guess}(p) \approx c_0 + c_1 p$) is usually required to start the matching process. This guess might be derived from instrument design parameters, header keywords, previous calibration solutions, or by manually identifying a few bright, unambiguous lines and calculating a rough linear dispersion. Using this initial guess, approximate wavelengths are calculated for all detected line centroids $p_i$.
    4.  **Automated Matching:** The core task is to automatically and robustly match the list of detected line pixel positions ( $p_i$ ) and their approximate wavelengths ( $\lambda_{guess}(p_i)$ ) to the reference list containing accurately known laboratory wavelengths ( $\lambda_{lab, j}$ ) for the specific lamp or sky species. This is often challenging due to potential non-linearities in the dispersion, instrumental shifts, missing lines (below detection threshold), spurious detections (noise peaks, cosmic rays), and line blending. Common automated matching algorithms employ techniques like:
        *   **Pattern Matching:** Comparing patterns of separations (in pixels or approximate wavelength) between groups of detected lines to patterns in the reference list.
        *   **Cross-Correlation:** Cross-correlating the observed spectrum (or a list of detected line positions) with a synthetic spectrum generated from the reference line list using the initial guess dispersion. The peak of the cross-correlation function indicates the shift needed to align the lists.
        *   **Iterative Refinement:** Starting with a few robust initial matches, iteratively fitting a low-order dispersion solution, predicting the positions of other reference lines, searching for detected lines near the predicted positions, adding new secure matches, and refitting the solution with increasing polynomial order or sophistication. Robust outlier rejection (e.g., using RANSAC or sigma-clipping on fit residuals) is essential during this iterative process to discard misidentifications.
    The output of this crucial stage is a high-fidelity list of matched pairs: measured pixel centroids ( $p_i$ ) and their corresponding accurately known laboratory wavelengths ( $\lambda_{lab, i}$ ). The number and distribution of these matched pairs across the detector range significantly influence the quality of the final dispersion fit.

*   **4.5.2 Fitting Dispersion Solutions (`specutils`)**
    With a reliable list of $(p_i, \lambda_{lab, i})$ pairs identified, the next step is to determine the mathematical function $\lambda(p)$ – the dispersion solution – that best describes the relationship between pixel coordinate $p$ and wavelength $\lambda$ across the entire spectrum.
    1.  **Choose Model Function:** A suitable mathematical function must be selected to represent the dispersion curve $\lambda(p)$. The choice depends on the spectrograph's characteristics and the required accuracy.
        *   **Polynomials:** Low-order polynomials are very common for many spectrographs, especially those with relatively simple optical paths where dispersion is expected to be smooth and slowly varying. A polynomial of degree $n$ is given by:
            $\lambda(p) = \sum_{k=0}^{n} c_k p^k = c_0 + c_1 p + c_2 p^2 + ... + c_n p^n$
            The order $n$ (typically ranging from 2 to 5) must be chosen carefully: high enough to capture any genuine non-linearity in the dispersion curve, but low enough to avoid overfitting the noise present in the measured line centroids ($p_i$). Overfitting can lead to spurious oscillations in the solution between the calibration lines.
        *   **Orthogonal Polynomials:** Basis functions like Chebyshev or Legendre polynomials are often preferred over standard polynomials, especially for higher orders ($n > 3-4$), because they exhibit better numerical stability and less correlation between coefficients during the fitting process.
        *   **Splines:** For highly complex or irregular dispersion curves, piecewise spline functions can offer more flexibility than global polynomials, fitting different smooth functions to different segments of the pixel range while ensuring continuity at the connection points (knots).
        The appropriate model might be suggested by instrument documentation or determined empirically by assessing the fit quality for different models.
    2.  **Fit the Model:** The coefficients ( $c_k$ for polynomials, or equivalent parameters for splines) of the chosen model function are determined by fitting the function $\lambda(p)$ to the identified $(p_i, \lambda_{lab, i})$ data points. This is almost universally done using a weighted least-squares minimization algorithm. Weights ( $w_i$ ) are often assigned to each data point based on the uncertainty in its pixel centroid measurement ( $\sigma_{p_i}$  ) or the signal-to-noise ratio of the identified line, giving more influence to precisely measured, bright lines: $w_i \propto 1/\sigma_{p_i}^2$ or $ w_i \propto SNR_i^2 $ . The goal is to find the coefficients $c_k$ that minimize the weighted sum of squared residuals:
        Minimize $\chi^2 = \sum_{i} w_i (\lambda_{lab, i} - \lambda(p_i))^2 = \sum_{i} w_i (\lambda_{lab, i} - \sum_{k=0}^{n} c_k p_i^k)^2$
        Standard libraries like `numpy.polynomial` (for various polynomial bases) or `scipy.optimize.curve_fit` combined with `astropy.modeling` provide robust tools for performing these weighted least-squares fits.
    3.  **Evaluate Fit Quality and Iterate:** After obtaining an initial fit, its quality must be rigorously assessed. The primary diagnostic is the set of residuals for each matched line: $res_i = \lambda_{lab, i} - \lambda(p_i)$. The root-mean-square (RMS) of these residuals provides a global measure of the typical accuracy of the wavelength solution (e.g., in Å or fraction of a pixel). It's crucial to also plot the residuals versus pixel position ($p_i$) or wavelength ($\lambda_{lab, i}$). Systematic trends in the residuals (e.g., waves, curves) indicate that the chosen model function (e.g., polynomial order) is inadequate to describe the true dispersion. Individual points with residuals significantly larger than the RMS (e.g., > 3-5 times the RMS) are likely outliers resulting from misidentified lines or poorly measured centroids. These outliers should be rejected (e.g., using sigma clipping on the residuals), and the dispersion function should be refitted using only the remaining reliable points. This process of fitting, examining residuals, rejecting outliers, and potentially increasing the model complexity (e.g., polynomial order) might be iterated until a satisfactory fit with randomly distributed residuals and an acceptable RMS value is achieved.
    4.  **Apply Solution:** Once a final, validated dispersion solution $\lambda(p)$ is obtained, it is used to transform the pixel axis of the extracted science spectrum $F(p)$ into a wavelength axis $\lambda$. For each pixel $p$ in the science spectrum, the corresponding wavelength $\lambda(p)$ is calculated using the fitted function. This results in the calibrated spectrum $F(\lambda)$ versus wavelength $\lambda$. Often, for convenience or further analysis (like co-addition), the spectrum is then resampled onto a regular grid (e.g., linear in wavelength or $\log \lambda$ ) using flux-conserving interpolation methods (like those available in `specutils.manipulation`). The final wavelength solution (e.g., the fitted coefficients) and the achieved RMS accuracy are critical metadata that must be stored, typically in the FITS header of the calibrated spectrum file.

    Addressing potential instrument flexure, which can cause small shifts in the wavelength solution between calibration and science exposures, is crucial for high-precision work like radial velocity measurements. Using simultaneously observed night sky emission lines as a secondary calibrator can help track and correct for these intra-exposure drifts (Prochaska et al., 2020). High-precision spectrographs often employ active stabilization or simultaneous calibration fibers to minimize such effects (Fischer et al., 2016).

The following Python example conceptually outlines the core steps of fitting a dispersion solution using identified arc line positions and applying it to a spectrum, utilizing `astropy.modeling` for the fitting process. It assumes the line identification step has already yielded a list of corresponding pixel centroids and laboratory wavelengths. The example focuses on fitting a polynomial model and assessing the fit quality via residuals.

```python
# Conceptual Example: Wavelength Calibration using Arc Lines
# Assumes arc_spec and science_spec are 1D spectra (Flux vs Pixel)
# and a reference line list (pixels, wavelengths) is available.

import numpy as np
from astropy.modeling import models, fitting
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    from specutils.manipulation import FluxConservingResampler
    specutils_available = True
except ImportError:
    print("specutils not found, skipping wavelength calibration example.")
    # Set flag or define dummy classes if needed for script to run
    specutils_available = False
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None

import astropy.units as u
import matplotlib.pyplot as plt # For plotting residuals

# --- Dummy Input Data ---
# Assume arc_spec_pix is a Spectrum1D object with flux vs pixel
# (or just the flux array if Spectrum1D not available)
pixels_arc = np.arange(100)
flux_arc = np.zeros_like(pixels_arc, dtype=float)
# Define known arc line pixel positions and wavelengths (replace with real values)
# Format: list of tuples (pixel_centroid, lab_wavelength_Angstrom)
arc_lines = [
    (10.2, 4046.56), (30.5, 4358.33), (51.3, 4678.15),
    (70.1, 5085.82), (92.8, 5460.74)
]
# Add some simulated arc line peaks to flux_arc for visualization
for p, l in arc_lines:
    flux_arc += 100 * np.exp(-0.5 * ((pixels_arc - p) / 1.0)**2)
flux_arc += np.random.normal(0, 1, size=pixels_arc.shape) # Add noise

# Assume science_spec_pix is a Spectrum1D object (flux vs pixel)
# (or just the flux array)
pixels_sci = np.arange(100)
flux_sci = np.sin(pixels_sci / 5.0) * 10 + 50 + np.random.normal(0, 2, size=pixels_sci.shape)
flux_sci_unit = u.adu # Assign units

# --- Perform Wavelength Calibration ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Create Spectrum1D objects if possible
        arc_spec_pix = Spectrum1D(flux=flux_arc*u.adu, spectral_axis=pixels_arc*u.pix)
        science_spec_pix = Spectrum1D(flux=flux_sci*flux_sci_unit, spectral_axis=pixels_arc*u.pix) # Use same pixel axis for simplicity

        # Separate identified pixel positions and lab wavelengths
        p_measured = np.array([p for p, l in arc_lines])
        lambda_lab = np.array([l for p, l in arc_lines]) * u.AA # Use Astropy units

        # --- Fit Dispersion Solution ---
        print("Fitting dispersion solution...")
        # Choose a polynomial model using astropy.modeling.models
        # Start with a reasonable degree, e.g., quadratic (degree=2) or cubic (degree=3)
        fit_degree = 2
        poly_model_init = models.Polynomial1D(degree=fit_degree)

        # Choose a fitter from astropy.modeling.fitting
        # LinearLSQFitter is suitable for standard polynomial fitting
        fitter = fitting.LinearLSQFitter()

        # Perform the fit: model = fitter(model_init, input_coords, output_coords)
        # Here, input is pixel, output is wavelength
        dispersion_solution = fitter(poly_model_init, p_measured, lambda_lab)
        print(f"Fitted Dispersion Solution (Degree={fit_degree}):")
        print(dispersion_solution)

        # --- Evaluate Fit Quality ---
        print("\nEvaluating fit quality...")
        # Calculate the wavelengths predicted by the fitted model at the measured pixel positions
        lambda_fit = dispersion_solution(p_measured)
        # Calculate the residuals (difference between lab wavelength and fitted wavelength)
        residuals = lambda_lab - lambda_fit
        # Calculate the Root Mean Square (RMS) of the residuals
        rms = np.sqrt(np.mean(residuals**2))
        print(f"Fit Residuals ({residuals.unit}): {residuals.value}")
        print(f"RMS of residuals: {rms:.4f}")

        # Plot residuals vs pixel position to check for systematic trends
        plt.figure(figsize=(8, 4))
        plt.scatter(p_measured, residuals.value)
        plt.axhline(0, color='grey', linestyle='--')
        plt.xlabel("Pixel Position")
        plt.ylabel(f"Residuals ({residuals.unit})")
        plt.title(f"Wavelength Fit Residuals (RMS = {rms:.4f})")
        plt.grid(True, alpha=0.3)
        plt.show()

        # --- Apply Solution to Science Spectrum ---
        print("\nApplying dispersion solution to science spectrum...")
        # Calculate the wavelength corresponding to each pixel in the science spectrum
        # Use the .value attribute to pass NumPy array to model if spectral_axis has units
        science_wavelengths = dispersion_solution(science_spec_pix.spectral_axis.value) * lambda_lab.unit

        # Create a new Spectrum1D object with the calibrated wavelength axis
        # The flux array remains the same, only the spectral_axis changes
        science_spec_cal = Spectrum1D(flux=science_spec_pix.flux, spectral_axis=science_wavelengths)
        print(f"Created science spectrum with calibrated wavelength axis.")
        print(f"Wavelength range: {science_spec_cal.spectral_axis.min():.2f} to {science_spec_cal.spectral_axis.max():.2f}")
        print(f"Flux units: {science_spec_cal.flux.unit}")

        # At this point, science_spec_cal holds the wavelength-calibrated spectrum.
        # It could be saved to a FITS file, potentially storing the dispersion
        # solution coefficients and RMS in the header for provenance.

    except Exception as e:
        # Catch any errors during the process
        print(f"An unexpected error occurred during wavelength calibration: {e}")
else:
     # Message if specutils was not available
     print("Skipping wavelength calibration example: specutils was not imported successfully.")

```

The Python script detailed above provides a practical walkthrough of the wavelength calibration process, a critical stage in spectroscopic data reduction. Utilizing dummy data representing an extracted arc lamp spectrum and identified line positions $(p_i, \lambda_{lab,i})$, it employs the `astropy.modeling` framework to define a polynomial function (`Polynomial1D`) representing the dispersion solution $\lambda(p)$. A least-squares fitter (`LinearLSQFitter`) is then used to determine the best-fit coefficients of this polynomial model based on the provided arc line data points. Crucially, the script calculates and displays the residuals between the laboratory wavelengths and those predicted by the fit, along with the Root Mean Square (RMS) error, providing quantitative measures of the calibration accuracy. A plot of residuals versus pixel position is generated to visually inspect for systematic errors, which might indicate the need for a higher-order polynomial or iterative refinement. Finally, the validated dispersion solution model is applied to the pixel coordinates of the science spectrum (also loaded or simulated as a `Spectrum1D` object) to compute the corresponding wavelength axis, resulting in a new `Spectrum1D` object, `science_spec_cal`, which represents the science spectrum accurately calibrated in wavelength space.

**4.6 Spectroscopic Flat-Fielding**

Just as imaging detectors exhibit pixel-to-pixel sensitivity variations, spectrographs also suffer from variations in throughput and response that must be corrected to obtain accurate relative fluxes and continuum shapes. Spectroscopic flat-fielding aims to remove these variations, which can arise from multiple sources: the intrinsic pixel-to-pixel quantum efficiency variations (PRNU) of the detector itself (similar to imaging flats), wavelength-dependent grating efficiency, varying transmission of filters and other optical elements, potentially non-uniform illumination of the slit or fibers, and even wavelength-dependent atmospheric absorption effects for ground-based data (though the latter is often handled separately during flux calibration). Failure to perform adequate spectroscopic flat-fielding can introduce significant spurious structure (both high-frequency noise from PRNU and broad ripples or slopes from instrumental/atmospheric effects) into the final spectrum, severely impacting spectrophotometric accuracy and the analysis of continuum shapes or relative line strengths (Massey et al., 1990; Bessell, 2005).

Spectroscopic flat-fields are typically obtained by observing a source that ideally provides uniform illumination across the entrance aperture (slit/fibers) and possesses a smooth, well-characterized continuum spectrum across the wavelength range of interest. Common sources include:
*   **Continuum Lamps (e.g., Halogen, Quartz lamps):** These are frequently used as they provide a bright, stable continuum source. Exposures are taken through the entire spectrograph system using the same configuration (grating angle, filters, slit width/fiber configuration) as the science observations. The lamp's intrinsic spectrum is relatively smooth but usually not perfectly flat, exhibiting some color temperature and potentially broad absorption/emission features from the lamp itself or internal optics.
*   **Twilight Sky:** Offers illumination closer in spectral shape and spatial distribution to the night sky, potentially providing a better match for correcting slit illumination profiles in long-slit data. However, its spectrum contains solar Fraunhofer lines and changes rapidly in brightness and color, making acquisition challenging.
*   **Dome Flats:** Similar to imaging dome flats, using an illuminated screen inside the dome. Can suffer from non-uniform illumination and color mismatch issues.
*   **Spectrophotometric Standard Stars:** While primarily used for absolute flux calibration (Chapter 5), observations of standard stars with known SEDs can, in principle, be used to derive the *relative* spectral response function of the instrument+atmosphere system. However, this requires very high SNR observations and doesn't easily correct for high-frequency pixel-to-pixel variations or spatial illumination effects unless combined with pixel flats from lamps.

The reduction process for spectroscopic flat-fields involves several steps, often performed on the 2D calibration frames before spectral extraction:
1.  Acquire multiple flat-field exposures ($F_i$) using the chosen source and the relevant instrument configuration.
2.  Process each individual 2D flat frame: subtract the appropriate master bias ($\bar{B}$) and master dark ($\bar{D}_{master}$, correctly scaled to the flat exposure time $t_{exp, flat}$). Let the result be $F'_{i}(p_x, p_y)$.
    $F'_{i}(p_x, p_y) = (F_i(p_x, p_y) - \bar{B}(p_x, p_y)) - \bar{D}_{master}(p_x, p_y) \times \frac{t_{exp, flat}}{t_{exp, dark}}$
3.  Combine the processed 2D flats ($F'_i$) using median or sigma-clipped averaging to create a high-SNR combined 2D flat frame, $\bar{F}'_{combined}(p_x, p_y)$. This frame contains the combined signature of pixel response, slit/fiber illumination, and the lamp/source spectrum shape projected onto the detector.
4.  **Model and Remove Spectral Shape:** The goal is to isolate the pixel response and illumination pattern from the intrinsic spectral shape of the calibration source ($S_{lamp}(\lambda)$ or $S_{lamp}(p_x)$). This is typically done by fitting a smooth function (e.g., low-order polynomial, spline) to the combined flat frame along the dispersion direction ($p_x$), averaging over the spatial direction ($p_y$) within the illuminated trace(s). This fit, $S_{fit}(p_x)$, models the overall spectral shape (lamp spectrum * instrument throughput).
5.  **Create Normalized 2D Flat:** Divide the combined 2D flat frame $\bar{F}'_{combined}(p_x, p_y)$ by this fitted spectral shape $S_{fit}(p_x)$ (broadcast along the spatial dimension). This division removes the dominant wavelength dependence, leaving primarily the spatial structure (slit/fiber profile) and the high-frequency pixel-to-pixel sensitivity variations.
    $\bar{F}_{spatial+pixel}(p_x, p_y) = \frac{\bar{F}'_{combined}(p_x, p_y)}{S_{fit}(p_x)}$
6.  **Normalize (Optional but common):** This 2D frame $\bar{F}_{spatial+pixel}$ is often further normalized by its overall median or mean value to create the final 2D master flat $\bar{F}_{master, 2D}$ with values fluctuating around unity.
7.  **Apply Correction:** The processed 2D science frame ($S'_{sci}(p_x, p_y)$, after bias and dark subtraction) is then divided pixel-by-pixel by this 2D master flat:
    $S_{flat\_corrected, 2D}(p_x, p_y) = \frac{S'_{sci}(p_x, p_y)}{\bar{F}_{master, 2D}(p_x, p_y)}$
    This correction is applied *before* spectral extraction (Section 4.4). Extracting the spectrum from $S_{flat\_corrected, 2D}$ yields a 1D spectrum corrected for both pixel response variations and spatial illumination effects.

    Alternatively, a 1D flat-field correction can be applied *after* extraction. In this case, a 1D response spectrum $R_{instr}(p_x)$ is extracted from the combined 2D flat $\bar{F}'_{combined}$. The lamp spectrum $S_{lamp}(p_x)$ is modeled and removed as in step 4 (by fitting the extracted 1D response or using a known lamp spectrum), and the result is normalized to create a 1D master flat $\bar{F}_{master, 1D}(p_x)$. The extracted 1D science spectrum $S_{sci, extracted}(p_x)$ is then divided by this 1D master flat. This approach primarily corrects for the overall spectral response shape but is less effective at correcting spatial illumination effects or if the pixel response varies significantly along the spatial direction within the extraction aperture. For precise work, the 2D correction is generally preferred.

For IFU data, flat-fielding strategies can be more complex, often involving separate corrections for pixel-to-pixel variations (from dome/internal flats) and relative fiber/slicer throughput variations (also from dome/internal flats), potentially followed by a correction for the global spectral response using twilight sky flats or standard stars (Weilbacher et al., 2020). Accurate spectroscopic flat-fielding is critical for achieving reliable spectrophotometry across the observed wavelength range.

**4.7 Basic Sky Subtraction Techniques for Spectra**

For ground-based astronomical spectroscopy, particularly in the optical and near-infrared, the observed spectrum of a celestial target is inevitably contaminated by superimposed emission from the Earth's atmosphere (the night sky spectrum). This sky emission originates from various processes, including recombination of ions and molecules in the upper atmosphere (airglow, producing numerous emission lines like OH, O₂, Na, [OI]), scattering of natural (moonlight) and artificial light sources (light pollution), and thermal emission from the atmosphere and telescope itself (dominant at longer IR wavelengths) (Massey et al., 1990; Rousselot et al., 2000; Sullivan & Simcoe, 2012). This sky spectrum can be significantly brighter than faint astronomical targets and exhibits complex structure with numerous sharp emission lines, especially in the red optical and near-infrared. Accurate subtraction of the sky background is therefore a critical step in revealing the true spectrum of the astronomical source. Without effective sky removal, the target spectrum remains heavily contaminated, potentially masking faint features or biasing measurements of continuum levels and line strengths.

The strategy for sky subtraction depends heavily on the spectrograph configuration and observing mode:

*   **Long-slit Spectroscopy:** When observing an object smaller than the slit length (e.g., a star, a compact galaxy nucleus, or a specific region within a larger nebula), the portions of the slit extending above and below the target record only the sky spectrum incident on those parts of the detector (plus detector signatures like residual bias or dark current). This provides a simultaneous or near-simultaneous measurement of the sky background adjacent to the target. The procedure typically involves:
    1.  **Identify Sky Regions:** On the 2D reduced frame (bias, dark, flat corrected), define one or more spatial regions along the slit length (i.e., ranges of rows or spatial pixels) that are clearly free from target flux and represent pure sky emission. These regions should ideally be close to the target region to minimize effects of spatial variations in sky brightness or instrument response.
    2.  **Extract Sky Spectrum:** Extract 1D spectra from these designated sky regions using the same spectral tracing (Section 4.3) and extraction parameters (aperture width, profile if using optimal extraction - Section 4.4) as used for the target itself. This ensures consistency in how the sky signal is measured.
    3.  **Combine/Model Sky:** If multiple sky regions were extracted, average or median-combine them to obtain a single, high-SNR master sky spectrum, $S_{sky}(p_x)$. Alternatively, especially if the sky background is expected to vary spatially along the slit (e.g., due to scattered light or detector effects), one can fit a low-order polynomial (e.g., linear or quadratic) to the sky flux values in the spatial direction *at each wavelength* ($p_x$) using the designated sky regions. This model can then be evaluated at the spatial position of the target trace ($y_{trace}(p_x)$) to estimate the sky background $B(x, y)$ at the target location for each pixel before extraction, or to generate a modeled 1D sky spectrum $S_{sky, model}(p_x)$ corresponding to the target's position.
    4.  **Subtract Sky:** Subtract the derived master sky spectrum $S_{sky}(p_x)$ (or the modeled sky spectrum $S_{sky, model}(p_x)$) from the extracted target spectrum $S_{target+sky}(p_x)$ (which contains both target and sky flux) obtained from the target aperture:
        $S_{target}(p_x) = S_{target+sky}(p_x) - S_{sky}(p_x)$
    This method assumes the sky emission spectrum is intrinsically the same at the target position as in the nearby sky regions, and that instrumental effects like variations in slit illumination or detector response have been adequately corrected by flat-fielding.

*   **Multi-Object Spectroscopy (MOS) / Fiber Spectroscopy:** In MOS or fiber-IFU systems, where each spectrum originates from a discrete location (slitlet or fiber) in the focal plane, obtaining a simultaneous sky measurement for each target fiber is usually not possible. Instead, dedicated fibers are strategically placed on predetermined blank sky regions within the instrument's field of view during observation planning (e.g., Dalton et al., 2022).
    1.  **Identify Sky Fibers:** Select the extracted 1D spectra corresponding to the designated sky fibers from the processed dataset.
    2.  **Combine/Model Sky:** Generate a master sky spectrum by averaging or median-combining the spectra from multiple sky fibers. Because fiber throughput can vary slightly even after flat-fielding, and the instrument's line spread function or wavelength solution might vary subtly across the detector, simply averaging sky fibers might not yield a perfect match to the sky seen by a target fiber. More sophisticated methods are often employed by survey pipelines:
        *   **Median Sky:** A simple median combination of all sky fiber spectra.
        *   **Sky Modeling:** Constructing a model of the sky spectrum based on the sky fibers that accounts for spatial variations across the focal plane or detector. This might involve fitting spline functions or using dimensionality reduction techniques like Principal Component Analysis (PCA) to capture the dominant modes of spectral variation across the sky fibers and interpolate the sky spectrum appropriate for each target fiber's location (e.g., Sharp & Parkinson, 2010; Bolton & Schlegel, 2010).
    3.  **Subtract Sky:** Subtract the master sky spectrum (or the modeled sky spectrum appropriate for the target fiber's location and characteristics) from each individual science target spectrum.

*   **Nodding/Chopping Techniques:** For observations where the source fills the aperture (e.g., a point source observed with a single fiber or a small IFU) or where extremely precise sky subtraction is critical (particularly in the infrared where thermal background is high and variable), observers often employ nodding or chopping strategies. This involves taking exposures offset slightly from the target position on presumably blank sky ("sky" frame) interleaved closely in time with the on-target exposures ("object" frame). The telescope "nods" between the object (A) and sky (B) positions, often in an A-B-B-A pattern to average out temporal variations. The sky frame serves directly as a sky background measurement taken under nearly identical conditions (time, airmass, instrument state). The processed sky frame (bias, dark, flat corrected) is then subtracted from the processed object frame: $S_{target} = S_{object} - S_{sky}$. This technique is highly effective at removing temporally varying sky emission and instrumental background offsets, but comes at the cost of reduced observing efficiency, as roughly half the time is spent observing blank sky. Variations include "nodding along the slit" for long-slit observations, where the object is placed at different positions along the slit in consecutive exposures.

**Challenges in Sky Subtraction:** Accurate sky subtraction remains one of the most significant challenges in ground-based spectroscopy, especially for faint targets or in spectral regions dominated by strong, variable sky emission lines (e.g., NIR OH lines - Davies, 2007). Key difficulties include:
*   **Sky Line Variability:** Airglow emission lines can vary in intensity on timescales of minutes, meaning sky frames taken too far apart in time from the science frame may not be representative.
*   **Residual Subtraction Errors:** Imperfect flat-fielding (leaving residual pixel sensitivity variations), instrument flexure causing small wavelength shifts between the object and sky spectra (leading to P-Cygni-like residuals around sharp sky lines), or errors in the sky model can lead to significant positive and negative artifacts after subtraction. These residuals can be particularly problematic near strong sky lines, potentially masking or mimicking faint astrophysical features. Advanced algorithms sometimes attempt to model and remove these residuals post-subtraction (e.g., ZAP - Soto et al., 2016; Kelson, 2003).
*   **Line Spread Function (LSF) Mismatch:** If the sky spectrum is derived from a different location on the detector, through a different fiber, or using a different extraction method, its effective LSF might differ slightly from the target spectrum's LSF. Subtracting spectra with mismatched LSFs, especially around sharp sky lines, can generate significant residuals.
*   **Sky Region Contamination:** Ensuring that designated "sky" regions (in long-slit or fiber observations) are truly free of faint, undetected astronomical sources is crucial, as any flux from underlying objects will contaminate the sky estimate and be erroneously subtracted from the target.

Achieving precise sky subtraction often requires careful observing strategies (e.g., nodding, sufficient sky fibers), meticulous data reduction (especially flat-fielding), and potentially the application of advanced modeling techniques tailored to the specific instrument and data characteristics (Noll et al., 2014).

**4.8 Examples in Practice (Python): Spectroscopic Reduction Steps**

The following examples illustrate practical applications of the core spectroscopic reduction steps discussed in this chapter, using Python libraries like `specutils` and `astropy.modeling`. These snippets provide concrete demonstrations of tasks such as applying a known wavelength solution, performing basic sky subtraction in a long-slit scenario, executing wavelength calibration using arc lamp lines, conceptually outlining optimal extraction, extracting spectra from IFU data cubes, applying flux calibration, and using pipeline-derived wavelength solutions. Each example is framed within a specific astronomical context to highlight the relevance of these techniques across different research areas. Dummy data is often used for clarity, but the principles and function calls are representative of real data workflows.

**4.8.1 Solar: Applying a Known Wavelength Solution**
Solar physics often involves high-resolution spectroscopy where precise wavelength information is critical for measuring Doppler shifts related to plasma flows, oscillations, or identifying elemental lines. Instrument pipelines or calibration databases frequently provide well-characterized dispersion solutions (the function relating pixel position to wavelength) derived from solar Fraunhofer lines or dedicated calibration procedures. This example demonstrates the straightforward process of taking a 1D solar spectrum, initially defined on a detector pixel axis, and applying a known dispersion solution (here, simulated as a simple linear model for simplicity) to convert the spectral axis into physical wavelength units (e.g., Angstroms), making the spectrum ready for physical analysis.

```python
import numpy as np
from astropy.modeling import models
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D, SpectralAxis
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Solar example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt # For plotting

# --- Dummy Input Data ---
# Simulate a solar spectrum segment vs pixel index
pixels = np.arange(500)
# Simulate a Gaussian absorption line profile + continuum + noise
continuum = 1000
line_center_pix = 250
line_width_pix = 10
line_depth = 500
flux = (continuum - line_depth * np.exp(-0.5 * ((pixels - line_center_pix) / line_width_pix)**2))
flux += np.random.normal(0, np.sqrt(continuum)/2.0, size=pixels.shape) # Add photon noise
flux_unit = u.adu # Assume counts/ADU

# --- Known Wavelength Solution ---
# Assume the dispersion solution lambda = f(pixel) is known, e.g., from pipeline/header
# Example: A simple linear dispersion lambda = 6560 + 0.1 * pixel (Angstrom)
# Represented using astropy.modeling for generality
dispersion_model = models.Linear1D(slope=0.1, intercept=6560.0)
wavelength_unit = u.AA

# --- Apply Wavelength Solution ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Create an initial Spectrum1D object with the pixel axis
        # Ensure units are assigned correctly
        solar_spec_pix = Spectrum1D(flux=flux*flux_unit, spectral_axis=pixels*u.pix)
        print(f"Input spectrum vs pixel axis: {solar_spec_pix.spectral_axis.min()} to {solar_spec_pix.spectral_axis.max()}")

        # Apply the dispersion model to the pixel axis to get the corresponding wavelengths
        # The .value attribute extracts the NumPy array if the axis has units
        wavelengths = dispersion_model(solar_spec_pix.spectral_axis.value) * wavelength_unit

        # Create a new Spectrum1D object using the original flux data but the new wavelength axis
        solar_spec_wav = Spectrum1D(flux=solar_spec_pix.flux, spectral_axis=wavelengths)

        print("\nSolar Example: Successfully applied known wavelength solution.")
        print(f"Calibrated wavelength axis: {solar_spec_wav.spectral_axis.min():.2f} to {solar_spec_wav.spectral_axis.max():.2f}")

        # --- Optional: Plot the result ---
        plt.figure(figsize=(8, 4))
        plt.plot(solar_spec_wav.spectral_axis, solar_spec_wav.flux)
        plt.xlabel(f"Wavelength ({solar_spec_wav.spectral_axis.unit})")
        plt.ylabel(f"Flux ({solar_spec_wav.flux.unit})")
        plt.title("Solar Spectrum with Calibrated Wavelength Axis")
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        # Catch potential errors
        print(f"An unexpected error occurred during solar wavelength application: {e}")
else:
     # Message if specutils was not available
     print("Skipping Solar example: specutils was not imported successfully.")
```

The Python code segment above effectively applies a pre-defined wavelength calibration to a solar spectrum initially represented on a detector pixel grid. It begins by creating a `specutils.Spectrum1D` object containing the flux values and their corresponding pixel coordinates. A known dispersion solution, represented here by an `astropy.modeling.Linear1D` model (though more complex models could be used), defines the mathematical relationship between pixel number and physical wavelength. The core operation involves evaluating this model function at each pixel coordinate of the input spectrum's spectral axis to compute the corresponding wavelength value. A new `Spectrum1D` object is then instantiated using the original flux data but replacing the pixel-based spectral axis with the newly calculated wavelength axis (including appropriate `astropy.units`). The resulting `solar_spec_wav` object thus represents the spectrum accurately calibrated in wavelength space, ready for physical analysis such as line identification or Doppler shift measurement. The optional plotting step visually confirms the transformation.

**4.8.2 Planetary: Basic Extraction and Sky Subtraction (Long-slit)**
Observing planets or other extended objects (like cometary comae or nebulae) with a long-slit spectrograph often allows for simultaneous measurement of the target and adjacent sky background within the same exposure. This facilitates sky subtraction, a critical step for removing atmospheric emission lines and scattered light. This example simulates a simplified 2D long-slit dataset containing a spatially confined target superimposed on a sky background spectrum. It demonstrates a basic sky subtraction workflow: defining spatial regions on the 2D frame corresponding to the target and off-target sky, performing simple boxcar extraction (summation) of the flux within these regions to obtain 1D spectra for the target+sky and the sky alone, scaling the sky spectrum appropriately, and finally subtracting the scaled sky spectrum from the target+sky spectrum to isolate the target's signal.

```python
import numpy as np
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Planetary example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt # For plotting

# --- Dummy 2D Long-slit Data Simulation ---
# Assume data is reduced (bias/dark/flat corrected)
# Axis 0: Dispersion (Wavelength), Axis 1: Spatial (along slit)
dispersion_len = 100
spatial_len = 50
dispersion_axis = np.linspace(7000, 8000, dispersion_len) * u.AA # Example wavelength axis
spatial_axis = np.arange(spatial_len) # Pixel index along slit

# Simulate sky background spectrum (e.g., with emission lines)
sky_spectrum_1d = (15 + 5 * np.sin(dispersion_axis.value / 30.0) +
                   20 * np.exp(-0.5 * ((dispersion_axis.value - 7500) / 5.0)**2)) # Sky line
# Simulate target spatial profile (e.g., Gaussian centered on slit)
target_center_spatial = 25.0
target_width_spatial = 3.0
target_profile = np.exp(-0.5 * ((spatial_axis - target_center_spatial) / target_width_spatial)**2)
# Simulate target intrinsic spectrum
target_spectrum_1d = (8 + 4 * np.cos(dispersion_axis.value / 50.0)) # Target continuum shape
# Create 2D data: Sky + Target*Profile + Noise
data_2d = (sky_spectrum_1d[:, np.newaxis] +
           target_profile[np.newaxis, :] * target_spectrum_1d[:, np.newaxis])
# Add Poisson noise (approximate, assuming counts)
data_2d = np.random.poisson(np.maximum(data_2d, 0)).astype(float) # Use counts for Poisson
data_2d += np.random.normal(0, 2, size=data_2d.shape) # Add read noise component
data_unit = u.adu # Assume units are ADU for counts

# --- Define Extraction Regions ---
# Target region (centered around pixel 25)
target_center_pix = int(target_center_spatial)
aperture_radius = int(target_width_spatial * 2) # Aperture width based on profile
target_rows = slice(target_center_pix - aperture_radius, target_center_pix + aperture_radius + 1)
num_target_pixels = target_rows.stop - target_rows.start
print(f"Target extraction aperture: rows {target_rows.start} to {target_rows.stop-1} ({num_target_pixels} pixels wide)")

# Sky regions (offset from target)
sky_offset = aperture_radius + 3 # Pixels away from target edge
sky_width = 10 # Number of pixels in each sky region
sky_rows1 = slice(target_rows.start - sky_offset - sky_width, target_rows.start - sky_offset)
sky_rows2 = slice(target_rows.stop + sky_offset, target_rows.stop + sky_offset + sky_width)
num_sky1_pixels = sky_rows1.stop - sky_rows1.start
num_sky2_pixels = sky_rows2.stop - sky_rows2.start
num_total_sky_pixels = num_sky1_pixels + num_sky2_pixels
print(f"Sky regions: {sky_rows1.start}-{sky_rows1.stop-1} and {sky_rows2.start}-{sky_rows2.stop-1} ({num_total_sky_pixels} pixels total)")

# --- Perform Extraction and Sky Subtraction ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Extract target + sky spectrum (simple sum over target rows)
        print("Extracting target+sky spectrum...")
        target_plus_sky_1d = np.sum(data_2d[:, target_rows], axis=1)

        # Extract sky spectrum from defined sky regions
        print("Extracting sky spectrum...")
        sky_1d_region1 = np.sum(data_2d[:, sky_rows1], axis=1)
        sky_1d_region2 = np.sum(data_2d[:, sky_rows2], axis=1)
        # Calculate average sky flux *per sky pixel*
        sky_1d_avg_per_pixel = (sky_1d_region1 + sky_1d_region2) / num_total_sky_pixels

        # Scale average sky per pixel to match the width of the target aperture
        sky_to_subtract = sky_1d_avg_per_pixel * num_target_pixels

        # Perform sky subtraction
        print("Performing sky subtraction...")
        target_final_1d = target_plus_sky_1d - sky_to_subtract

        # --- Create Spectrum1D Object for the Result ---
        # Assign units to the final flux array
        target_flux_final = target_final_1d * data_unit
        # Create the final Spectrum1D object
        target_spec = Spectrum1D(flux=target_flux_final, spectral_axis=dispersion_axis)

        print("\nPlanetary Example: Basic extraction and sky subtraction complete.")
        print(f"Final target spectrum flux range: {target_spec.flux.min():.1f} to {target_spec.flux.max():.1f} {target_spec.flux.unit}")

        # --- Optional: Plotting for Verification ---
        plt.figure(figsize=(10, 6))
        plt.plot(target_spec.spectral_axis, target_spec.flux, label='Target (Sky Subtracted)')
        # Plot scaled sky per pixel for comparison
        plt.plot(target_spec.spectral_axis, (sky_1d_avg_per_pixel * data_unit), label='Average Sky per Pixel', alpha=0.7, linestyle=':')
        plt.xlabel(f"Wavelength ({target_spec.spectral_axis.unit})")
        plt.ylabel(f"Flux ({target_spec.flux.unit})")
        plt.title("Planetary Spectrum after Basic Extraction and Sky Subtraction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        # Catch potential errors
        print(f"An unexpected error occurred during planetary sky subtraction: {e}")
else:
     # Message if specutils was not available
     print("Skipping Planetary example: specutils was not imported successfully.")

```

The Python script above simulates and processes a simplified long-slit spectroscopic observation, focusing on the common task of sky subtraction for planetary or extended object studies. It first generates a dummy 2D data array representing dispersed light along one axis and spatial position along the slit on the other, including simulated target signal, sky background, and noise. Key steps involve defining distinct spatial regions (slices along the spatial axis) corresponding to the target aperture and one or more off-target sky regions. A basic boxcar extraction is performed by summing the flux within these defined row slices for each wavelength channel, yielding a 1D spectrum for the target combined with sky, and 1D spectra for the sky regions. The sky spectra are averaged (per pixel) and then scaled by the width of the target aperture to estimate the total sky contribution within the target extraction window. This scaled sky spectrum is subtracted from the target+sky spectrum, and the final sky-subtracted result is encapsulated in a `specutils.Spectrum1D` object, associating the resulting flux with the known wavelength axis and units. The optional plot helps visualize the effectiveness of the sky removal.

**4.8.3 Stellar: Wavelength Calibration using Arc Lamp**
Accurate wavelength calibration is paramount in stellar spectroscopy for tasks like measuring precise radial velocities, identifying elemental absorption lines for abundance analysis, or studying stellar atmospheric properties. This process typically relies on calibration exposures of arc lamps taken in sequence with the science observations. The arc lamp produces a spectrum rich in sharp emission lines at precisely known laboratory wavelengths. This example revisits the core logic of wavelength calibration (similar to Section 4.5.2), explicitly framing it within the context of calibrating a stellar spectrum. It involves identifying arc lines in a calibration spectrum, fitting a dispersion model (pixel vs. known wavelength), and applying that model to the pixel axis of the extracted stellar spectrum.

```python
import numpy as np
from astropy.modeling import models, fitting
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Stellar example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt # For plotting

# --- Dummy Input Data (Similar to 4.5.2 but framed for stellar context) ---
# Simulate extracted arc lamp spectrum (flux vs pixel)
pixels_arc = np.arange(200) # Longer pixel range example
flux_arc = np.random.normal(5, 1, size=pixels_arc.shape) # Base noise
# Define known arc line pixel positions and wavelengths (e.g., ThAr lines)
arc_lines = [ # (pixel_centroid, lab_wavelength_Angstrom) - More lines for better fit
    (15.8, 5015.68), (32.1, 5187.75), (55.3, 5341.09), (81.9, 5506.12),
    (107.2, 5650.70), (135.5, 5801.99), (161.0, 5928.80), (188.7, 6074.34)
]
# Add simulated arc line peaks
for p, l in arc_lines:
    flux_arc += 200 * np.exp(-0.5 * ((pixels_arc - p) / 1.2)**2)

# Simulate extracted stellar spectrum (flux vs pixel)
pixels_star = np.arange(200)
# Realistic continuum + absorption lines (e.g., simplified Balmer + metal lines) + noise
continuum_star = 1500 * (1 - 0.0005 * (pixels_star - 100)) # Sloped continuum
flux_star = continuum_star * (
    (1 - 0.6 * np.exp(-0.5 * ((pixels_star - 40) / 3.0)**2)) * # Line 1
    (1 - 0.4 * np.exp(-0.5 * ((pixels_star - 95) / 4.0)**2)) * # Line 2
    (1 - 0.5 * np.exp(-0.5 * ((pixels_star - 170) / 3.5)**2))  # Line 3
)
flux_star += np.random.normal(0, np.sqrt(np.maximum(flux_star, 0))/2 + 5, size=pixels_star.shape) # Noise model
flux_star_unit = u.electron # Assume flux in electrons after gain correction

# --- Perform Wavelength Calibration ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Create Spectrum1D objects if possible
        arc_spec_pix = Spectrum1D(flux=flux_arc*u.adu, spectral_axis=pixels_arc*u.pix) # Arc often in ADU
        star_spec_pix = Spectrum1D(flux=flux_star*flux_star_unit, spectral_axis=pixels_star*u.pix)

        # Extract measured pixel positions and lab wavelengths from identified arc lines
        p_measured = np.array([p for p, l in arc_lines])
        lambda_lab = np.array([l for p, l in arc_lines]) * u.AA # Use Astropy units

        # --- Fit Dispersion Solution ---
        print("Fitting dispersion solution using arc lines...")
        # Choose polynomial degree (e.g., cubic for potentially more curvature)
        fit_degree = 3
        poly_model_init = models.Polynomial1D(degree=fit_degree)
        fitter = fitting.LinearLSQFitter() # Or fitting.FittingWithOutlierRemoval for robustness
        # Fit model: wavelength = f(pixel)
        dispersion_solution = fitter(poly_model_init, p_measured, lambda_lab)
        print(f"Fitted Dispersion Solution (Degree={fit_degree}):")
        print(dispersion_solution)

        # --- Evaluate Fit Quality ---
        print("\nEvaluating fit quality...")
        lambda_fit = dispersion_solution(p_measured)
        residuals = lambda_lab - lambda_fit
        rms = np.sqrt(np.mean(residuals**2))
        print(f"RMS of residuals: {rms:.4f}")
        # Optional: Plot residuals as in previous example to check for systematics

        # --- Apply Solution to Stellar Spectrum ---
        print("\nApplying dispersion solution to stellar spectrum...")
        star_wavelengths = dispersion_solution(star_spec_pix.spectral_axis.value) * lambda_lab.unit
        star_spec_cal = Spectrum1D(flux=star_spec_pix.flux, spectral_axis=star_wavelengths)
        print(f"Calibrated stellar spectrum created.")
        print(f"Wavelength range: {star_spec_cal.spectral_axis.min():.2f} to {star_spec_cal.spectral_axis.max():.2f}")

        # --- Optional: Plot calibrated spectra ---
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=False) # Separate x-axis initially
        # Plot arc spectrum vs pixel
        ax[0].plot(arc_spec_pix.spectral_axis, arc_spec_pix.flux, label='Arc Spectrum')
        ax[0].plot(p_measured, np.interp(p_measured, pixels_arc, flux_arc)*u.adu, 'ro', label='Identified Lines')
        ax[0].set_xlabel("Pixel Index")
        ax[0].set_ylabel(f"Flux ({arc_spec_pix.flux.unit})")
        ax[0].set_title('Arc Lamp Spectrum (vs Pixel)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        # Plot stellar spectrum vs calibrated wavelength
        ax[1].plot(star_spec_cal.spectral_axis, star_spec_cal.flux)
        ax[1].set_xlabel(f"Wavelength ({star_spec_cal.spectral_axis.unit})")
        ax[1].set_ylabel(f"Flux ({star_spec_cal.flux.unit})")
        ax[1].set_title('Stellar Spectrum (vs Calibrated Wavelength)')
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Catch potential errors
        print(f"An unexpected error occurred during stellar wavelength calibration: {e}")
else:
     # Message if specutils was not available
     print("Skipping Stellar example: specutils was not imported successfully.")
```

This Python script provides a concrete example of performing wavelength calibration for a stellar spectrum using an associated arc lamp exposure. It simulates both the arc spectrum (with peaks at specified pixel locations corresponding to known laboratory wavelengths) and a stellar spectrum (with simulated absorption features) initially defined on a pixel axis. The core logic mirrors the general wavelength calibration process: it extracts the pixel centroids ($p_i$) and laboratory wavelengths ($\lambda_{lab, i}$) of the identified arc lines, uses `astropy.modeling` to fit a polynomial dispersion solution $\lambda(p)$ to these points via least squares, and calculates the RMS of the fit residuals to quantify the calibration accuracy. This derived dispersion solution is then applied to the pixel axis of the stellar spectrum to generate its corresponding wavelength axis, resulting in a `specutils.Spectrum1D` object calibrated in wavelength. The included plot visualizes both the input arc spectrum (highlighting identified lines) and the final wavelength-calibrated stellar spectrum, clearly showing the transformation from instrumental pixels to physical wavelengths.

**4.8.4 Exoplanetary: Extract Host Star Spectrum (Conceptual)**
In the context of searching for or characterizing exoplanets using the radial velocity (RV) method, obtaining a high-quality spectrum of the host star is paramount. MOS or fiber-fed echelle spectrographs are often used to acquire these spectra efficiently. The raw 2D data frame contains spectra from multiple fibers, including the target star and potentially sky or calibration fibers. Extracting the target star's spectrum requires accurately tracing its path on the detector and then summing the flux along the spatial profile, ideally using an optimal extraction algorithm to maximize the signal-to-noise ratio, which is crucial for precise RV measurements. This conceptual example outlines the core logic of optimal extraction for a single stellar spectrum from a simulated multi-fiber 2D frame, assuming the trace location, spatial profile, variance map, and background level are known or have been previously determined.

```python
# Conceptual Example: Extracting one spectrum from a simulated multi-fiber frame
# using simplified optimal extraction logic.

import numpy as np
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Exoplanetary example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt # For plotting

# --- Assume Inputs Are Known/Calculated ---
# data_2d_multi: Reduced 2D frame (bias/dark/flat corrected) with multiple spectra
# trace_center_func: Function p_x -> y_center defining the star's trace
# spatial_profile_func: Function (p_y, y_center) -> profile_value(p_y) (normalized)
# variance_map: 2D array of pixel variances (in flux units squared)
# background_map: 2D array of the estimated background at each pixel

# --- Dummy Data (highly simplified for one trace) ---
disp_len = 200 # Number of pixels along dispersion
spatial_len = 15 # Number of pixels along spatial direction for this fiber
# Simulate the 2D data for just one fiber trace
data_2d_fiber = np.random.normal(5, 1.5, size=(disp_len, spatial_len)) # Background noise + read noise variance = 1.5^2
# Add a fake star trace with a defined profile
trace_center_func = lambda x: spatial_len/2.0 - 0.5 + 2 * np.sin(x / 50.0) # Curved trace
pixels_disp = np.arange(disp_len)
pixels_spatial = np.arange(spatial_len)
yy, xx = np.meshgrid(pixels_spatial, pixels_disp) # Note: meshgrid order vs image shape convention
# Simulate stellar spectrum shape
stellar_spec_1d = 500 + 200 * np.cos(pixels_disp / 10.0)
# Simulate Gaussian spatial profile
spatial_sigma = 1.8
def gaussian_profile(y, center, sigma=spatial_sigma):
    # Ensure normalization over the discrete pixels
    prof = np.exp(-0.5 * ((y - center) / sigma)**2)
    return prof / np.sum(prof) # Normalize sum to 1 over the spatial pixels

trace_centers = trace_center_func(pixels_disp)
star_signal = stellar_spec_1d[:, np.newaxis] * gaussian_profile(pixels_spatial[np.newaxis, :], trace_centers[:, np.newaxis], sigma=spatial_sigma)
data_2d_fiber += star_signal # Add signal to background
# Assume perfect background subtraction for this example
background_map = np.ones_like(data_2d_fiber) * 5.0
data_minus_bkg = data_2d_fiber - background_map
# Assume variance map is known (read noise squared + photon noise approx)
# Photon noise variance = signal (in e-) if gain=1. Assume gain=1, data in e-.
variance_map = (1.5**2) + np.maximum(data_2d_fiber, 0) # Read noise^2 + Signal_e-
data_unit = u.electron # Assume data is in electrons

# --- Optimal Extraction (Looping implementation for clarity) ---
print("Performing simplified optimal extraction...")
extracted_flux = np.zeros(disp_len)
extracted_variance = np.zeros(disp_len) # Also calculate variance of extracted flux

for i in range(disp_len): # Loop over each column (wavelength step)
    # Get spatial profile P(y) at this column (using the known trace center)
    profile_y = gaussian_profile(pixels_spatial, trace_centers[i], sigma=spatial_sigma)
    # Get variance sigma^2(y) for pixels in this column
    variance_y = variance_map[i, :]
    # Calculate optimal weights w(y) = P(y)^2 / sigma^2(y)
    # Avoid division by zero if variance is zero (e.g., masked pixel)
    weights_y = np.zeros_like(profile_y)
    valid_pix = variance_y > 0
    weights_y[valid_pix] = profile_y[valid_pix]**2 / variance_y[valid_pix]

    # Calculate optimally extracted flux F_opt (Eq. from Sec 4.4)
    numerator = np.sum(weights_y * data_minus_bkg[i, :])
    denominator = np.sum(weights_y * profile_y)

    if denominator > 0: # Avoid division by zero if all weights are zero
        extracted_flux[i] = numerator / denominator
        # Calculate variance of the optimally extracted flux
        # Variance(F_opt) = Sum(P^2 / sigma^2) / (Sum(P^3 / sigma^2))^2 --> Simpler: Var(F_opt) = 1 / Sum(weights)
        # Variance formula from Horne 1986 is sum(P)/sum(P^2/sigma^2), needs profile normalized to flux units not 1
        # Simpler approximation/result often used: Var(F_opt) = 1 / Sum(weights_y/profile_y) or related forms
        # Variance derived by Marsh (1989) is sum(P*w) / (sum(P*w)^2) = 1 / sum(P*w)? Check derivation.
        # Let's use the common result Var(F_opt) = sum(P_i) / sum(w_i) = 1 / sum(P^2/var) ?
        # Variance from Horne (1986) Eq A15 is Variance(f_lambda) = sum P_j / sum (P_j^2 / V_j)
        # Where P_j is the normalized profile, V_j is variance. Here P is normalized to 1.
        extracted_variance[i] = 1.0 / np.sum(weights_y) # Assuming profile P is normalized to sum=1 for weight calc? Revisit exact formula/normalization.
        # A robust approach often relies on propagating variance pixel-by-pixel based on the linear combination.
        # Var(Sum(a_i * x_i)) = Sum(a_i^2 * Var(x_i)). Here a_i = weights_y / denominator_sum?
        # Let's use the simpler formula Var(F_opt) approx sum(weights^2 * variance) / (sum(weights))^2? No.
        # Var(F_opt) = 1 / sum(P^2/var) from several sources. Let's use that.
        var_sum_term = np.sum(profile_y[valid_pix]**2 / variance_y[valid_pix])
        if var_sum_term > 0:
             extracted_variance[i] = 1.0 / var_sum_term
        else:
             extracted_variance[i] = np.inf

    else:
        extracted_flux[i] = 0
        extracted_variance[i] = np.inf # Infinite variance if no valid data

# --- Create Final Spectrum Object ---
if specutils_available:
    try:
        # Assume wavelength solution 'dispersion_solution' is known from calibration
        # wavelengths = dispersion_solution(pixels_disp) * u.AA # Placeholder
        wavelengths = np.linspace(5000, 5500, disp_len) * u.AA # Dummy wavelength axis
        # Create uncertainty object
        from astropy.nddata import StdDevUncertainty
        uncertainty = StdDevUncertainty(np.sqrt(extracted_variance) * data_unit) # Convert variance to std dev

        # Create Spectrum1D object
        host_star_spec = Spectrum1D(flux=extracted_flux * data_unit,
                                    spectral_axis=wavelengths,
                                    uncertainty=uncertainty)

        print("\nExoplanetary Example: Conceptual optimal extraction complete.")
        print(f"Extracted spectrum wavelength range: {host_star_spec.spectral_axis.min():.1f} to {host_star_spec.spectral_axis.max():.1f}")
        print(f"Mean Extracted Flux: {np.mean(host_star_spec.flux):.1f}")

        # --- Optional: Plotting ---
        plt.figure(figsize=(10, 5))
        plt.errorbar(host_star_spec.spectral_axis.value, host_star_spec.flux.value,
                     yerr=host_star_spec.uncertainty.array, fmt='.', ecolor='lightgray', capsize=0, label='Extracted Flux +/- 1sigma')
        plt.plot(host_star_spec.spectral_axis.value, host_star_spec.flux.value, 'b-', label='Extracted Flux')
        plt.xlabel(f"Wavelength ({host_star_spec.spectral_axis.unit})")
        plt.ylabel(f"Flux ({host_star_spec.flux.unit})")
        plt.title("Optimally Extracted Host Star Spectrum (Conceptual)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred during exoplanet host extraction: {e}")
else:
     print("Skipping Exoplanetary example: specutils unavailable.")
```

This conceptual Python code illustrates the core logic behind optimally extracting a single spectrum, such as that of an exoplanet host star, from a 2D spectral image, assuming necessary inputs like the trace position, spatial profile model, variance map, and background map are predetermined. The script iterates through each column (wavelength channel) along the dispersion axis. Within each column, it calculates the optimal weights for each pixel along the spatial axis based on the spatial profile model ($P(y;x)$) and the pixel variance ($\sigma^2(x,y)$), using the formula $w = P^2 / \sigma^2$. These weights are then used to compute a weighted sum of the background-subtracted flux values, which is appropriately normalized to yield the optimally extracted flux $F_{opt}(x)$ for that wavelength channel, maximizing the signal-to-noise ratio. The script also calculates the corresponding variance of the extracted flux (using a standard formula derived from error propagation of the weighted sum). Finally, the resulting 1D arrays of extracted flux and variance are combined with a known wavelength solution to create a `specutils.Spectrum1D` object, representing the high-quality extracted spectrum ready for precise analysis like radial velocity measurement.

**4.8.5 Galactic: Extract Spectrum from IFU Cube Region**
Integral Field Unit (IFU) spectroscopy provides spatially resolved spectra across a 2D field, resulting in a 3D data cube (two spatial dimensions, one spectral). A common analysis task in Galactic astronomy, for example when studying star-forming HII regions or planetary nebulae, is to extract the integrated spectrum from a specific spatial region of interest within the cube. This involves summing the flux from all spatial pixels (spaxels) falling within the defined region at each wavelength step. This example demonstrates this process: loading a simulated IFU data cube, defining a spatial region (e.g., a circular mask in pixel coordinates), summing the flux within this mask along the spectral axis, and constructing the final 1D integrated spectrum using the WCS information to define the wavelength axis.

```python
import numpy as np
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Galactic example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits # Needed if loading real cube
import matplotlib.pyplot as plt # For plotting
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
cube_file = 'ifu_cube_sample.fits' # Assume this exists from Example 2.7.5 or create dummy
output_spec_file = 'hii_region_integrated_spec.fits' # Output 1D spectrum

# Create dummy IFU cube if needed
if not os.path.exists(cube_file):
    print(f"File {cube_file} not found, creating dummy IFU cube file.")
    n_wave, n_y, n_x = 150, 20, 20
    cube_data = np.random.normal(10, 1, size=(n_wave, n_y, n_x)).astype(np.float32)
    # Add a fake emission region spatially and spectrally
    yy, xx = np.indices((n_y, n_x))
    spatial_profile = np.exp(-0.5 * (((xx - 10)/3)**2 + ((yy - 10)/3)**2)) # Gaussian blob
    spectral_profile = np.exp(-0.5 * ((np.arange(n_wave) - 75) / 10.0)**2) # Emission line profile at index 75
    cube_data += 50 * spectral_profile[:, np.newaxis, np.newaxis] * spatial_profile[np.newaxis, :, :]
    # Create dummy WCS header
    w = WCS(naxis=3)
    w.wcs.crpix = [1, n_x/2 + 0.5, n_y/2 + 0.5]
    w.wcs.cdelt = np.array([2.0, -0.5/3600, 0.5/3600])
    w.wcs.crval = [6500, 266.4, -29.0]
    w.wcs.ctype = ['WAVE', 'RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['Angstrom', 'deg', 'deg']
    dummy_header = w.to_header()
    dummy_header['BUNIT'] = '1e-20 erg/s/cm2/A/pix'
    # Assume data in Primary HDU
    hdu0 = fits.PrimaryHDU(cube_data.T, header=dummy_header) # Transpose for FITS order
    hdul = fits.HDUList([hdu0])
    hdul.writeto(cube_file, overwrite=True)

# --- Define Spatial Region for Integration ---
# Example: Define a circular region using pixel coordinates
center_x_pix = 10.0 # Column index (0-based)
center_y_pix = 10.0 # Row index (0-based)
radius_pix = 3.0   # Radius in pixels
print(f"Defining circular region: center=({center_x_pix}, {center_y_pix}), radius={radius_pix} pixels")

# --- Extract Integrated Spectrum ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Load the IFU data cube
        print(f"Loading IFU data cube: {cube_file}")
        with fits.open(cube_file) as hdul:
            # Find the data HDU (similar logic to Example 2.7.5)
            cube_hdu = None
            if 'SCI' in hdul: cube_hdu = hdul['SCI']
            elif 'DATA' in hdul: cube_hdu = hdul['DATA']
            elif hdul[0].data is not None and hdul[0].header['NAXIS'] >= 3: cube_hdu = hdul[0]
            elif len(hdul) > 1 and hdul[1].is_image and hdul[1].header['NAXIS'] >= 3: cube_hdu = hdul[1]
            else: raise IndexError("Data cube HDU not found.")

            cube_data = cube_hdu.data # Shape (n_wave, n_y, n_x) if read normally by astropy
            cube_header = cube_hdu.header
            cube_wcs = WCS(cube_header)
            cube_unit = u.Unit(cube_header.get('BUNIT', 'adu')) # Get units

            # Need dimensions for mask creation (NumPy shape order might differ from FITS NAXIS order)
            shape_numpy = cube_data.shape
            if cube_wcs.naxis == 3: # Assume 3D WCS
                 # Map numpy axes to WCS axes (usually spectral is first or last)
                 spec_axis_numpy = cube_wcs.wcs.spec # Index of spectral axis in WCS (0-based)
                 # Need to figure out which numpy axis corresponds to this
                 # Example logic assuming spec is slowest or fastest changing numpy axis
                 if spec_axis_numpy == cube_wcs.naxis - 1 : # Spectral is last WCS axis (NAXIS1) -> fastest numpy axis
                      numpy_spec_axis_idx = -1
                 elif spec_axis_numpy == 0 : # Spectral is first WCS axis (NAXIS3 or NAXISN) -> slowest numpy axis
                      numpy_spec_axis_idx = 0
                 else: # Intermediate axis? Assume 0 for now. Needs careful check.
                      print("Warning: Cannot robustly determine numpy spectral axis index from WCS. Assuming 0.")
                      numpy_spec_axis_idx = 0

                 spatial_shape = list(shape_numpy)
                 del spatial_shape[numpy_spec_axis_idx]
                 n_y, n_x = spatial_shape[0], spatial_shape[1] # Assumes (spec, y, x) or (y, x, spec) order
                 print(f"Deduced spatial shape: ({n_y}, {n_x})")
            else:
                 raise ValueError("WCS object is not 3-dimensional.")


        # Create a 2D boolean mask for the spatial region
        yy_pix, xx_pix = np.indices((n_y, n_x)) # Create pixel grids
        mask_region = ((xx_pix - center_x_pix)**2 + (yy_pix - center_y_pix)**2 <= radius_pix**2)
        print(f"Created spatial mask: {np.sum(mask_region)} pixels in region.")

        # Integrate flux within the mask at each spectral plane
        # Need to apply mask across correct axes
        print("Integrating flux within spatial mask...")
        if numpy_spec_axis_idx == 0: # Assumes shape (n_wave, n_y, n_x)
             flux_integrated = np.sum(cube_data[:, mask_region], axis=1) # Sum over masked spatial pixels
        elif numpy_spec_axis_idx == 2: # Assumes shape (n_y, n_x, n_wave)
             # Mask needs broadcasting (n_y, n_x, 1) ? No, apply mask first
             masked_data = cube_data[mask_region, :] # Selects pixels, result is (N_pix_in_mask, n_wave)
             flux_integrated = np.sum(masked_data, axis=0) # Sum over pixels in mask
        else: # Handle other cases or raise error
             raise NotImplementedError("Unsupported numpy axis order for integration.")

        # Create the spectral axis using WCS information
        print("Creating spectral axis from WCS...")
        n_wave_points = shape_numpy[numpy_spec_axis_idx]
        spectral_pixel_coords = np.arange(n_wave_points)
        # Use wcs_pix2world, providing dummy spatial coords (e.g., reference pixel)
        # Ensure pixel coords are provided for ALL WCS axes
        all_pix_coords = [0] * cube_wcs.naxis
        all_pix_coords[cube_wcs.wcs.lng] = cube_wcs.wcs.crpix[cube_wcs.wcs.lng] - 1
        all_pix_coords[cube_wcs.wcs.lat] = cube_wcs.wcs.crpix[cube_wcs.wcs.lat] - 1
        all_pix_coords[cube_wcs.wcs.spec] = spectral_pixel_coords
        # Get world coordinates for all axes, then extract spectral axis
        world_coords = cube_wcs.pixel_to_world(*all_pix_coords)
        spectral_axis_values = world_coords[cube_wcs.wcs.spec]
        spectral_axis_unit = u.Unit(cube_wcs.wcs.cunit[cube_wcs.wcs.spec])
        spectral_axis = spectral_axis_values * spectral_axis_unit

        # Create the final 1D Spectrum1D object
        region_spec = Spectrum1D(flux=flux_integrated * cube_unit, # Apply original flux unit
                                 spectral_axis=spectral_axis)
        # Add metadata about the region
        region_spec.meta['REGION'] = f'Circular aperture: center=({center_x_pix:.1f},{center_y_pix:.1f})pix, radius={radius_pix:.1f}pix'
        region_spec.meta['NPIX_SUM'] = np.sum(mask_region)

        print("\nGalactic Example: Integrated spectrum extraction from IFU cube region complete.")
        print(f"Resulting spectrum wavelength range: {region_spec.spectral_axis.min():.1f} to {region_spec.spectral_axis.max():.1f}")

        # --- Optional: Plotting ---
        plt.figure(figsize=(8, 4))
        plt.plot(region_spec.spectral_axis, region_spec.flux)
        plt.xlabel(f"Wavelength ({region_spec.spectral_axis.unit})")
        plt.ylabel(f"Integrated Flux ({region_spec.flux.unit})")
        plt.title(f"Integrated Spectrum from IFU Cube Region")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Optional: Save the extracted spectrum
        # region_spec.write(output_spec_file, overwrite=True) # Needs appropriate writer

    except FileNotFoundError:
        print(f"Error: Cube file not found at {cube_file}.")
    except IndexError:
        print(f"Error: Could not find expected HDU structure or index in {cube_file}.")
    except Exception as e:
        # General error handling
        print(f"An unexpected error occurred during IFU region extraction: {e}")
else:
     # Message if specutils was not available
     print("Skipping Galactic example: specutils was not imported successfully.")
```

This script tackles the task of extracting an integrated 1D spectrum from a user-defined spatial region within a 3D IFU data cube, a common analysis step for studying extended Galactic objects. It begins by loading the FITS data cube and its associated World Coordinate System (WCS) information using `astropy.io.fits` and `astropy.wcs`. A spatial region of interest is defined, in this case, a circular aperture specified by its center coordinates and radius in pixel space. A boolean mask representing this region is created using NumPy array operations. The core operation involves applying this 2D spatial mask to the 3D data cube and summing the flux values of the selected spaxels along the two spatial dimensions for each spectral channel (wavelength slice). Careful handling of NumPy array indexing is required depending on the axis order (e.g., `[wave, y, x]` or `[y, x, wave]`). The corresponding spectral axis (wavelengths) for the resulting 1D spectrum is derived from the cube's WCS object. Finally, the integrated flux array and the derived spectral axis are combined into a `specutils.Spectrum1D` object, representing the total spectrum emitted from the specified spatial region.

**4.8.6 Extragalactic: Apply Wavelength & Flux Calibration**
Obtaining physically meaningful spectra of extragalactic objects, such as galaxies or quasars, requires not only wavelength calibration but also flux calibration. Flux calibration converts the instrumental signal (e.g., counts per second) into absolute physical flux density units (e.g., erg s⁻¹ cm⁻² Å⁻¹ or Jy). This typically involves observing spectrophotometric standard stars – stars with accurately known spectral energy distributions (SEDs) – with the same instrument configuration used for the science target. By comparing the observed spectrum of the standard star (in counts/s) to its known true spectrum (in flux units), a sensitivity function (or inverse sensitivity function) can be derived, representing the instrument's efficiency as a function of wavelength. This example demonstrates applying both a pre-derived wavelength solution and a pre-derived sensitivity function to an extracted galaxy spectrum (assumed to be in counts or count rate units) to produce a final, fully calibrated spectrum in absolute flux units.

```python
import numpy as np
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    # Interpolation might be needed if grids don't match
    from specutils.manipulation import LinearInterpolatedResampler
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Extragalactic example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    # Define dummy resampler if needed
    class LinearInterpolatedResampler:
        def __call__(self, spec, grid): return spec # No-op dummy
    specutils_available = False # Set flag

import astropy.units as u
from astropy.modeling import models
import matplotlib.pyplot as plt # For plotting

# --- Dummy Input Data ---
# Assume galaxy_spec_pix is extracted spectrum (counts vs pixel)
pixels = np.arange(300)
# Simulate galaxy spectrum with continuum break and emission line
continuum = 100 + 100 * (pixels / 300.0) # Rising continuum
break_point = 150
continuum[pixels > break_point] *= 0.6 # Continuum break
emission_line = 200 * np.exp(-0.5 * ((pixels - 200) / 5.0)**2)
flux_counts = np.random.poisson(continuum + emission_line).astype(float) * u.count
exposure_time = 600.0 * u.s # Assume exposure time

# Assume dispersion_solution (pixel -> wavelength) is known (e.g., from arc cal)
# Example: lambda = 3700 + 5.0 * p + 0.001 * p^2 Angstrom
dispersion_solution = models.Polynomial1D(degree=2, c0=3700, c1=5.0, c2=0.001)
wavelengths = dispersion_solution(pixels) * u.AA

# Assume sensitivity_func_wav is known (typically derived from standard star obs)
# Units: (Physical Flux Units) / (Instrumental Rate Units)
# Example: (erg/s/cm2/A) / (count/s)
# Simulate sensitivity dropping at blue/red ends
sensitivity_values = (1e-17 * np.exp(-0.5*((wavelengths.value - 5500)/800)**2) + 0.5e-17) \
                     * u.erg / u.s / u.cm**2 / u.AA / (u.count / u.s)
# Ensure sensitivity function uses the same wavelength grid initially for simplicity
# In reality, it might need resampling (shown conceptually below)
sensitivity_func_wav = Spectrum1D(flux=sensitivity_values, spectral_axis=wavelengths)


# --- Apply Calibrations ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Create Spectrum1D object for the raw counts vs pixel spectrum
        galaxy_spec_counts_pix = Spectrum1D(flux=flux_counts, spectral_axis=pixels*u.pix)

        # Apply wavelength calibration first
        galaxy_spec_counts_wav = Spectrum1D(flux=galaxy_spec_counts_pix.flux,
                                          spectral_axis=wavelengths)
        print("Applied wavelength calibration.")

        # Convert counts to count rate by dividing by exposure time
        print(f"Converting to count rate (Exposure Time = {exposure_time})...")
        galaxy_spec_rate_wav = galaxy_spec_counts_wav / exposure_time
        print(f"Count rate spectrum units: {galaxy_spec_rate_wav.flux.unit}")


        # Apply flux calibration by dividing the count rate spectrum by the sensitivity function
        print("Applying flux calibration using sensitivity function...")
        # CRITICAL: Ensure the sensitivity function is defined on the same spectral axis
        # grid as the science spectrum. If not, resample one to match the other.
        if not np.allclose(galaxy_spec_rate_wav.spectral_axis.value, sensitivity_func_wav.spectral_axis.value):
            print("Warning: Spectral axes of science and sensitivity differ. Resampling sensitivity...")
            # Choose a resampling method (FluxConserving or LinearInterpolated)
            resampler = LinearInterpolatedResampler(extrapolation_treatment='zero_fill') # Fill ends with zero
            # Resample sensitivity function onto the science spectrum's wavelength grid
            sensitivity_resampled = resampler(sensitivity_func_wav, galaxy_spec_rate_wav.spectral_axis)
            # Perform the division using the resampled sensitivity
            galaxy_spec_flux_cal = galaxy_spec_rate_wav / sensitivity_resampled
        else:
            # If grids match exactly, direct division is possible
            galaxy_spec_flux_cal = galaxy_spec_rate_wav / sensitivity_func_wav


        print("\nExtragalactic Example: Wavelength and flux calibration applied.")
        print(f"Final calibrated flux units: {galaxy_spec_flux_cal.flux.unit}")
        print(f"Flux range: {galaxy_spec_flux_cal.flux.min():.2E} to {galaxy_spec_flux_cal.flux.max():.2E}")

        # --- Optional: Plotting ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Plot count rate spectrum
        axes[0].plot(galaxy_spec_rate_wav.spectral_axis, galaxy_spec_rate_wav.flux)
        axes[0].set_ylabel(f"Count Rate ({galaxy_spec_rate_wav.flux.unit})")
        axes[0].set_title("Galaxy Spectrum (Count Rate vs Wavelength)")
        axes[0].grid(True, alpha=0.3)
        # Plot flux calibrated spectrum
        axes[1].plot(galaxy_spec_flux_cal.spectral_axis, galaxy_spec_flux_cal.flux)
        axes[1].set_xlabel(f"Wavelength ({galaxy_spec_flux_cal.spectral_axis.unit})")
        axes[1].set_ylabel(f"Flux Density ({galaxy_spec_flux_cal.flux.unit})")
        axes[1].set_title("Flux Calibrated Galaxy Spectrum")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    except Exception as e:
        # Catch potential errors
        print(f"An unexpected error occurred during extragalactic calibration: {e}")
else:
     # Message if specutils was not available
     print("Skipping Extragalactic example: specutils unavailable.")

```

This script demonstrates the final crucial calibration steps required to transform an observed extragalactic spectrum from instrumental units into physically meaningful units. It begins with a simulated extracted galaxy spectrum represented initially in counts versus pixel coordinates. First, a pre-determined wavelength solution (derived from arc lamps, represented by `dispersion_solution`) is applied to establish the wavelength axis ($\lambda$). The spectrum's flux units are then converted from total counts to count rate (counts/second) by dividing by the exposure time. The core flux calibration step involves dividing this count rate spectrum by a pre-derived sensitivity function (`sensitivity_func_wav`). This sensitivity function, typically obtained from observations of spectrophotometric standard stars (see Chapter 5), quantifies the instrument's response in physical flux units per count rate as a function of wavelength. The script includes logic to handle potential mismatches between the wavelength grids of the science spectrum and the sensitivity function, employing resampling (conceptually shown using `specutils.manipulation.LinearInterpolatedResampler`) if necessary before performing the division. The final `Spectrum1D` object (`galaxy_spec_flux_cal`) represents the fully calibrated galaxy spectrum in absolute flux density units (e.g., erg s⁻¹ cm⁻² Å⁻¹), ready for scientific interpretation and comparison with physical models.

**4.8.7 Cosmology: Apply Pipeline Wavelength Solution (SDSS)**
Large spectroscopic surveys like the Sloan Digital Sky Survey (SDSS) employ sophisticated, automated data reduction pipelines to process vast numbers of spectra. The resulting calibrated spectra are often distributed in FITS files where the wavelength scale corresponding to the flux array is not stored as a separate array column but is instead defined implicitly by specific keywords in the FITS header. A common convention (used by SDSS) is to define a non-linear wavelength solution, often linear in the logarithm of wavelength, using header coefficients. This example demonstrates how to read such a FITS header, extract the relevant coefficients (e.g., `COEFF0` and `COEFF1` for $\log_{10} \lambda = \mathrm{COEFF0} + \mathrm{COEFF1} \times p$), and apply this functional form to the pixel indices to reconstruct the correct wavelength axis for the associated flux data.

```python
import numpy as np
from astropy.io import fits
# Requires specutils: pip install specutils
try:
    from specutils import Spectrum1D
    specutils_available = True
except ImportError:
    print("specutils not found, skipping Cosmology example.")
    # Define dummy Spectrum1D if needed for script structure
    class Spectrum1D:
        def __init__(self, flux=None, spectral_axis=None):
            self.flux = flux
            self.spectral_axis = spectral_axis
            if hasattr(flux, 'unit'): self.flux_unit = flux.unit
            else: self.flux_unit = None
            if hasattr(spectral_axis, 'unit'): self.spectral_axis_unit = spectral_axis.unit
            else: self.spectral_axis_unit = None
    specutils_available = False # Set flag
import astropy.units as u
import matplotlib.pyplot as plt # For plotting
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
# Assume flux is in HDU 1 (coadd), ivar in HDU 2, mask in HDU 3 etc.
# Wavelength solution from HDU 1 header keywords
sdss_spec_file = 'sdss_qso_spec_sample.fits'

# Create dummy file if it doesn't exist
if not os.path.exists(sdss_spec_file):
    print(f"Creating dummy file: {sdss_spec_file}")
    n_pixels = 1000
    # Define typical SDSS wavelength solution keywords
    coeff0 = 3.579 # log10(Angstrom) at pixel 0
    coeff1 = 0.0001 # log10(Angstrom) per pixel
    # Generate dummy flux for a QSO (continuum + broad lines)
    dummy_pixels = np.arange(n_pixels)
    dummy_loglam = coeff0 + coeff1 * dummy_pixels
    dummy_lam = 10**dummy_loglam
    flux = (15 + 5 * (dummy_lam / 5000.0)**-1 + # Power law continuum
            30 * np.exp(-0.5 * ((dummy_lam - 4861) / 50.0)**2) + # Broad Hbeta
            50 * np.exp(-0.5 * ((dummy_lam - 6563) / 70.0)**2) ) # Broad Halpha
    flux += np.random.normal(0, 2, size=n_pixels) # Add noise
    flux_unit_str = "1e-17 erg / (Angstrom cm2 s)"
    # Create HDUs
    hdu0 = fits.PrimaryHDU() # Usually minimal primary
    hdr1 = fits.Header()
    hdr1['NAXIS'] = 1
    hdr1['NAXIS1'] = n_pixels
    hdr1['BUNIT'] = flux_unit_str
    hdr1['COEFF0'] = (coeff0, 'log10(lambda) zeropoint')
    hdr1['COEFF1'] = (coeff1, 'log10(lambda) dispersion')
    hdr1['OBJECT'] = 'SimQSO'
    hdu1 = fits.PrimaryHDU(flux.astype(np.float32), header=hdr1) # Often flux in Primary for older spec-*.fits
    # Add empty extensions for ivar, mask if needed to mimic structure
    hdu2 = fits.ImageHDU(name='IVAR')
    hdu3 = fits.ImageHDU(name='MASK')
    hdul = fits.HDUList([hdu1, hdu2, hdu3]) # Example structure
    hdul.writeto(sdss_spec_file, overwrite=True)


# --- Read Flux and Apply Wavelength Solution ---
if specutils_available: # Proceed only if specutils is imported
    try:
        # Open the SDSS spec FITS file
        print(f"Loading SDSS-like spectrum: {sdss_spec_file}")
        with fits.open(sdss_spec_file) as hdul:
            hdul.info()
            # SDSS spec-* files often have flux in Primary HDU (index 0)
            # Or sometimes in HDU 1 ('COADD') for newer formats. Check needed.
            try:
                spec_hdu = hdul[0] # Assume Primary HDU for older spec format
                flux_data = spec_hdu.data
                header = spec_hdu.header
                print("Accessed flux data from Primary HDU (HDU 0).")
            except IndexError:
                 print("Error: Primary HDU not found.")
                 raise # Cannot proceed
            except ValueError: # Handle case where Primary HDU has no data
                 print("Primary HDU has no data, trying HDU 1 ('COADD')...")
                 try:
                      spec_hdu = hdul['COADD']
                      flux_data = spec_hdu.data
                      header = spec_hdu.header
                      print("Accessed flux data from 'COADD' extension (HDU 1).")
                 except (KeyError, IndexError):
                      print("Error: Could not find flux data in HDU 0 or HDU 1/'COADD'.")
                      raise # Cannot proceed


            # Extract the wavelength solution coefficients from the header
            try:
                coeff0 = header['COEFF0'] # log10(Angstrom) zeropoint
                coeff1 = header['COEFF1'] # log10(Angstrom) dispersion per pixel
                print(f"Found wavelength solution keywords: COEFF0={coeff0:.4f}, COEFF1={coeff1:.6f}")
            except KeyError:
                print("Error: Wavelength solution keywords COEFF0/COEFF1 not found in header.")
                raise # Cannot proceed

            # Extract flux units
            flux_unit = u.Unit(header.get('BUNIT', 'adu')) # Use specified unit or default

        # --- Calculate Wavelength Axis ---
        # Determine the number of pixels from the flux data array shape
        n_pixels = len(flux_data)
        # Create an array of pixel indices (0-based)
        pixel_indices = np.arange(n_pixels)
        # Apply the SDSS log-linear wavelength solution formula
        log_wavelengths = coeff0 + coeff1 * pixel_indices
        wavelengths = 10**log_wavelengths * u.AA # Assume Angstroms

        # --- Create Spectrum1D Object ---
        print("Creating Spectrum1D object with calculated wavelength axis...")
        qso_spec_cal = Spectrum1D(flux=flux_data * flux_unit, spectral_axis=wavelengths)

        print("\nCosmology Example: Applied SDSS pipeline log-linear wavelength solution.")
        print(f"Resulting spectrum wavelength range: {qso_spec_cal.spectral_axis.min():.1f} to {qso_spec_cal.spectral_axis.max():.1f}")
        print(f"Resulting spectrum flux units: {qso_spec_cal.flux.unit}")

        # --- Optional: Plotting ---
        plt.figure(figsize=(10, 5))
        plt.plot(qso_spec_cal.spectral_axis, qso_spec_cal.flux)
        plt.xlabel(f"Wavelength ({qso_spec_cal.spectral_axis.unit})")
        plt.ylabel(f"Flux Density ({qso_spec_cal.flux.unit})")
        plt.title(f"SDSS-like QSO Spectrum (Pipeline Wavelength Calibration)")
        plt.grid(True, alpha=0.3)
        plt.show()

    except FileNotFoundError:
        print(f"Error: Spectrum file not found at {sdss_spec_file}.")
    except IndexError:
        print(f"Error: Could not find expected HDU structure in {sdss_spec_file}.")
    except KeyError as e:
        print(f"Error: Required keyword {e} missing from header.")
    except Exception as e:
        # General error handling
        print(f"An unexpected error occurred during SDSS spectrum processing: {e}")
else:
     # Message if specutils was not available
     print("Skipping Cosmology example: specutils was not imported successfully.")

```

This final example demonstrates how to handle wavelength calibration for spectra distributed by large survey pipelines like SDSS, where the dispersion solution is often encoded in FITS header keywords rather than stored as a separate array. The script opens the FITS file and accesses the HDU containing the 1D flux array (often the primary HDU or a 'COADD' extension). Crucially, it reads specific keywords from the header (`COEFF0` and `COEFF1` in the SDSS convention) that define the parameters of the log-linear wavelength solution: $\log_{10} \lambda = \mathrm{COEFF0} + \mathrm{COEFF1} \times p$, where $p$ is the 0-based pixel index. Using these coefficients, the script calculates the corresponding wavelength value for each pixel index along the flux array. This generated wavelength array, combined with the flux data and appropriate units (also typically read from the header's `BUNIT` keyword), is then used to construct a `specutils.Spectrum1D` object, yielding a fully wavelength-calibrated spectrum ready for analysis, such as redshift determination for cosmological studies.

---

**References**

Allington-Smith, J., & Content, R. (2002). Integral field spectroscopy: A mature technology at last. *Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences, 360*(1795), 1201–1215. https://doi.org/10.1098/rsta.2001.0976 *(Note: Foundational IFU review, pre-2020)*
*   *Summary:* Although published before 2020, this paper provides a foundational overview of Integral Field Unit (IFU) technologies (lenslets, fibers, slicers) described in Section 4.1. It establishes the context for modern IFU instruments like MUSE.

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project, whose ecosystem includes `specutils` (Section 4.2), `astropy.modeling` (used for fitting dispersion solutions, Section 4.5.2), and core data handling capabilities leveraged throughout spectroscopic reduction.

Ayres, T. (2022). High-resolution astronomical spectroscopy in the twenty-first century. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 380*(2234), 20210222. https://doi.org/10.1098/rsta.2021.0222
*   *Summary:* Provides a contemporary overview of high-resolution spectroscopy, touching upon instrument types like echelle spectrographs (Section 4.1) and the scientific drivers requiring precise wavelength calibration (Section 4.5) and data reduction.

Crawford, S. M., Earl, N., Lim, P. L., Deil, C., Tollerud, E. J., Morris, B. M., Bray, E., Conseil, S., Donath, A., Fowler, J., Ginsburg, A., Kulumani, S., Pascual, S., Perren, G., Sipőcz, B., Weaver, B. A., Williams, R., Teuben, P., & Astropy Collaboration. (2023). specutils: A Python package for spectral analysis. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.8040075
*   *Summary:* This Zenodo record archives a version of `specutils`, the Astropy-affiliated package central to representing and manipulating spectroscopic data (Section 4.2) and performing tasks like wavelength calibration (Section 4.5) as demonstrated in the examples.

Dalton, G., Trager, S., Abrams, D. C., Bonifacio, P., Aguerri, J. A. L., Alpaslan, M., Balcells, M., Barker, R., Battaglia, G., Bellido-Tirado, O., Benson, A., Best, P., Bland-Hawthorn, J., Bridges, T., Brinkmann, J., Brusa, M., Cabral, J., Caffau, E., Carter, D., … Zurita, C. (2022). 4MOST: Project overview and information for the First Call for Proposals. *The Messenger, 186*, 3–11. https://doi.org/10.18727/0722-6691/5267
*   *Summary:* Describes the 4MOST multi-object fiber spectrograph facility. It exemplifies the type of MOS instrument discussed in Section 4.1, highlighting the need for automated tracing and extraction pipelines for large numbers of fibers, including sky fibers (Section 4.7).

Fischer, D. A., Anglada-Escude, G., Arriagada, P., Baluev, R. V., Bean, J. L., Bouchy, F., Buchhave, L. A., Carroll, T., Chakraborty, A., Crepp, J. R., Dawson, R. I., Diddams, S. A., Dumusque, X., Eastman, J. D., Endl, M., Figueira, P., Ford, E. B., Foreman-Mackey, D., Fournier, P., … Wright, J. T. (2016). State of the field: Extreme precision radial velocities. *Publications of the Astronomical Society of the Pacific, 128*(964), 066001. https://doi.org/10.1088/1538-3873/128/964/066001 *(Note: Pre-2020, but landmark review on relevant high-res techniques)*
*   *Summary:* Reviews instrumentation and techniques for extreme precision radial velocities, often relying on high-resolution echelle spectrographs. Discusses the critical need for hyper-stable wavelength calibration (Section 4.5) often involving simultaneous or bracketed arc lamps or laser frequency combs.

Haynes, D., Ellis, S., Bryant, J. J., Bland-Hawthorn, J., Case, S., Content, R., Farrell, T., Gers, L., Goodwin, M., Konstantopoulos, I. S., Lawrence, J., Pai, N., Patterson, R., Roth, M. M., Sharma, S., Shortridge, K., Trowland, H., Vuong, M., Xavier, P., & Zheng, J. (2023). Hector – A new multi-object integral field spectrograph for the Anglo-Australian Telescope. *Publications of the Astronomical Society of Australia, 40*, e034. https://doi.org/10.1017/pasa.2023.31
*   *Summary:* Describes the Hector instrument, a modern example of a fiber-based IFU system. This provides a concrete example of the IFU technology mentioned in Section 4.1 and whose complex data requires sophisticated tracing, extraction (Section 4.4), and flat-fielding (Section 4.6).

Hinkle, K. H., Stauffer, J. R., Plavchan, P. P., & Wallace, L. (2021). Infrared Astronomical Spectroscopy with High Spectral Resolution. *Publications of the Astronomical Society of the Pacific, 133*(1027), 092001. https://doi.org/10.1088/1538-3873/ac1a3a
*   *Summary:* This review focuses on high-resolution infrared spectroscopy, often employing echelle spectrographs (Section 4.1). It discusses challenges specific to IR data reduction, including sky subtraction (Section 4.7) and wavelength calibration (Section 4.5) in this regime using sky lines or lamps.

Kerber, F., Nave, G., & Sansonetti, C. J. (2008). The spectrum of Th–Ar hollow cathode lamps in the 900–4500 nm region: establishing wavelength standards for the calibration of VLT spectrographs. *The Astrophysical Journal Supplement Series, 178*(2), 374–393. https://doi.org/10.1086/589708 *(Note: Pre-2020, but key reference for ThAr line lists)*
*   *Summary:* Provides an extensive line list for ThAr lamps in the near-infrared, crucial for the wavelength calibration (Section 4.5) of many astronomical spectrographs operating in this range. Represents the type of laboratory data needed for calibration.

Maraston, C., Goddard, D., Thomas, D., Parikh, T., Li, C., Breda, I., & Riffel, R. A. (2020). The Ninth Data Release of the Sloan Digital Sky Survey: First Spectroscopic Data from the SDSS-III Apache Point Observatory Galactic Evolution Experiment. *Monthly Notices of the Royal Astronomical Society, 496*(3), 3028–3040. https://doi.org/10.1093/mnras/staa1650 *(Note: Paper refers to BOSS/APOGEE reduction, relevant methods)*
*   *Summary:* While describing SDSS-III/APOGEE data, the paper implicitly covers reduction techniques relevant to large surveys, including reliance on sky lines for NIR wavelength calibration (Section 4.5.1), a key technique discussed in Section 4.5.

Murphy, M. T., Tzanavaris, P., Webb, J. K., & Lovis, C. (2007). Selection of ThAr lines for wavelength calibration of echelle spectra and implications for variations in the fine-structure constant. *Monthly Notices of the Royal Astronomical Society, 378*(1), 221–230. https://doi.org/10.1111/j.1365-2966.2007.11768.x *(Note: Pre-2020, but key paper on precise wavelength calibration)*
*   *Summary:* Focuses on selecting optimal ThAr lines for achieving high-precision wavelength calibration in echelle spectra, critical for fundamental physics measurements and relevant to the discussion of arc lamp calibration (Section 4.5).

Noll, S., Kausch, W., Barden, M., Jones, A. M., Szyszka, C., Kimeswenger, S., & Vinther, J. (2014). An atmospheric radiation model for Cerro Paranal. II. Emission model. *Astronomy & Astrophysics, 567*, A25. https://doi.org/10.1051/0004-6361/201423616 *(Note: Pre-2020, models sky emission relevant to subtraction)*
*   *Summary:* Describes a detailed model for atmospheric emission lines at Cerro Paranal. While pre-2020, it provides context for the complexity of the sky spectrum (Section 4.7) that needs to be subtracted and highlights advanced modeling approaches.

Prochaska, J. X., Hennawi, J. F., Westfall, K. B., Cooke, R., Wang, F., Hsyu, T., & Emg, D. (2020). PypeIt: The Python Spectroscopic Data Reduction Pipeline. *Journal of Open Source Software, 5*(54), 2308. https://doi.org/10.21105/joss.02308
*   *Summary:* This paper introduces PypeIt, a widely used Python-based spectroscopic data reduction pipeline designed to handle data from many different spectrographs. It implements many of the algorithms discussed in this chapter (tracing, extraction, wavelength calibration, sky subtraction - Sections 4.3-4.7).

Weilbacher, P. M., Palsa, R., Streicher, O., Conseil, S., Bacon, R., Boogaard, L., Borisova, E., Brinchmann, J., Contini, T., Feltre, A., Guérou, A., Kollatschny, W., Krajnović, D., Maseda, M. V., Paalvast, M., Pécontal-Rousset, A., Pello, R., Richard, J., Roth, M. M., … Wisotzki, L. (2020). The MUSE data reduction pipeline. *Astronomy & Astrophysics, 641*, A28. https://doi.org/10.1051/0004-6361/202037985
*   *Summary:* Describes the data reduction pipeline for the MUSE instrument, a prominent IFU spectrograph (Section 4.1). It details the specific challenges and solutions implemented for reducing complex IFU data, including tracing, extraction (Section 4.4), flat-fielding (Section 4.6) and sky subtraction (Section 4.7).

