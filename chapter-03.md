---

# Chapter 3

# Basic Image Reduction: Instrument Signature Removal

---

![imagem](imagem.png)
*This chapter addresses the essential procedures for basic image reduction, focusing predominantly on data acquired with Charge-Coupled Device (CCD) or similar array detectors commonly employed in optical and near-infrared astronomy. The primary objective of this processing stage is the systematic removal or mitigation of instrumental artifacts and signatures inherent in the raw data, thereby transforming the detector's output into a representation that more accurately reflects the incident photon flux from celestial sources. The discussion commences with a detailed examination of the various instrumental effects that contaminate raw images, elucidating their physical origins within the detector and readout electronics, including bias structures, dark current generation, read noise, pixel sensitivity variations, non-linearities, saturation effects, cosmetic defects like bad pixels, and the impact of transient events such as cosmic rays. Subsequently, the chapter outlines standard methodologies for representing image data computationally, emphasizing the importance of propagating uncertainties and maintaining data quality information using structures like NumPy arrays and the specialized `astropy.nddata.CCDData` object. Core reduction steps are then meticulously described: bias level subtraction, accounting for the baseline electronic offset; dark current correction, addressing thermally generated signal; and flat-field correction, designed to compensate for pixel-to-pixel quantum efficiency differences and large-scale illumination non-uniformities. The construction of high-quality master calibration frames (bias, dark, flat) through statistical combination of multiple individual exposures is detailed, alongside techniques for identifying and masking defective pixels. Methods for detecting and removing artifacts caused by cosmic ray impacts are also presented. Finally, the chapter synthesizes these individual steps into a cohesive practical workflow, illustrating the typical sequence of operations within an image reduction script and highlighting the utility of dedicated software packages like `ccdproc` for streamlining this process.*

---

**3.1 Understanding Instrumental Signatures**

Raw data frames obtained directly from astronomical array detectors, such as CCDs or CMOS sensors, are inevitably imprinted with a variety of instrumental signatures originating from the detector physics, readout electronics, and observing environment. These artifacts obscure the true astrophysical signal and must be carefully characterized and removed or mitigated during the basic data reduction process to enable accurate scientific analysis (Lesser, 2015; Janesick, 2001). A thorough understanding of the nature and origin of these signatures is fundamental for selecting appropriate correction strategies and assessing the quality of the reduced data.

*   **Bias Level and Structure:** Even in the complete absence of light and with zero exposure time, reading out a detector yields a non-zero signal known as the bias level or zero point. This represents a baseline electronic offset added during the charge-to-voltage conversion and subsequent amplification and digitization stages. It is primarily determined by the stable DC voltage levels within the readout circuitry. While ideally constant across the detector, the bias often exhibits some spatial structure. This can include a mean offset, potentially subtle row-dependent or column-dependent variations (banding), low-frequency gradients, or even fixed patterns related to the amplifier or ADC characteristics. Furthermore, the bias level can exhibit slight variations over time due to temperature fluctuations or electronic drifts. The primary goal of bias subtraction (Section 3.3) is to remove this baseline offset. The **overscan region**, a set of virtual pixels clocked out after the physical pixels without accumulating charge from the photosensitive area, is often used to estimate the bias level on a frame-by-frame basis, potentially capturing short-term variations. However, using dedicated zero-exposure bias frames averaged into a master bias is often preferred for characterizing stable spatial structures.

*   **Read Noise ($\sigma_{read}$):** The readout process itself introduces random electronic noise, known as read noise. This fundamental noise source arises from stochastic processes within the output amplifier (e.g., thermal Johnson noise, shot noise in transistor currents) and the digitization process (quantization noise). It represents the irreducible uncertainty associated with measuring the charge in a pixel during a single readout event and is typically characterized by its root-mean-square (RMS) value in electrons (e⁻). Read noise is independent of the signal level (until saturation effects dominate) and independent of exposure time. Its magnitude depends on the detector design, operating temperature, and readout speed (faster readout generally leads to higher read noise). Minimizing read noise is a key goal in scientific detector design, especially for observations of faint sources where the signal may be comparable to or less than the read noise level. Its value, usually provided by the observatory or measured during characterization, is essential for calculating the total noise budget and uncertainty associated with each pixel measurement (see Section 3.2).

*   **Dark Current ($I_{dark}$):** Even in complete darkness, thermal energy within the silicon lattice can generate electron-hole pairs, mimicking the photoelectric effect. These thermally generated electrons accumulate in the pixel potential wells during an exposure, contributing a signal known as dark current (Janesick, 2001). The rate of dark current generation is strongly dependent on temperature, approximately doubling for every 5-7°C increase in silicon. It also varies significantly from pixel to pixel due to localized impurities or defects in the silicon lattice, leading to a "dark fixed pattern noise." While often small in cryogenically cooled scientific detectors, dark current can become significant for long exposures or at higher operating temperatures. It accumulates linearly with exposure time ($t_{exp}$). The total dark signal ($S_{dark}$) in electrons is approximately $S_{dark}=I_{dark} \times t_{exp}$, where $I_{dark}$ is the dark current rate (e.g., in e⁻/pixel/second). Dark current subtraction (Section 3.4) aims to remove this thermally generated signal, typically using dedicated "dark frames" acquired with the same exposure time and temperature as the science frames but with the shutter closed. Poisson noise associated with the dark current itself ($\sqrt{S_{dark}}$) contributes to the overall noise budget.

*   **Pixel Response Non-Uniformity (PRNU) and Flat-Field Variations:** Pixels across a detector do not respond identically to the same amount of incident light. **Pixel Response Non-Uniformity (PRNU)** refers to the intrinsic pixel-to-pixel variations in sensitivity (Quantum Efficiency, QE), primarily caused by minute differences in pixel size, gate structure, or material properties resulting from the manufacturing process. These typically manifest as high-spatial-frequency variations. Additionally, larger-scale variations in illumination reaching the detector plane contribute to the overall flat-field structure. **Vignetting**, a gradual decrease in illumination towards the edges of the field of view, is caused by the telescope optics obscuring off-axis light rays. Shadows cast by **dust particles** on filters or detector windows create localized "donut" or spot-like artifacts. Interference fringes (**fringing**) can arise from multiple reflections within the thinned silicon substrate, particularly problematic at near-infrared wavelengths where silicon becomes partially transparent. The combined effect of PRNU and large-scale illumination patterns necessitates **flat-field correction** (Section 3.5), which aims to normalize the response of all pixels to unity by dividing the science image by a normalized image of a uniformly illuminated source. Since QE is wavelength-dependent, flat-fields must be obtained using the same filter as the science observations.

*   **Bad Pixels and Cosmetic Defects:** Manufacturing imperfections inevitably lead to some pixels exhibiting anomalous behavior. **Dead pixels** have very low or zero sensitivity. **Hot pixels** exhibit excessively high dark current, appearing bright even in short dark exposures. **Flickering pixels** show unstable or fluctuating dark current or sensitivity. **Charge traps** are defects that can capture and later release electrons, potentially affecting charge transfer efficiency (in CCDs) or causing persistence effects. Clusters of bad pixels or entire bad columns/rows can also occur. Identifying these defects and creating a **bad pixel mask (BPM)** (Section 3.7) is crucial, allowing affected pixels to be ignored or interpolated during subsequent processing and analysis.

*   **Saturation and Blooming:** Each pixel has a finite **full well capacity**, the maximum number of electrons it can hold. When the incident flux is high enough to fill the potential well during an exposure, the pixel **saturates**. Beyond this point, the pixel response becomes highly non-linear, and the measured signal no longer accurately reflects the incident flux. In many CCD designs, excess charge above the full well capacity can spill into adjacent pixels along the column direction during charge transfer, an effect known as **blooming**, creating bright streaks emanating from saturated sources. Saturation and blooming render the affected pixels (and potentially their neighbors) scientifically unusable for quantitative analysis and must be flagged (often using the BPM).

*   **Non-Linearity:** Ideally, the relationship between the number of collected electrons ($N_e$) and the output ADU value should be perfectly linear up to saturation. In practice, slight deviations from linearity can occur, particularly at high signal levels approaching full well capacity. This **non-linearity** means the detector gain ($e^-/\mathrm{ADU}$) is not strictly constant. For high-precision photometry, non-linearity must be characterized (typically using a series of flat-field exposures with varying illumination levels) and corrected, often by applying a polynomial correction function to the measured ADU values. Modern sensor controllers sometimes perform linearity corrections internally (Rauscher, 2021).

*   **Cosmic Rays:** High-energy charged particles from space (cosmic rays) constantly bombard the Earth and space-based observatories. When these particles pass through a silicon detector, they deposit energy via ionization, creating localized trails or spots of charge that are indistinguishable from photon-generated signal in a single exposure. These events appear as sharp, often elongated or multi-pixel, bright features in astronomical images. Their occurrence rate depends on altitude, latitude, shielding, and detector thickness, but typically ranges from ~1 to a few events per cm² per minute. Since cosmic rays are transient events affecting random pixels in each exposure, they can be identified and removed (Section 3.8) by comparing multiple dithered exposures of the same field or by using algorithms that detect their characteristic sharp profiles, distinct from the smoother Point Spread Function (PSF) of astronomical sources.

Careful characterization of these instrumental signatures, often performed by observatory staff and made available through calibration databases or pipeline documentation (e.g., Greenhouse et al., 2023 for JWST), is essential for applying the correct reduction procedures described in the following sections.

**3.2 Representing Images: NumPy Arrays and `astropy.nddata`**

Computationally, astronomical images obtained from array detectors are fundamentally represented as multi-dimensional arrays of numerical values. The **NumPy** library, providing the `ndarray` object, is the cornerstone for handling such array data in Python (Harris et al., 2020). A raw or reduced image is typically stored as a 2D NumPy array where each element corresponds to a pixel and holds the measured ADU value or a calibrated physical quantity (e.g., electrons, flux). NumPy's strength lies in its efficient element-wise operations (vectorization), mathematical functions, slicing, and indexing capabilities, which are essential for performing image arithmetic during reduction (e.g., subtracting a bias array, dividing by a flat-field array).

However, scientific data analysis requires more than just the primary data values. A proper representation must also incorporate information about the **uncertainty** associated with each measurement and the **quality** or validity of each pixel. Simply storing the image data in a raw NumPy array loses this crucial context. Propagating uncertainties correctly through the reduction process is vital for assessing the statistical significance of final measurements. Likewise, tracking bad pixels or saturated regions is necessary to avoid incorporating faulty data into the analysis. The calculation of uncertainty typically considers contributions from various noise sources. For data in ADU, the variance ($\sigma^2$) in a pixel can often be approximated as $\sigma^2_{ADU} \approx (\sigma_{read}/g)^2 + S_{total}/g$, where $\sigma_{read}$ is the read noise in electrons, $g$ is the gain in $e^-/\mathrm{ADU}$, and $S_{total}$ is the total signal in ADU (source + sky + dark). More accurately, using signals in electrons ($N_e$), the variance is $\sigma^2_{e^-} \approx \sigma_{read}^2 + N_{e,total}$, reflecting Poisson noise from the total detected electrons plus the read noise variance.

The **`astropy.nddata`** sub-package provides a framework for encapsulating n-dimensional astronomical data along with its associated metadata, uncertainty, and masks (Astropy Collaboration et al., 2022). The central class for optical/IR imaging data is **`astropy.nddata.CCDData`**. A `CCDData` object acts as a container holding several key attributes:
*   `.data`: A NumPy array containing the primary pixel data (e.g., ADUs, electrons, or flux-calibrated values).
*   `.uncertainty`: An object representing the uncertainty associated with each pixel in the `.data` array. This is commonly an `astropy.nddata.StdDevUncertainty` object holding a NumPy array of pixel standard deviations, or an `astropy.nddata.VarianceUncertainty` holding pixel variances. Correctly populating and propagating this uncertainty is critical. For raw data, the uncertainty often includes contributions from read noise and Poisson (shot) noise from the source signal and dark current: $\sigma_{total}^2 = \sigma_{read}^2 + S_{source} + S_{dark} + S_{sky}$, where signals $S$ are typically calculated in electrons. During reduction, uncertainty propagation rules (standard error propagation for addition, subtraction, multiplication, division) must be applied.
*   `.mask`: A boolean NumPy array of the same shape as `.data`, where `True` indicates a pixel that should be ignored or considered invalid (e.g., bad pixels, saturated pixels, cosmic rays). This mask is respected by many analysis functions.
*   `.meta`: A dictionary-like object (typically `fits.Header`) holding the metadata associated with the data (e.g., FITS header keywords).
*   `.unit`: An `astropy.units.Unit` object specifying the physical unit of the values in the `.data` array (e.g., `u.adu`, `u.electron`, `u.Jy`).
*   `.wcs`: An `astropy.wcs.WCS` object describing the World Coordinate System transformation.

Using `CCDData` (or the more general `NDData`) throughout the reduction pipeline offers significant advantages. Arithmetic operations (addition, subtraction, multiplication, division) between `CCDData` objects automatically handle uncertainty propagation (if uncertainties are correctly defined and correlated errors are negligible) and combine masks appropriately (e.g., a pixel masked in either input becomes masked in the output). Units are also checked for compatibility, preventing dimensionally inconsistent operations. Packages like `ccdproc` (Section 3.9) are designed to operate directly on `CCDData` objects, reading necessary information (like gain or read noise stored in the `.meta` attribute) and updating the `.data`, `.uncertainty`, and `.mask` attributes correctly after each processing step. This structured approach greatly enhances the robustness and traceability of the data reduction process compared to manually managing separate NumPy arrays for data, uncertainty, and masks. For example, subtracting a master bias `CCDData` object (with its own uncertainty, primarily from read noise in the individual bias frames) from a raw science `CCDData` object will correctly compute the resulting uncertainty in the bias-subtracted frame using standard error propagation for subtraction: $\sigma_{result}^2 = \sigma_{science}^2 + \sigma_{bias}^2$. Similarly, flat-field division correctly propagates the uncertainties from both the science frame and the master flat frame: $(\sigma_{result}/S_{result})^2 \approx (\sigma_{science}/S_{science})^2 + (\sigma_{flat}/S_{flat})^2$.

**3.3 Bias Subtraction**

The first fundamental step in reducing data from CCDs and many CMOS/IR arrays is the removal of the bias level, the electronic offset present even in zero-length exposures (Section 3.1). This offset must be subtracted to establish a correct zero point for the subsequent measurement of photon-generated signal and dark current. Failure to remove the bias accurately will propagate systematic errors throughout the entire reduction process. The subtraction is typically performed using either information from overscan regions or dedicated bias frames.

**Overscan Correction:** Many scientific CCDs incorporate an **overscan region**. This consists of columns or rows adjacent to the photosensitive imaging area that are read out using the same electronics but do not collect photo-charge. The pixel values in the overscan region therefore represent only the electronic bias level (plus read noise) for that specific readout cycle. By calculating the median or clipped mean value within the overscan region(s) associated with a science frame, one can estimate the bias level for that particular exposure. This estimate can then be subtracted from the entire image (including the imaging area). Some reduction routines also fit a low-order polynomial (e.g., linear or quadratic) to the overscan region along one dimension (e.g., rows) and subtract this fit row-by-row or column-by-column to account for simple gradients in the bias structure captured by the overscan.
*   **Pros:** Captures short-term fluctuations in the bias level from one frame to the next. Does not require separate bias exposures. Can correct for simple spatial variations if fitted.
*   **Cons:** Assumes the bias level and structure in the overscan accurately reflects the level across the entire imaging area (may not hold if significant 2D structure exists). Adds the read noise from the overscan estimation to the science frame noise. Requires the detector to have well-defined and stable overscan regions. Does not correct complex 2D bias patterns.

The subtraction using a scalar overscan level $B_{ov}$ (in ADU) from science frame $S_{raw}$ is simply:
$S_{bias\_subtracted}(x, y) = S_{raw}(x, y) - B_{ov}$
If a fitted function $B_{ov}(x, y)$ (e.g., varying only with $y$) is used, the subtraction is pixel-wise.

**Master Bias Frame Correction:** The more common approach for scientific imaging involves acquiring a series of dedicated **bias frames** – exposures with the minimum possible integration time (effectively zero) taken with the shutter closed, typically acquired in batches during calibration periods. These frames capture the stable spatial structure of the bias across the detector, as well as the average bias level under specific operating conditions (temperature, readout mode). To minimize the contribution of read noise from the bias correction step itself, multiple (typically 10-20 or more) individual bias frames ($B_i$) are statistically combined to create a low-noise **master bias frame** ($\bar{B}$). The most robust combination method is usually the median, as it effectively rejects outliers (e.g., cosmic rays hitting during the short bias readouts). Alternatively, a sigma-clipped average can be used.

$\bar{B}(x, y) = \mathrm{median} \{ B_i(x, y) \}$ or $\bar{B}(x, y) = \mathrm{mean}_{\sigma-clip} \{ B_i(x, y) \}$

where $(x, y)$ denotes the pixel coordinates. This master bias frame, representing the average, stable bias pattern, is then subtracted pixel-by-pixel from each science frame ($S_{raw}$) and other calibration frames (darks, flats).

$S_{bias\_subtracted}(x, y) = S_{raw}(x, y) - \bar{B}(x, y)$

*   **Pros:** Accurately removes stable spatial structure (2D patterns) in the bias. Reduces the noise introduced by the bias subtraction step (noise in $\bar{B}$ is $\approx \sigma_{read} / \sqrt{N_{bias}}$ for average combination, or slightly higher for median, where $N_{bias}$ is the number of combined frames).
*   **Cons:** Does not account for short-term temporal variations in the overall bias level (often addressed by also subtracting a scalar overscan level if available *after* master bias subtraction). Requires dedicated calibration time. Assumes bias structure is stable between acquisition of bias frames and science frames (usually true if temperature and settings are stable).

For highest precision, a combination is sometimes used: subtract the master bias frame to remove structure, then subtract the scalar median overscan level from the *same frame* to correct for any temporal offset relative to the master bias epoch. Regardless of the method, accurate bias subtraction is paramount. The resulting bias-subtracted image should have pixel values fluctuating around zero (in regions with no signal or dark current), dominated by read noise and photon/dark current shot noise. Proper uncertainty propagation during subtraction is essential: $\sigma_{subtracted}^2 = \sigma_{raw}^2 + \sigma_{master\_bias}^2$.

The following Python code exemplifies the creation of a master bias frame from multiple individual bias frames and its subsequent subtraction from a science image. This process utilizes the `ccdproc` library, which simplifies the combination and subtraction operations while handling `CCDData` objects, including metadata and potential uncertainty propagation. The combination uses a median algorithm, known for its robustness against outlier pixel values, such as those potentially caused by cosmic rays hitting during the short bias exposures. Sigma clipping is optionally applied before the median combination for further outlier rejection. The resulting master bias represents the best estimate of the stable electronic baseline of the detector.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
# ccdproc provides tools for CCD data reduction steps
# Ensure installed: pip install ccdproc
import ccdproc
import os # For dummy file creation/check

# --- Assume we have a list of bias FITS file paths ---
# Replace with actual paths to your bias frames
bias_files = [f'bias_{i+1:03d}.fits' for i in range(15)] # Example list of 15 files
# --- Assume we have a raw science FITS file path ---
science_file = 'science_raw.fits'
# --- Output file path for the master bias and the processed science frame ---
master_bias_file = 'master_bias.fits'
output_bias_subtracted_file = 'science_bias_sub.fits'

# Create dummy files for demonstration if they don't exist
for fname in bias_files + [science_file]:
    if not os.path.exists(fname):
        print(f"File {fname} not found, creating dummy file.")
        # Simulate bias structure + noise
        bias_level = 500
        bias_structure = np.random.normal(0, 2, size=(100, 100)) * (np.arange(100)/100.0)[:, np.newaxis] # Simple gradient
        read_noise_amp = 5
        data = bias_level + bias_structure + np.random.normal(0, read_noise_amp, size=(100, 100))
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        # Add essential keywords if it's a science frame for later steps
        if fname == science_file:
             hdu.header['OBSTYPE'] = 'SCIENCE'
             hdu.header['EXPTIME'] = 120.0
        else:
             hdu.header['OBSTYPE'] = 'BIAS'
             hdu.header['EXPTIME'] = 0.0
        hdu.writeto(fname, overwrite=True)

try:
    # --- Create Master Bias Frame ---
    print(f"Attempting to create master bias from {len(bias_files)} files...")
    # Read bias frames into a list of CCDData objects.
    # It's crucial to specify the unit of the data (e.g., 'adu') for potential
    # future operations involving physical units (like gain correction).
    bias_ccds = []
    for f in bias_files:
        try:
            ccd = CCDData.read(f, unit='adu')
            bias_ccds.append(ccd)
        except FileNotFoundError:
            print(f"Warning: Bias file {f} not found. Skipping.")
        except Exception as read_err:
            print(f"Warning: Error reading bias file {f}: {read_err}. Skipping.")

    if not bias_ccds:
        raise ValueError("No valid bias frames were loaded.")

    # Use ccdproc.Combiner to manage the combination process.
    combiner = ccdproc.Combiner(bias_ccds)

    # Apply sigma clipping before combining to reject outliers (e.g., cosmic rays).
    # low_thresh/high_thresh: Number of standard deviations for clipping.
    # func: Function to calculate the central value (median is robust).
    # dev_func: Function to calculate the deviation (standard deviation).
    combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median, dev_func=np.ma.std)

    # Combine the clipped frames using the median algorithm.
    master_bias = combiner.median_combine()
    # Store the number of combined frames in the master bias metadata.
    master_bias.meta['NCOMBINE'] = len(bias_ccds) # Use actual number loaded
    print(f"Master bias created by median combining {len(bias_ccds)} frames.")

    # Save the resulting master bias frame to a FITS file.
    master_bias.write(master_bias_file, overwrite=True)
    print(f"Master bias saved to {master_bias_file}")

    # --- Subtract Master Bias from Science Frame ---
    # Read the raw science frame, also specifying units.
    print(f"Reading raw science frame: {science_file}")
    science_ccd = CCDData.read(science_file, unit='adu')

    # Perform bias subtraction using ccdproc.subtract_bias.
    # This function conveniently handles CCDData objects. If the input CCDData
    # objects have .uncertainty attributes defined, subtract_bias will also
    # propagate the uncertainties according to standard error propagation rules.
    # It requires that master_bias and science_ccd have compatible shapes.
    print("Subtracting master bias from science frame...")
    science_bias_subtracted = ccdproc.subtract_bias(science_ccd, master_bias)

    # Add a record of the processing step to the FITS header history.
    # This is crucial for data provenance.
    science_bias_subtracted.meta['HISTORY'] = f'Bias subtracted using {os.path.basename(master_bias_file)}'

    # Save the bias-subtracted science frame to a new FITS file.
    science_bias_subtracted.write(output_bias_subtracted_file, overwrite=True)
    print(f"Bias-subtracted science frame saved to {output_bias_subtracted_file}")
    print(f"Mean value after bias subtraction: {np.mean(science_bias_subtracted.data):.2f} ADU")


except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
except CCDDataProcessingError as e:
    # Catch errors specific to ccdproc operations
    print(f"Error during CCD processing: {e}")
except ValueError as e:
    # Catch errors like no valid files loaded
    print(f"Value Error: {e}")
except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred during bias correction: {e}")

```

The execution of this Python script demonstrates the standard procedure for bias correction in astronomical image processing, leveraging the `ccdproc` library for efficiency and robustness. Initially, it reads a collection of individual bias frames (zero-second exposures) into `CCDData` objects, ensuring units are specified. The `ccdproc.Combiner` class is then employed to manage the combination process; sigma clipping is applied to reject outlier pixels, and the `median_combine` method is used to generate a low-noise master bias frame, effectively capturing the stable electronic offset and its spatial structure. This master bias is saved for later use and documentation. Subsequently, the script reads the raw science frame, again as a `CCDData` object, and utilizes the `ccdproc.subtract_bias` function. This function performs the pixel-wise subtraction of the master bias from the science frame, automatically handling uncertainty propagation if uncertainties were defined in the input objects. A `HISTORY` record is added to the output FITS header, documenting the applied processing step, and the final bias-subtracted science frame is saved to disk, ready for subsequent reduction steps like dark subtraction and flat-fielding.

**3.4 Dark Current Correction**

Following bias subtraction, the next instrumental effect to address is dark current, the signal generated thermally within pixels during the exposure duration (Section 3.1). This signal adds to the astronomical flux and associated noise, and its removal is crucial, particularly for observations involving long integration times or conducted with detectors that are not sufficiently cooled, leading to significant thermal electron generation. The standard correction involves subtracting a master dark frame, created from dedicated exposures taken in darkness but with the same exposure time and operating temperature as the science frames. This master dark characterizes the average spatial pattern and magnitude of the dark signal accumulated during that specific exposure time.

The Python code below implements the creation of a master dark frame and its application to a bias-subtracted science frame. It assumes that individual dark frames (matching the science exposure time) and a master bias frame have already been generated or are available. Each individual dark frame is first bias-subtracted. Then, these processed darks are combined, typically using a median or sigma-clipped average via `ccdproc.Combiner`, to produce the final master dark frame, reducing noise and rejecting outliers like cosmic hits during the dark exposures. Finally, the master dark is subtracted from the bias-subtracted science frame using `ccdproc.subtract_dark`. This function can also handle cases where the science and master dark exposure times differ by appropriately scaling the master dark, although obtaining matching darks is strongly preferred for accuracy.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
import ccdproc
import astropy.units as u
import os # For dummy file creation/check

# --- Assume we have a list of dark FITS file paths (matching science exposure) ---
science_exp_time = 120.0 # seconds
dark_files = [f'dark_{i+1:03d}_{int(science_exp_time)}s.fits' for i in range(10)] # Example list for 120s darks
# --- Assume master bias file exists ---
master_bias_file = 'master_bias.fits'
# --- Assume bias-subtracted science file exists ---
bias_subtracted_science_file = 'science_bias_sub.fits'
# --- Output file paths ---
master_dark_file = f'master_dark_{int(science_exp_time)}s.fits'
output_dark_subtracted_file = 'science_dark_sub.fits'

# Create dummy files for demonstration if they don't exist
# Create master bias if missing
if not os.path.exists(master_bias_file):
    print(f"File {master_bias_file} not found, creating dummy file.")
    master_bias_data = np.ones((100, 100)) * 500
    fits.PrimaryHDU(master_bias_data.astype(np.float32)).writeto(master_bias_file, overwrite=True)
# Create bias-subtracted science file if missing
if not os.path.exists(bias_subtracted_science_file):
    print(f"File {bias_subtracted_science_file} not found, creating dummy file.")
    # Simulate sky + source after bias subtraction
    science_data = np.random.normal(20, 5, size=(100, 100)) # Background around 0 + noise
    # Add a fake source
    yy, xx = np.indices((100, 100))
    science_data += 50 * np.exp(-0.5 * (((xx - 50)/10)**2 + ((yy - 50)/10)**2))
    hdr = fits.Header({'EXPTIME': science_exp_time})
    fits.PrimaryHDU(science_data.astype(np.float32), header=hdr).writeto(bias_subtracted_science_file, overwrite=True)
# Create dark files if missing
for fname in dark_files:
    if not os.path.exists(fname):
        print(f"File {fname} not found, creating dummy file.")
        # Simulate bias + dark signal + noise
        bias_level = 500
        dark_current_rate = 0.1 # ADU/s
        dark_signal = dark_current_rate * science_exp_time
        read_noise_amp = 5
        # Add some hot pixels pattern
        hot_pix_pattern = np.zeros((100, 100))
        hot_pix_pattern[np.random.randint(0, 100, 10), np.random.randint(0, 100, 10)] = dark_signal * 5
        data = (bias_level + dark_signal + hot_pix_pattern +
                np.random.normal(0, read_noise_amp, size=(100, 100)))
        hdr = fits.Header({'EXPTIME': science_exp_time})
        fits.PrimaryHDU(data.astype(np.float32), header=hdr).writeto(fname, overwrite=True)

try:
    # --- Load Master Bias ---
    print(f"Loading master bias from {master_bias_file}")
    # Ensure unit is specified if bias values are in ADU
    master_bias = CCDData.read(master_bias_file, unit='adu')

    # --- Create Master Dark ---
    print(f"Attempting to create master dark from {len(dark_files)} files...")
    dark_frames_bias_sub = []
    loaded_dark_exposure = None # To store exposure time from first valid dark
    for f in dark_files:
        try:
            dark_ccd = CCDData.read(f, unit='adu')
            # Check exposure time consistency (important!)
            current_exp_time = dark_ccd.header.get('EXPTIME', -1)
            if loaded_dark_exposure is None: # First valid dark frame
                loaded_dark_exposure = current_exp_time
                if not np.isclose(loaded_dark_exposure, science_exp_time):
                     print(f"Warning: Dark exposure time {loaded_dark_exposure}s differs from science exposure {science_exp_time}s. Scaling will be required if used directly.")
            elif not np.isclose(current_exp_time, loaded_dark_exposure):
                 print(f"Warning: Exposure time mismatch in dark file {f} ({current_exp_time}s vs {loaded_dark_exposure}s). Skipping.")
                 continue

            # Subtract bias from the individual dark frame
            dark_bias_sub = ccdproc.subtract_bias(dark_ccd, master_bias)
            dark_frames_bias_sub.append(dark_bias_sub)
        except FileNotFoundError:
            print(f"Warning: Dark file {f} not found. Skipping.")
        except KeyError:
            print(f"Warning: EXPTIME keyword missing in {f}. Skipping.")
        except Exception as read_err:
            print(f"Warning: Error reading/processing dark file {f}: {read_err}. Skipping.")


    if not dark_frames_bias_sub:
        raise ValueError("No valid dark frames were loaded or processed.")

    # Combine the bias-subtracted dark frames using median combine
    print(f"Combining {len(dark_frames_bias_sub)} bias-subtracted dark frames...")
    dark_combiner = ccdproc.Combiner(dark_frames_bias_sub)
    # Optional: Apply sigma clipping to reject cosmic rays in darks
    dark_combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median, dev_func=np.ma.std)
    master_dark = dark_combiner.median_combine()
    master_dark.meta['NCOMBINE'] = len(dark_frames_bias_sub)
    # Store the exposure time for which this master dark is valid
    master_dark.meta['EXPTIME'] = loaded_dark_exposure

    # Save the master dark frame
    master_dark.write(master_dark_file, overwrite=True)
    print(f"Master dark saved to {master_dark_file}")

    # --- Subtract Master Dark from Science Frame ---
    # Read the bias-subtracted science frame
    print(f"Reading bias-subtracted science frame: {bias_subtracted_science_file}")
    science_bias_sub = CCDData.read(bias_subtracted_science_file) # Assume units are handled correctly if saved previously

    # Retrieve science exposure time accurately from its header
    science_exposure_time_hdr = science_bias_sub.header.get('EXPTIME', None)
    if science_exposure_time_hdr is None:
        raise ValueError("EXPTIME keyword missing from science frame header.")
    data_exposure_time = science_exposure_time_hdr * u.s
    dark_exposure_time = loaded_dark_exposure * u.s # Use time derived from loaded darks

    # Determine if scaling is needed based on header values
    scale_dark_needed = not np.isclose(data_exposure_time.value, dark_exposure_time.value)
    if scale_dark_needed:
         print(f"Note: Science exposure {data_exposure_time} differs from master dark exposure {dark_exposure_time}. Scaling dark frame.")


    # Perform dark subtraction using ccdproc.subtract_dark
    # Provide exposure times and units explicitly.
    # Set 'scale=True' only if exposure times differ significantly.
    print("Subtracting master dark from science frame...")
    science_dark_subtracted = ccdproc.subtract_dark(
        ccd=science_bias_sub,
        master=master_dark,
        dark_exposure=dark_exposure_time, # Exposure time of the master dark itself
        data_exposure=data_exposure_time, # Exposure time of the science frame to correct
        exposure_unit=u.s,                # Unit for the exposure times
        scale=scale_dark_needed           # Enable scaling if times differ
    )

    # Add history entry
    science_dark_subtracted.meta['HISTORY'] = f'Dark subtracted using {os.path.basename(master_dark_file)} (Scaled: {scale_dark_needed})'

    # Save the dark-subtracted science frame
    science_dark_subtracted.write(output_dark_subtracted_file, overwrite=True)
    print(f"Dark-subtracted science frame saved to {output_dark_subtracted_file}")
    print(f"Mean value after dark subtraction: {np.mean(science_dark_subtracted.data):.2f}")


except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
except CCDDataProcessingError as e:
    print(f"Error during CCD processing: {e}")
except ValueError as e:
    print(f"Value Error: {e}")
except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred during dark correction: {e}")

```

The Python code above executes the crucial steps for dark current correction. It begins by loading a pre-computed master bias frame. Then, it iterates through a list of individual raw dark frames (ensuring they have the correct exposure time), reads each as a `CCDData` object, and subtracts the master bias from each one using `ccdproc.subtract_bias`. These bias-subtracted dark frames are collected and robustly combined using `ccdproc.Combiner` with median combination (and optional sigma clipping) to create the master dark frame, which represents the average thermal signal for the specified exposure time. This master dark is saved, and its valid exposure time is recorded. Finally, the script reads the bias-subtracted science frame, retrieves the exposure times for both the science frame and the master dark from their metadata, and performs the dark subtraction using `ccdproc.subtract_dark`. This function handles potential scaling if the exposure times differ (although using matching times is preferred) and updates the data and potentially uncertainty, saving the dark-subtracted result ready for flat-fielding.

**3.5 Flat-Field Correction**

Flat-field correction is arguably the most critical step for achieving accurate relative photometry across an image, correcting for variations in sensitivity from pixel to pixel (PRNU) as well as larger-scale throughput variations like vignetting or shadows from dust (Section 3.1). This correction requires images of a uniformly illuminated source, known as flat-field frames, taken through the same filter as the science observations. Common sources include illuminated dome screens, the twilight sky, or the night sky itself. Creating a high signal-to-noise, normalized master flat frame for each filter and dividing the science data by it effectively removes these instrumental response variations, ideally yielding an image where equal incident flux produces equal measured signal across the detector.

The following code demonstrates the creation of a master flat frame and its application. It assumes individual flat frames for a specific filter, along with master bias and an appropriate master dark frame (or knowledge that dark current is negligible for the flat exposure time), are available. Each raw flat is processed (bias and dark subtracted). These processed flats are then combined using `ccdproc.Combiner` to maximize signal-to-noise. A crucial step follows: the combined flat is normalized, typically by dividing by its median or mean value, to create the master flat with pixel values centered around 1.0. This normalized master flat is then divided into the bias- and dark-subtracted science frame using `ccdproc.flat_correct`.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
import ccdproc
import astropy.units as u
import os # For dummy file creation/check

# --- Assume list of flat field FITS files for a specific filter ---
filter_name = 'R'
flat_files = [f'flat_{i+1:03d}_filter{filter_name}.fits' for i in range(10)]
# --- Assume master bias and master dark files exist ---
master_bias_file = 'master_bias.fits'
# Use dark appropriate for flat exposure time (or assume negligible)
flat_exp_time = 5.0 # seconds
# Check if a dark for this exposure exists, otherwise maybe use 0 dark or scale
master_dark_file_for_flats = f'master_dark_{int(flat_exp_time)}s.fits'
# --- Assume dark-subtracted science file exists ---
dark_subtracted_science_file = 'science_dark_sub.fits' # Should be for same filter R
# --- Output file paths ---
master_flat_file = f'master_flat_filter{filter_name}.fits'
output_flat_corrected_file = 'science_final_redux.fits' # Final reduced frame

# Create dummy files for demonstration if they don't exist
# Create master bias if missing
if not os.path.exists(master_bias_file):
    print(f"File {master_bias_file} not found, creating dummy file.")
    fits.PrimaryHDU((np.ones((100, 100)) * 500).astype(np.float32)).writeto(master_bias_file, overwrite=True)
# Create master dark for flats if missing (simulate negligible dark for simplicity)
if not os.path.exists(master_dark_file_for_flats):
    print(f"File {master_dark_file_for_flats} not found, creating dummy file (zero dark).")
    hdr = fits.Header({'EXPTIME': flat_exp_time})
    fits.PrimaryHDU(np.zeros((100, 100)).astype(np.float32), header=hdr).writeto(master_dark_file_for_flats, overwrite=True)
# Create dark-subtracted science file if missing
if not os.path.exists(dark_subtracted_science_file):
     print(f"File {dark_subtracted_science_file} not found, creating dummy file.")
     hdr = fits.Header({'FILTER': filter_name, 'BUNIT':'electron'}) # Assume science is in electrons now
     fits.PrimaryHDU(np.random.normal(100, 10, size=(100, 100)).astype(np.float32), header=hdr).writeto(dark_subtracted_science_file, overwrite=True)
# Create flat files if missing
for fname in flat_files:
    if not os.path.exists(fname):
        print(f"File {fname} not found, creating dummy file.")
        # Simulate bias + flat signal + noise + vignetting
        bias_level = 500
        flat_level = 20000
        read_noise_amp = 5
        yy, xx = np.indices((100, 100))
        # Simple vignetting
        vignette = 1.0 - 0.3 * (((xx - 50)/50)**2 + ((yy - 50)/50)**2)
        # Pixel-to-pixel variation
        prnu = np.random.normal(1.0, 0.02, size=(100, 100))
        data = bias_level + (flat_level * vignette * prnu) + np.random.normal(0, read_noise_amp, size=(100, 100))
        hdr = fits.Header({'EXPTIME': flat_exp_time, 'FILTER': filter_name})
        fits.PrimaryHDU(data.astype(np.float32), header=hdr).writeto(fname, overwrite=True)

try:
    # --- Load Master Bias and Master Dark (appropriate for flats) ---
    print(f"Loading master bias from {master_bias_file}")
    master_bias = CCDData.read(master_bias_file, unit='adu') # Assume bias is ADU

    print(f"Loading master dark {master_dark_file_for_flats} (for flats)")
    master_dark_for_flats = CCDData.read(master_dark_file_for_flats) # Units should match subsequent data
    try:
        # Check if units are compatible, assume ADU if dark has no unit
        if master_dark_for_flats.unit is None: master_dark_for_flats.unit = 'adu'
        dark_for_flat_exposure = master_dark_for_flats.header['EXPTIME'] * u.s
    except KeyError:
        print(f"Warning: EXPTIME not found in {master_dark_file_for_flats} header. Cannot reliably scale dark.")
        # Handle error appropriately, e.g., assume it matches flat exp time if negligible dark expected
        dark_for_flat_exposure = flat_exp_time * u.s # Risky assumption

    # Determine if dark scaling is needed
    flat_exp_time_unit = flat_exp_time * u.s
    scale_dark = not np.isclose(flat_exp_time_unit.value, dark_for_flat_exposure.value)
    if scale_dark:
         print(f"Note: Flat exposure ({flat_exp_time_unit}) differs from dark exposure ({dark_for_flat_exposure}). Scaling dark.")

    # --- Create Master Flat for the specified filter ---
    print(f"Processing {len(flat_files)} flat frames for filter {filter_name}...")
    processed_flats = []
    for f in flat_files:
        try:
            # Read raw flat, assume ADU initially
            flat_ccd = CCDData.read(f, unit='adu')
            # Check filter consistency
            if flat_ccd.header.get('FILTER', '') != filter_name:
                 print(f"Warning: Filter mismatch in {f} (Expected {filter_name}, got {flat_ccd.header.get('FILTER', 'N/A')}). Skipping.")
                 continue
            # Check exposure time if needed (or assume it's correct)
            current_flat_exp = flat_ccd.header.get('EXPTIME', -1) * u.s
            if not np.isclose(current_flat_exp.value, flat_exp_time_unit.value):
                 print(f"Warning: Exposure time mismatch in {f} ({current_flat_exp} vs {flat_exp_time_unit}). Skipping.")
                 continue

            # Process individual flat: subtract bias, then dark
            flat_bias_sub = ccdproc.subtract_bias(flat_ccd, master_bias)
            flat_proc = ccdproc.subtract_dark(
                flat_bias_sub,
                master_dark_for_flats,
                dark_exposure=dark_for_flat_exposure,
                data_exposure=current_flat_exp, # Use actual flat exposure time
                exposure_unit=u.s,
                scale=scale_dark
            )
            processed_flats.append(flat_proc)
        except FileNotFoundError:
             print(f"Warning: Flat file {f} not found. Skipping.")
        except Exception as read_err:
             print(f"Warning: Error reading/processing flat file {f}: {read_err}. Skipping.")


    if not processed_flats:
        raise ValueError(f"No valid flat frames were loaded or processed for filter {filter_name}.")

    # Combine processed flat frames using median combine
    print(f"Combining {len(processed_flats)} processed flat frames...")
    flat_combiner = ccdproc.Combiner(processed_flats)
    # Optional: Apply sigma clipping
    flat_combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median, dev_func=np.ma.std)
    combined_flat = flat_combiner.median_combine()

    # Normalize the combined flat to create the master flat (values around 1)
    # Use nanmedian to be robust against potential masked pixels from clipping/BPM
    median_flat_value = np.nanmedian(combined_flat.data)
    if median_flat_value is None or np.isclose(median_flat_value, 0):
         raise ValueError("Median of combined flat is invalid (zero or all masked). Cannot normalize.")
    print(f"Normalizing combined flat by median value: {median_flat_value:.2f}")
    # Division handles units correctly if combined_flat has units (e.g., ADU or electron)
    master_flat = combined_flat.divide(median_flat_value)
    master_flat.meta['NCOMBINE'] = len(processed_flats)
    master_flat.meta['NORMVAL'] = (median_flat_value, 'Value used for normalization')
    master_flat.meta['FILTER'] = filter_name

    # Save the master flat frame
    master_flat.write(master_flat_file, overwrite=True)
    print(f"Master flat saved to {master_flat_file}")

    # --- Flat Correct the Science Frame ---
    # Read the dark-subtracted science frame
    print(f"Reading dark-subtracted science frame: {dark_subtracted_science_file}")
    science_dark_sub = CCDData.read(dark_subtracted_science_file) # Assume units are correct

    # Check if science frame filter matches the master flat
    science_filter = science_dark_sub.header.get('FILTER', 'UNKNOWN')
    if science_filter != filter_name:
        print(f"Error: Science frame filter '{science_filter}' does not match master flat filter '{filter_name}'.")
        raise ValueError("Filter mismatch between science and master flat.")

    # Perform flat correction using ccdproc.flat_correct
    # This divides the science data by the master flat and propagates uncertainty.
    print("Applying flat correction to science frame...")
    science_flat_corrected = ccdproc.flat_correct(science_dark_sub, master_flat)

    # Add history entry
    science_flat_corrected.meta['HISTORY'] = f'Flat corrected using {os.path.basename(master_flat_file)}'

    # Save the fully reduced science frame
    science_flat_corrected.write(output_flat_corrected_file, overwrite=True)
    print(f"Flat-corrected science frame saved to {output_flat_corrected_file}")
    print(f"Mean value after flat correction: {np.mean(science_flat_corrected.data):.2f}")


except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
except CCDDataProcessingError as e:
    print(f"Error during CCD processing: {e}")
except ValueError as e:
    print(f"Value Error: {e}")
except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred during flat fielding: {e}")

```

This Python script implements the complete process of generating and applying a master flat-field correction using `ccdproc`. It starts by loading the necessary master bias and an appropriate master dark frame (potentially scaling the dark if exposure times differ, although matching is preferred). It then iterates through a list of raw flat-field frames for a specific filter, applying bias and dark corrections to each individual flat using `ccdproc.subtract_bias` and `ccdproc.subtract_dark`. These processed flats are robustly combined, typically via median, using `ccdproc.Combiner` to create a high signal-to-noise combined flat. A critical step follows: the combined flat is normalized by dividing by its median value, ensuring the resulting master flat represents relative pixel sensitivity variations around unity. Finally, the dark-subtracted science frame (verified to be from the same filter) is divided by this normalized master flat using `ccdproc.flat_correct`, which also handles uncertainty propagation. The resulting science frame is thus corrected for pixel-to-pixel sensitivity variations and large-scale illumination effects, yielding data suitable for accurate photometric analysis.

**3.6 Building Master Calibration Frames**

The creation of high-quality master calibration frames – master bias ($\bar{B}$), master dark ($\bar{D}_{master}$), and master flat ($\bar{F}_{master}$) – is a cornerstone of reliable image reduction. As illustrated in the preceding sections, these master frames are not derived from single calibration exposures but are constructed by statistically combining multiple individual frames of the same type (bias, dark, or flat) taken under identical conditions (exposure time, temperature, filter). This combination process serves two crucial purposes: reducing random noise inherent in individual exposures and identifying and rejecting outlier pixel values caused by transient events like cosmic rays or temporary pixel defects (Gillessen et al., 2020; Vacca, 2021). The quality of these master frames directly impacts the accuracy and precision of the final reduced science data.

**Noise Reduction:** Any single calibration exposure contains inherent random noise, primarily read noise (for bias frames) and shot noise from dark current or flat-field illumination. By combining $N$ independent calibration frames, the random noise contribution in the resulting master frame can be significantly reduced. If the combination method is averaging, the noise (standard deviation) in the averaged frame ($\sigma_{master, avg}$) is reduced by a factor of $\sqrt{N}$ compared to a single frame ($\sigma_{single}$):

$\sigma_{master, avg} = \frac{\sigma_{single}}{\sqrt{N}}$

If the median is used for combination, the noise reduction is slightly less efficient but still substantial, approximately:

$\sigma_{master, median} \approx \sqrt{\frac{\pi}{2N}} \sigma_{single} \approx \frac{1.253 \sigma_{single}}{\sqrt{N}}$ (for Gaussian noise)

Combining a sufficient number of frames (typically $N \ge 10-20$) ensures that the noise introduced by the master calibration frame during the subtraction or division steps is significantly smaller than the noise inherent in the science data itself, preventing the calibration process from unduly degrading the final signal-to-noise ratio.

**Outlier Rejection:** Individual calibration frames can be affected by transient events like cosmic rays (especially significant in longer dark or flat exposures) or have pixels exhibiting temporary anomalous behavior. Simple averaging would incorporate these outliers, potentially biasing the master frame. Statistical combination methods are designed to identify and reject such deviant pixels before the final combination:
*   **Median Combination:** The median is inherently robust to outliers. If fewer than 50% of the frames have an outlier at a given pixel, the median value will not be significantly affected. It is often the preferred method for combining bias and dark frames due to its simplicity and robustness.
*   **Sigma-Clipped Averaging:** This method calculates the mean and standard deviation (or median and median absolute deviation for more robustness) of pixel values across the stack of frames at each $(x, y)$ position. Pixels deviating by more than a specified threshold (e.g., 3-sigma) from the central value are temporarily excluded (masked), and the mean (or median) is recalculated using only the remaining valid pixels. This process can be iterated. Sigma clipping is effective if the underlying distribution of good pixel values is roughly symmetric and outliers are relatively few. It can be slightly more noise-efficient than the median if outliers are well-handled, making it a common choice for combining flat-field frames where maximizing SNR is paramount.
*   **Min/Max Rejection:** A simpler form where the highest and lowest pixel values (or a certain number/percentage of highest/lowest) at each position are discarded before averaging the remaining values. It is less statistically robust than sigma clipping but computationally faster. Other variants like percentile clipping exist.

**Implementation:** Libraries like `ccdproc` provide convenient tools (e.g., `ccdproc.Combiner`) that implement these combination algorithms, often incorporating masking and uncertainty propagation. The `Combiner` class can be initialized with a list of `CCDData` objects (representing the individual calibration frames). Methods like `.median_combine()` or `.average_combine()` perform the final combination after optional rejection steps like `.sigma_clipping()` or `.minmax_clipping()` have been applied. The user specifies clipping thresholds and central/deviation statistics. It is crucial that the input frames are properly processed (e.g., bias-subtracted for darks and flats) *before* combination. The resulting master calibration frame should ideally have its associated uncertainty calculated based on the input uncertainties and the combination method used, allowing for correct error propagation in subsequent reduction steps.

Creating robust master calibration frames requires careful acquisition of a sufficient number of high-quality individual calibration exposures taken under conditions matching the science data, followed by statistically sound combination techniques to minimize noise and reject outliers.

**3.7 Identifying and Masking Bad Pixels**

No astronomical detector is perfect; manufacturing variations and operational stresses inevitably lead to some pixels exhibiting behavior deviating significantly from the norm. These "bad" pixels can introduce spurious signals or data loss, potentially corrupting scientific measurements if not properly handled. Identifying these pixels and creating a **Bad Pixel Mask (BPM)** is therefore an essential part of data reduction and quality assessment (Lesser, 2015). The BPM is typically a boolean image of the same dimensions as the detector, where `True` (or a non-zero integer flag) indicates a bad pixel and `False` (or zero) indicates a good pixel. This mask is then used to exclude compromised pixels from subsequent calculations.

Types of bad pixels include:
*   **Dead Pixels:** Pixels with very low or zero quantum efficiency. They appear consistently dark in flat-field frames.
*   **Hot Pixels:** Pixels with significantly elevated dark current compared to their neighbors. They appear consistently bright in dark frames, with brightness scaling with exposure time.
*   **Unstable/Flickering Pixels:** Pixels whose bias level, dark current, or sensitivity fluctuates unpredictably over time. These are harder to identify with static masks but may be flagged by monitoring temporal variations.
*   **Bad Columns/Rows/Regions:** Manufacturing defects can sometimes affect entire columns, rows, or larger contiguous areas of the detector. These are often identified visually or through automated detection of correlated anomalous behavior.
*   **Charge Traps:** Lattice defects that temporarily capture and later release electrons, potentially affecting charge transfer efficiency in CCDs or causing persistence effects. These are typically identified through specialized tests.

Methods for identifying bad pixels often utilize the master calibration frames generated during reduction:
*   **From Master Flat:** Pixels showing significantly low response in the *normalized* master flat frame (e.g., values < 0.5 or < 0.8, depending on tolerance) are flagged as dead or low-QE. Conversely, pixels with excessively high response (e.g., > 1.5 or > 2.0) might also be flagged, although this is less common unless indicating unstable behavior. Thresholds are typically based on standard deviations from the mean normalized value (1.0).
*   **From Master Dark:** Pixels exhibiting values significantly above the median dark level in a long-exposure master dark frame are identified as hot pixels. A threshold based on multiple standard deviations above the median dark current, or an absolute count threshold, is commonly used.
*   **From Master Bias:** While less common for identifying *bad* pixels, extreme outliers in the master bias frame (many sigma from the mean) might indicate pixels with fundamental readout problems.
*   **From Temporal Stability Analysis:** Comparing multiple dark or flat frames taken over time can reveal pixels whose values fluctuate significantly more than expected from noise alone, indicating instability. Calculating the standard deviation across a stack of darks or flats at each pixel can highlight such flickering pixels.
*   **Manufacturer Data:** Detector manufacturers often provide initial maps of known cosmetic defects identified during laboratory testing. This serves as a valuable starting point for the BPM.

Once identified through these various methods, the locations of bad pixels from different tests are compiled into a single, comprehensive master BPM. This mask should ideally be created and maintained by the observatory staff, as bad pixels can sometimes evolve over time (new ones appearing, some healing spontaneously or through annealing processes). The BPM is often stored as a separate FITS file (typically an image with integer flags or 0/1 values) or incorporated directly into the data quality (DQ) extensions of processed science files.

**Using the Mask:** The BPM is crucial for subsequent processing and analysis. When incorporated into the `.mask` attribute of a `CCDData` object (Section 3.2), it signals to processing functions (like those in `ccdproc` or analysis tools like `photutils`) that these pixels should be ignored during calculations:
*   During image combination steps (e.g., co-adding science frames), masked pixels are typically excluded from median or average calculations at that location.
*   In photometric or spectroscopic measurements, flux contributions from masked pixels falling within measurement apertures are usually omitted, or the measurement might be flagged as potentially unreliable.
*   For visualization or certain types of analysis where contiguous data are required (e.g., some deconvolution algorithms), it might be necessary to **interpolate** over bad pixels using the values of nearby valid neighbors. Simple methods include replacing the bad pixel with the median or mean of its immediate neighbors. More sophisticated techniques involve 2D interpolation schemes. However, interpolation introduces correlated noise and artificially fills in data, so it should be used judiciously and documented clearly. It is generally preferable to simply ignore (mask) bad pixels whenever the analysis algorithm allows.

Creating and consistently applying a reliable BPM is vital for preventing corrupted data from influencing scientific results and ensuring the overall quality and veracity of the reduced data products.

**3.8 Cosmic Ray Detection and Removal Algorithms**

Cosmic rays, high-energy charged particles impinging on the detector, represent a significant source of transient artifacts in astronomical images, particularly for space-based observations and long ground-based exposures (van Dokkum, 2001; Zhang & Bloom, 2020). They manifest as sharp, localized spikes or elongated streaks of charge, often spanning only one or a few pixels, that are unrelated to the astronomical scene. If not removed, they can be mistaken for real astrophysical objects (e.g., faint stars, galaxies, supernovae) or corrupt photometric and spectroscopic measurements by artificially inflating pixel values. Effective cosmic ray identification and removal is therefore a critical component of image reduction.

The key characteristic distinguishing cosmic rays from genuine sources is their transient nature (appearing in only one exposure at a given sky location if images are dithered) and, usually, their sharp spatial profile compared to the instrument's Point Spread Function (PSF), which describes how a point source is blurred by the optics and atmosphere. Several algorithms exploit these properties for detection and removal:

*   **Median Combination of Dithered Exposures:** This is often the most robust method when multiple ($N \ge 3$, ideally $\ge 5$) exposures of the same field are available, taken with small spatial offsets (dithers) between them. The images must first be accurately registered to align the celestial sources. Then, for each pixel location in the final combined grid, the stack of corresponding pixel values from the individual aligned frames is considered. Taking the median value of this stack effectively rejects cosmic rays, as a cosmic ray hit will affect only one frame at that specific grid location and thus appear as an extreme outlier unlikely to be chosen as the median. Sigma clipping can be applied before the median for further robustness. This method simultaneously combines the science data and removes cosmic rays but requires a specific observing strategy (dithering) and sufficient exposures.
*   **Thresholding/Sigma Clipping (Single Image):** Simpler algorithms operate on individual images by comparing a pixel's value to its local neighbors (e.g., within a 3x3 or 5x5 box). If the central pixel significantly exceeds the local median or mean by a predetermined threshold (e.g., 5 times the estimated local standard deviation, $5\sigma$), it might be flagged as a cosmic ray. While computationally fast, this approach is prone to errors: it can miss lower-energy cosmic rays that don't exceed the threshold, erroneously clip the bright, sharp cores of real stars or compact objects that mimic cosmic ray profiles locally, or misidentify statistical noise spikes as cosmic rays, especially in low signal-to-noise data. Its effectiveness is limited, particularly for preserving faint or sharp astrophysical features.
*   **Laplacian Edge Detection (e.g., L.A.Cosmic):** More sophisticated algorithms leverage the difference in sharpness between cosmic rays and PSF-convolved sources. Cosmic rays typically have much sharper gradients (edges) than even unresolved stars. The L.A.Cosmic algorithm (van Dokkum, 2001), widely implemented (e.g., in the `astroscrappy` Python package - McCully et al., 2018), capitalizes on this. It operates roughly as follows:
    1.  **Image Modeling:** Create a model of the smooth astronomical background and sources, often by applying a median filter to the image, which suppresses sharp features like cosmic rays and stellar cores.
    2.  **Noise Modeling:** Estimate the expected noise (variance) at each pixel, typically combining read noise and Poisson noise calculated from the model image created in step 1.
    3.  **Laplacian Calculation:** Compute the discrete Laplacian (approximating the second spatial derivative) of the original image. Sharp features like cosmic ray edges produce strong positive or negative peaks in the Laplacian image.
    4.  **Candidate Identification:** Identify pixels where the Laplacian signal, normalized by the expected noise in the Laplacian (derived from the noise model), exceeds a primary significance threshold (e.g., `sigclip` parameter, often ~4-5). These are initial cosmic ray candidates.
    5.  **Refinement with Neighbors:** Refine the candidate list by applying secondary criteria. A contrast limit (`objlim`) compares the original pixel value to its immediate neighbors in the median-filtered model; if the pixel is not significantly brighter than this smoothed version (i.e., it might be the core of a real object), it is rejected. A neighbor threshold might require a minimum number of adjacent pixels to also be flagged as part of the cosmic ray event.
    6.  **Mask Growth:** Optionally grow the mask around identified cosmic ray pixels slightly to ensure the entire affected region is flagged.
    7.  **Pixel Replacement (Optional):** After identification, the algorithm typically updates a mask indicating the cosmic ray locations. Some implementations also offer options to replace the flagged pixel values, often using values from the median-filtered model image created in step 1 or employing more sophisticated interpolation schemes. However, relying solely on the mask and ignoring flagged pixels in subsequent analysis is often preferred over interpolation.
    L.A.Cosmic and similar morphology-based algorithms (like `DetectCR` by empowering RSE communities) are generally effective at identifying cosmic rays while preserving most astronomical sources, though careful tuning of parameters (signal thresholds, contrast limits, filtering scales) might be required depending on the data characteristics (PSF size, crowding, noise levels). Recent developments also explore deep learning approaches for cosmic ray detection, which can potentially learn more complex features to distinguish cosmic rays from astrophysical objects (Zhang & Bloom, 2020; Jia et al., 2023).

*   **Comparison of Consecutive Exposures:** For time-series observations or repeated exposures taken without significant dithering, comparing consecutive frames can identify pixels showing sudden, large increases in flux inconsistent with typical source variability or expected noise fluctuations. Subtracting consecutive frames can highlight these transient events.

The choice of method depends heavily on the available data (single vs. multiple exposures, dithering strategy) and the scientific requirements for fidelity. Applying cosmic ray rejection *before* co-adding multiple exposures (if median combination is not used) is generally preferred. If only single exposures exist, algorithms like L.A.Cosmic are commonly employed. The removed pixel locations are typically recorded in the data quality mask (BPM or a separate cosmic ray mask). Careful visual inspection of the cleaned image and the generated mask is always recommended to ensure that real astrophysical features were not inadvertently removed or altered by the process.

**3.9 Practical Workflow: A Standard CCD Reduction Script (`ccdproc`)**

Bringing together the steps discussed above, a typical workflow for basic CCD image reduction follows a specific, logical sequence to ensure artifacts are removed correctly without introducing biases from subsequent steps. While manual implementation using NumPy and Astropy core functions is possible, the **`ccdproc`** package, an Astropy-affiliated library, provides a convenient and robust framework specifically designed for this purpose, operating seamlessly with `CCDData` objects and handling metadata and uncertainty propagation (Craig et al., 2017). Adopting a structured workflow, often implemented as a Python script, ensures consistency and reproducibility.

The standard processing sequence, facilitated by `ccdproc`, is generally:

1.  **Data Organization and Ingestion:**
    *   Collect and organize raw data files: science frames, individual bias frames, individual dark frames (ideally matching science exposure times), and individual flat-field frames (for each filter used).
    *   Use tools like `ccdproc.ImageFileCollection` to easily discover and group these files based on FITS header keywords (e.g., `IMAGETYP`, `FILTER`, `EXPTIME`).
    *   Read relevant files into lists of `CCDData` objects, ensuring units (`adu`) and essential metadata (gain, read noise from headers) are correctly associated.
2.  **Master Bias Creation:**
    *   Use `ccdproc.Combiner` initialized with the list of bias `CCDData` objects.
    *   Apply optional sigma clipping (`combiner.sigma_clipping(...)`).
    *   Generate the master bias frame using `combiner.median_combine()` or `combiner.average_combine()`.
    *   Save the master bias `CCDData` object to a FITS file.
3.  **Master Dark Creation:**
    *   For each required exposure time:
        *   Initialize `ccdproc.Combiner` with the list of corresponding raw dark `CCDData` objects.
        *   Subtract the master bias from each dark frame *before* combination (this can sometimes be handled within the `Combiner` framework or done separately using `ccdproc.subtract_bias` on each frame). An alternative, often simpler approach, is to use `ccdproc.Combiner`'s ability to take a `master_bias` argument directly during combination.
        *   Apply optional sigma clipping.
        *   Generate the master dark frame using median or average combination.
        *   Save the master dark `CCDData` object (clearly indicating the exposure time in the filename or metadata).
4.  **Master Flat Creation (per filter):**
    *   For each filter:
        *   Initialize `ccdproc.Combiner` with the list of corresponding raw flat `CCDData` objects.
        *   Ensure bias subtraction (e.g., passing `master_bias` to `Combiner` or `subtract_bias` first).
        *   Ensure dark subtraction using the appropriate master dark (e.g., passing `master_dark` and exposure time info to `Combiner` or using `subtract_dark` first). `ccdproc` functions often require explicit exposure time information for scaling if necessary.
        *   Apply optional sigma clipping.
        *   Generate the combined flat using median or average combination.
        *   Normalize the combined flat (e.g., dividing by its median value) to create the master flat (values ~1).
        *   Save the master flat `CCDData` object (clearly indicating the filter).
5.  **Bad Pixel Mask Creation (Optional but Recommended):**
    *   Generate a BPM based on thresholds applied to the master darks (hot pixels), master flats (dead/low QE pixels), and potentially other diagnostics or manufacturer data (Section 3.7).
    *   Save the BPM as a FITS image (e.g., 0 for good, 1 for bad).
6.  **Science Frame Processing (Iterate through science frames):**
    *   Read the raw science frame into a `CCDData` object.
    *   Associate the BPM with the `CCDData` object's `.mask` attribute.
    *   **Overscan Correction (Optional):** Apply `ccdproc.subtract_overscan` if relevant and desired.
    *   **Bias Correction:** Apply `ccdproc.subtract_bias` using the master bias.
    *   **Dark Correction:** Apply `ccdproc.subtract_dark` using the appropriate master dark, providing necessary exposure time information (`data_exposure`, `dark_exposure`, `exposure_unit`) for potential scaling via the `scale` argument.
    *   **Flat Correction:** Apply `ccdproc.flat_correct` using the appropriate master flat for the science frame's filter. Check for filter consistency.
    *   **Cosmic Ray Rejection (Optional):** Apply `ccdproc.cosmicray_lacosmic`, providing gain and read noise. This updates the `.mask`. Optionally, interpolate flagged pixels if desired, though often masking is sufficient.
    *   **Gain Correction (Optional but Recommended):** Apply `ccdproc.gain_correct` to convert data from ADU to electrons, simplifying noise analysis and physical interpretation. Update the `.unit` attribute.
    *   **Uncertainty Calculation (Optional but Recommended):** Use `ccdproc.create_deviation` to calculate the uncertainty array based on gain, read noise, and the processed data values, attaching it to the `CCDData` object's `.uncertainty` attribute. This function applies the standard noise model.
    *   Save the fully processed science frame (`CCDData` object, including data, uncertainty, mask, updated header with HISTORY) to a new FITS file.

The `ccdproc` package simplifies many of these steps by providing high-level functions that operate on `CCDData` objects, automatically handling tasks like uncertainty propagation (assuming input uncertainties are correctly defined and uncorrelated) and basic header updates. Implementing this workflow within a well-documented Python script ensures a systematic, repeatable, and traceable reduction process.

**3.10 Examples in Practice (Python): Image Reduction Workflows**

The following examples illustrate the application of the basic image reduction steps, primarily using functions from the `ccdproc` library, within specific astronomical contexts. These snippets demonstrate how the core procedures of bias subtraction, dark correction, flat-fielding, and cosmic ray removal are applied to data from solar, planetary, stellar, exoplanetary, Galactic, extragalactic, and cosmological observations. While the underlying principles are similar, the specific steps emphasized or the parameters chosen may vary depending on the typical characteristics of the data in each field (e.g., exposure times, importance of faint features, presence of extended sources). These examples assume master calibration files have been previously created or are available, focusing on their application to science data.

**3.10.1 Solar: Applying Flat-Field to SDO/AIA (Illustrative)**
Data from space-based solar observatories like SDO are typically delivered after extensive pipeline processing by the instrument teams, often including sophisticated calibration steps beyond basic reduction (e.g., geometric corrections, alignment, differential rotation correction). Therefore, users rarely perform fundamental bias, dark, or flat corrections themselves. However, understanding these steps conceptually is useful. This example provides an illustrative snippet showing how one *would* apply a flat-field correction to an AIA image using `ccdproc`, assuming, for demonstration purposes, that a suitable master flat exists and prior corrections (like bias/dark removal, which are often negligible or handled differently for EUV detectors) have been performed or are not needed. The focus here is purely on the syntax of applying the flat correction step itself.

```python
# Conceptual example: Applying a pre-existing master flat to an AIA image
import numpy as np
from astropy.nddata import CCDData, CCDDataProcessingError
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Solar example.")
    ccdproc_available = False
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
aia_image_file = 'aia_image_preflat.fits'
master_flat_aia_file = 'master_flat_aia.fits'
output_file = 'aia_image_flat_corrected.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    if not os.path.exists(aia_image_file):
        print(f"Creating dummy file: {aia_image_file}")
        hdu = fits.PrimaryHDU((np.random.rand(100, 100) * 500 + 50).astype(np.float32))
        hdu.header['INSTRUME'] = 'AIA'
        hdu.header['BUNIT'] = 'DN'
        hdu.writeto(aia_image_file, overwrite=True)
    if not os.path.exists(master_flat_aia_file):
        print(f"Creating dummy file: {master_flat_aia_file}")
        # Flat should be around 1, add some structure
        yy, xx = np.indices((100, 100))
        flat_data = 1.0 + 0.05 * np.sin(xx / 10.0) + np.random.normal(0, 0.01, size=(100, 100))
        hdu = fits.PrimaryHDU(flat_data.astype(np.float32))
        hdu.header['INSTRUME'] = 'AIA_FLAT'
        hdu.writeto(master_flat_aia_file, overwrite=True)

if ccdproc_available:
    try:
        # Load the (pre-processed or raw) AIA image data.
        # Specify units if known, otherwise assume arbitrary units or ADU.
        print(f"Loading image: {aia_image_file}")
        # Use dummy data if actual read fails
        try:
            aia_image_ccd = CCDData.read(aia_image_file, unit='adu') # Use 'adu' or actual unit
        except FileNotFoundError:
             print(f"Warning: File {aia_image_file} not found, using dummy data.")
             aia_image_ccd = CCDData(np.random.rand(100, 100) * 500 + 50, unit='adu', meta={'INSTRUME': 'AIA'})


        # Load the corresponding normalized master flat field.
        # The flat should have values around 1.0 and no physical units (or be dimensionless).
        print(f"Loading master flat: {master_flat_aia_file}")
        # Use dummy data if actual read fails
        try:
            master_flat_aia = CCDData.read(master_flat_aia_file, unit='') # Dimensionless unit
        except FileNotFoundError:
             print(f"Warning: File {master_flat_aia_file} not found, using dummy data.")
             yy, xx = np.indices((100, 100))
             flat_data = 1.0 + 0.05 * np.sin(xx / 10.0) + np.random.normal(0, 0.01, size=(100, 100))
             master_flat_aia = CCDData(flat_data, unit='', meta={'INSTRUME': 'AIA_FLAT'})


        # Apply flat correction using ccdproc.flat_correct.
        # This function divides the data (and propagates uncertainty if present)
        # by the master flat field.
        print("Applying conceptual flat correction...")
        aia_flat_corrected = ccdproc.flat_correct(aia_image_ccd, master_flat_aia)

        # Add a history record to the corrected image metadata.
        aia_flat_corrected.meta['HISTORY'] = 'Flat-field corrected (conceptual)'

        # Save the result (optional - commented out for dummy data)
        # aia_flat_corrected.write(output_file, overwrite=True)
        print(f"Conceptual flat correction applied. Result mean: {np.mean(aia_flat_corrected.data):.2f}")
        print(f"(If successful, result would normally be saved to {output_file})")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred in the conceptual Solar example: {e}")
else:
     print("Skipping conceptual Solar example: ccdproc unavailable.")

```

This conceptual Python code demonstrates the core operation of flat-field correction in a solar physics context, specifically for illustrative purposes with SDO/AIA data which is typically pipeline-processed. It first loads the target image and a corresponding pre-normalized master flat field (representing pixel sensitivity variations) into `CCDData` objects using `ccdproc.CCDData.read`. The key step is the application of `ccdproc.flat_correct`, which performs the pixel-wise division of the science image data by the master flat data. This function is designed to correctly handle the associated uncertainties (if present in the input `CCDData` objects) according to standard error propagation rules for division. A `HISTORY` keyword is added to the resulting `CCDData` object's metadata to document the processing step. While actual SDO/AIA reduction is handled upstream, this example clarifies the fundamental flat-fielding operation common across many astronomical imaging datasets.

**3.10.2 Planetary: Bias and Flat Correction for Jupiter Image**
Ground-based imaging of bright planets like Jupiter often involves relatively short exposures to freeze atmospheric seeing or capture rapid rotation. In such cases, the contribution from dark current may be negligible compared to the sky background and source flux, especially if the detector is reasonably cooled. The primary reduction steps then typically involve subtracting the electronic bias offset and correcting for pixel sensitivity variations and illumination patterns using a flat field. This example simulates this common planetary imaging reduction scenario, applying master bias subtraction followed by master flat correction to a raw Jupiter image using `ccdproc` functions.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Planetary example.")
    ccdproc_available = False
import astropy.units as u
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
jupiter_raw_file = 'jupiter_raw_R.fits'
master_bias_file = 'master_bias.fits'
master_flat_file = 'master_flat_filterR.fits' # Flat for the correct filter
output_file = 'jupiter_reduced_R.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    # Master Bias
    if not os.path.exists(master_bias_file):
        print(f"Creating dummy file: {master_bias_file}")
        fits.PrimaryHDU((np.ones((128, 128)) * 510).astype(np.float32)).writeto(master_bias_file, overwrite=True)
    # Master Flat (Filter R)
    if not os.path.exists(master_flat_file):
        print(f"Creating dummy file: {master_flat_file}")
        yy, xx = np.indices((128, 128))
        flat_data = 1.0 - 0.1 * (((xx - 64)/64)**2 + ((yy - 64)/64)**2) # Simple vignetting
        flat_data += np.random.normal(0, 0.01, size=(128, 128))
        hdu = fits.PrimaryHDU(flat_data.astype(np.float32))
        hdu.header['FILTER'] = 'R'
        hdu.writeto(master_flat_file, overwrite=True)
    # Raw Jupiter Image
    if not os.path.exists(jupiter_raw_file):
        print(f"Creating dummy file: {jupiter_raw_file}")
        # Simulate bias + source + sky + flat field effect + noise
        bias_level = 510
        read_noise = 5
        # Load the dummy flat data to apply its effect
        flat_data = fits.getdata(master_flat_file)
        # Simulate Jupiter (bright disk) + sky
        yy, xx = np.indices((128, 128))
        jupiter_signal = 15000 * np.exp(-0.5 * (((xx - 64)/20)**2 + ((yy - 64)/25)**2))
        sky_level = 1000
        raw_signal = (jupiter_signal + sky_level) * flat_data # Apply flat effect
        raw_noisy_signal = bias_level + np.random.poisson(raw_signal) + np.random.normal(0, read_noise, size=(128, 128))
        hdr = fits.Header({'FILTER': 'R', 'OBJECT': 'Jupiter'})
        fits.PrimaryHDU(raw_noisy_signal.astype(np.float32), header=hdr).writeto(jupiter_raw_file, overwrite=True)


if ccdproc_available:
    try:
        # Load the raw Jupiter image
        print(f"Loading raw Jupiter image: {jupiter_raw_file}")
        try:
            jupiter_raw = CCDData.read(jupiter_raw_file, unit='adu')
        except FileNotFoundError:
             print(f"Warning: File {jupiter_raw_file} not found, using dummy data.")
             jupiter_raw = CCDData(np.random.rand(128, 128) * 10000 + 510, unit='adu', meta={'FILTER': 'R'})

        # Load the master bias frame
        print(f"Loading master bias: {master_bias_file}")
        try:
            master_bias = CCDData.read(master_bias_file, unit='adu')
        except FileNotFoundError:
             print(f"Warning: File {master_bias_file} not found, using dummy data.")
             master_bias = CCDData(np.ones((128, 128)) * 510, unit='adu')

        # Load the master flat frame for the corresponding filter ('R')
        print(f"Loading master flat: {master_flat_file}")
        try:
            master_flat_R = CCDData.read(master_flat_file, unit='') # Normalized flat is unitless
        except FileNotFoundError:
             print(f"Warning: File {master_flat_file} not found, using dummy data.")
             master_flat_R = CCDData(np.random.normal(loc=1.0, scale=0.02, size=(128, 128)), unit='')


        # --- Perform Bias Subtraction ---
        print("Subtracting master bias...")
        jupiter_bias_sub = ccdproc.subtract_bias(jupiter_raw, master_bias)

        # --- Perform Flat Correction ---
        # Assuming dark current is negligible for the short exposure typically
        # used for bright planets, we proceed directly to flat correction.
        print("Applying flat correction (assuming negligible dark current)...")
        # Ensure the filter matches if metadata exists
        if jupiter_bias_sub.meta.get('FILTER') != master_flat_R.meta.get('FILTER', jupiter_bias_sub.meta.get('FILTER')):
            print(f"Warning: Filter mismatch between science ({jupiter_bias_sub.meta.get('FILTER')}) and flat ({master_flat_R.meta.get('FILTER')}). Proceeding cautiously.")

        jupiter_flat_corr = ccdproc.flat_correct(jupiter_bias_sub, master_flat_R)

        # Add history records
        jupiter_flat_corr.meta['HISTORY'] = f'Bias subtracted using {os.path.basename(master_bias_file)}'
        jupiter_flat_corr.meta['HISTORY'] = f'Flat corrected using {os.path.basename(master_flat_file)}'

        # Save the reduced image (optional - commented out for dummy data)
        # jupiter_flat_corr.write(output_file, overwrite=True)
        print(f"Planetary reduction complete (Bias, Flat). Result mean: {np.mean(jupiter_flat_corr.data):.1f} ADU")
        print(f"(If successful, result would normally be saved to {output_file})")


    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the Planetary example: {e}")
else:
     print("Skipping Planetary example: ccdproc unavailable.")

```

This Python script addresses a common reduction scenario encountered in planetary science imaging, particularly for bright objects where exposures are short and dark current can often be neglected. It loads the raw image of Jupiter (or a similar target), the corresponding master bias frame, and the master flat frame taken through the same filter (e.g., 'R' band), all as `CCDData` objects using `ccdproc.CCDData.read`. The script then sequentially applies the necessary corrections using `ccdproc` functions: first, `ccdproc.subtract_bias` removes the electronic offset, and second, `ccdproc.flat_correct` divides the bias-subtracted image by the normalized master flat to correct for pixel sensitivity and illumination variations. Appropriate `HISTORY` records are added to the output `CCDData` object's metadata, documenting the applied reduction steps. The resulting image, corrected for bias and flat-field effects, is then ready for further analysis, such as image sharpening or feature measurement.

**3.10.3 Stellar: Full Reduction Sequence for Open Cluster Image**
Observations of stellar fields, such as open or globular clusters, often involve longer exposure times compared to bright planet imaging, aiming to detect fainter stars. Consequently, dark current accumulation can become significant, necessitating its correction in addition to bias subtraction and flat-fielding. This example demonstrates the complete, standard reduction sequence for such a scenario: bias subtraction, dark current subtraction (using a master dark frame matching the science exposure time), and flat-field correction (using a master flat for the appropriate filter). This sequence ensures that all major static instrumental signatures are removed in the correct order, yielding a science-ready image suitable for stellar photometry or astrometry.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Stellar example.")
    ccdproc_available = False
import astropy.units as u
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
science_exp_time_val = 180.0 # seconds
filter_name = 'V'
cluster_raw_file = f'cluster_raw_{filter_name}.fits'
master_bias_file = 'master_bias.fits'
master_dark_file = f'master_dark_{int(science_exp_time_val)}s.fits' # Dark matching science exposure
master_flat_file = f'master_flat_filter{filter_name}.fits'
output_file = f'cluster_reduced_{filter_name}.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    # Master Bias
    if not os.path.exists(master_bias_file):
        print(f"Creating dummy file: {master_bias_file}")
        fits.PrimaryHDU((np.ones((150, 150)) * 520).astype(np.float32)).writeto(master_bias_file, overwrite=True)
    # Master Dark (matching exposure)
    if not os.path.exists(master_dark_file):
        print(f"Creating dummy file: {master_dark_file}")
        dark_signal = np.random.normal(30, 3, size=(150, 150)) # Dark signal (already bias subtracted)
        hdr = fits.Header({'EXPTIME': science_exp_time_val, 'BUNIT':'adu'}) # Store exposure time
        fits.PrimaryHDU(dark_signal.astype(np.float32), header=hdr).writeto(master_dark_file, overwrite=True)
    # Master Flat (V filter)
    if not os.path.exists(master_flat_file):
        print(f"Creating dummy file: {master_flat_file}")
        flat_data = np.random.normal(1.0, 0.03, size=(150, 150)) # Normalized flat around 1
        hdr = fits.Header({'FILTER': filter_name})
        fits.PrimaryHDU(flat_data.astype(np.float32), header=hdr).writeto(master_flat_file, overwrite=True)
    # Raw Cluster Image
    if not os.path.exists(cluster_raw_file):
        print(f"Creating dummy file: {cluster_raw_file}")
        # Simulate bias + dark + source + flat effect + noise
        bias_level = 520
        dark_data = fits.getdata(master_dark_file) # Use the dummy master dark signal
        flat_data = fits.getdata(master_flat_file)
        read_noise = 6
        # Simulate faint stars
        sky_level = 800
        star_field = np.zeros((150, 150))
        n_stars = 50
        x_pos = np.random.uniform(0, 150, n_stars)
        y_pos = np.random.uniform(0, 150, n_stars)
        fluxes = 10**(np.random.uniform(2, 4, n_stars)) # Wide range of fluxes
        yy, xx = np.indices((150, 150))
        psf_sigma = 1.8
        for x, y, flux in zip(x_pos, y_pos, fluxes):
             dist_sq = (xx - x)**2 + (yy - y)**2
             star_field += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Combine components
        raw_signal = (sky_level + star_field) * flat_data # Apply flat effect
        raw_noisy_signal = bias_level + dark_data + np.random.poisson(raw_signal) + np.random.normal(0, read_noise, size=(150, 150))
        hdr = fits.Header({'EXPTIME': science_exp_time_val, 'FILTER': filter_name, 'OBJECT': 'SimCluster'})
        fits.PrimaryHDU(raw_noisy_signal.astype(np.float32), header=hdr).writeto(cluster_raw_file, overwrite=True)


if ccdproc_available:
    try:
        # --- Load Calibration Frames ---
        print(f"Loading master bias: {master_bias_file}")
        try:
            master_bias = CCDData.read(master_bias_file, unit='adu')
        except FileNotFoundError:
             print(f"Warning: File {master_bias_file} not found, using dummy data.")
             master_bias = CCDData(np.ones((150, 150)) * 520, unit='adu')

        print(f"Loading master dark: {master_dark_file}")
        try:
            master_dark = CCDData.read(master_dark_file) # Read master dark
            # Ensure units match science data or can be converted
            if master_dark.unit is None: master_dark.unit = 'adu'
            # Get dark exposure time from its header
            dark_exposure_time = master_dark.header['EXPTIME'] * u.s
        except FileNotFoundError:
             print(f"Warning: File {master_dark_file} not found, using dummy data.")
             dark_exposure_time = science_exp_time_val * u.s
             master_dark = CCDData(np.random.normal(30, 3, size=(150, 150)), unit='adu', meta={'EXPTIME': science_exp_time_val})
        except KeyError:
             print(f"Warning: EXPTIME missing in {master_dark_file} header. Assuming {science_exp_time_val}s.")
             dark_exposure_time = science_exp_time_val * u.s
             if not hasattr(master_dark, 'meta'): master_dark.meta = {} # Ensure meta exists
             master_dark.meta['EXPTIME'] = science_exp_time_val # Add it


        print(f"Loading master flat: {master_flat_file}")
        try:
            master_flat = CCDData.read(master_flat_file, unit='') # Unitless normalized flat
            flat_filter = master_flat.header.get('FILTER', 'UNKNOWN')
        except FileNotFoundError:
             print(f"Warning: File {master_flat_file} not found, using dummy data.")
             master_flat = CCDData(np.random.normal(1.0, 0.03, size=(150, 150)), unit='', meta={'FILTER': filter_name})
             flat_filter = filter_name


        # --- Load Raw Science Frame ---
        print(f"Loading raw science frame: {cluster_raw_file}")
        try:
            cluster_raw = CCDData.read(cluster_raw_file, unit='adu')
            science_exposure_time = cluster_raw.header['EXPTIME'] * u.s
            science_filter = cluster_raw.header.get('FILTER', 'UNKNOWN')
        except FileNotFoundError:
             print(f"Warning: File {cluster_raw_file} not found, using dummy data.")
             science_exposure_time = science_exp_time_val * u.s
             science_filter = filter_name
             cluster_raw = CCDData(np.random.rand(150, 150) * 1000 + 520 + 30, unit='adu', meta={'EXPTIME': science_exp_time_val, 'FILTER': science_filter})
        except KeyError:
             print(f"Warning: EXPTIME or FILTER missing from {cluster_raw_file} header.")
             # Use assumed values cautiously
             science_exposure_time = science_exp_time_val * u.s
             science_filter = filter_name

        # --- Perform Full Reduction Sequence ---
        # 1. Bias Subtraction
        print("Subtracting bias...")
        cluster_bias_sub = ccdproc.subtract_bias(cluster_raw, master_bias)

        # 2. Dark Subtraction
        # Check if scaling is needed
        scale_dark_needed = not np.isclose(science_exposure_time.value, dark_exposure_time.value)
        if scale_dark_needed:
            print(f"Note: Science exposure {science_exposure_time} != dark exposure {dark_exposure_time}. Scaling dark.")
        print("Subtracting dark...")
        cluster_dark_sub = ccdproc.subtract_dark(
            cluster_bias_sub, master_dark,
            dark_exposure=dark_exposure_time, data_exposure=science_exposure_time,
            exposure_unit=u.s, scale=scale_dark_needed
        )

        # 3. Flat Correction
        # Check filter consistency
        if science_filter != flat_filter:
            raise ValueError(f"Filter mismatch: Science={science_filter}, Flat={flat_filter}")
        print("Applying flat correction...")
        cluster_flat_corr = ccdproc.flat_correct(cluster_dark_sub, master_flat)

        # Add history
        cluster_flat_corr.meta['HISTORY'] = f'Bias subtracted using {os.path.basename(master_bias_file)}'
        cluster_flat_corr.meta['HISTORY'] = f'Dark subtracted using {os.path.basename(master_dark_file)} (Scaled: {scale_dark_needed})'
        cluster_flat_corr.meta['HISTORY'] = f'Flat corrected using {os.path.basename(master_flat_file)}'

        # (Optional: Gain correction here if needed)

        # Save the final reduced frame (optional - commented out for dummy data)
        # cluster_flat_corr.write(output_file, overwrite=True)
        print(f"Stellar reduction complete (Bias, Dark, Flat). Result mean: {np.mean(cluster_flat_corr.data):.1f}")
        print(f"(If successful, result would normally be saved to {output_file})")


    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the Stellar example: {e}")
else:
     print("Skipping Stellar example: ccdproc unavailable.")

```

This code segment implements the canonical three-step reduction process for a typical stellar field observation, often requiring significant exposure times. It begins by loading the necessary master calibration frames (`master_bias`, `master_dark` corresponding to the science exposure time, and `master_flat` for the correct filter) and the raw science image, all represented as `CCDData` objects. The reduction proceeds sequentially using `ccdproc`: `ccdproc.subtract_bias` removes the bias, `ccdproc.subtract_dark` removes the thermal signal (checking for and applying scaling if exposure times don't match), and `ccdproc.flat_correct` corrects for pixel sensitivity and illumination effects using the appropriate filter flat. Each step updates the data (and potentially uncertainty and mask attributes) of the `CCDData` object, and `HISTORY` entries are added to the metadata. The final output is a fully reduced science frame, cleaned of the dominant static instrumental signatures and ready for analysis like photometry of the cluster stars.

**3.10.4 Exoplanetary: Cosmic Ray Removal from Transit Image**
High-precision photometry, such as that required for detecting the shallow dips caused by transiting exoplanets, is highly sensitive to image artifacts. Cosmic rays hitting the detector during an exposure can mimic transient astrophysical signals or significantly corrupt photometric measurements if they fall on the target star or within the photometric aperture. Therefore, identifying and mitigating cosmic rays is a crucial step, especially for single exposures or when median combination of dithered frames is not feasible. This example demonstrates the application of the L.A.Cosmic algorithm, implemented in `astroscrappy` and accessible via `ccdproc`, to identify cosmic ray hits in a (previously bias-, dark-, and flat-corrected) image intended for transit photometry.

```python
import numpy as np
from astropy.nddata import CCDData, CCDDataProcessingError
import astropy.units as u
# Requires ccdproc and its dependency astroscrappy:
# pip install ccdproc astroscrappy
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Exoplanetary example.")
    ccdproc_available = False
from astropy.io import fits # For dummy file creation
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
# Assume input is reduced (bias, dark, flat corrected), units might be electrons
transit_image_file = 'transit_image_reduced.fits'
output_file = 'transit_image_cr_cleaned.fits' # Image with CRs masked

# Create dummy file if it doesn't exist
if ccdproc_available:
    if not os.path.exists(transit_image_file):
        print(f"Creating dummy file: {transit_image_file}")
        # Simulate a star field + background noise + CR
        im_size = (80, 80)
        data = np.random.normal(loc=100.0, scale=3.0, size=im_size).astype(np.float32) # Background + read noise contribution in e-
        # Add a star PSF
        yy, xx = np.indices(im_size)
        psf_sigma = 2.0
        star_flux = 5000.0
        data += star_flux * np.exp(-0.5 * (((xx - 40)/psf_sigma)**2 + ((yy - 40)/psf_sigma)**2))
        # Add a fake cosmic ray hit (sharp)
        cr_loc_y, cr_loc_x = 20, 60
        cr_flux = 300.0
        data[cr_loc_y, cr_loc_x] = cr_flux
        data[cr_loc_y+1, cr_loc_x] = cr_flux * 0.8 # Simulate small trail
        # Create header with gain/readnoise needed by L.A.Cosmic
        hdr = fits.Header()
        hdr['GAIN'] = (1.5, 'e-/ADU (Example, even if data in e-)')
        hdr['RDNOISE'] = (5.0, 'Read noise in electrons')
        hdr['BUNIT'] = 'electron'
        # Create CCDData object and write to file
        ccd = CCDData(data, unit='electron', meta=hdr, mask=np.zeros_like(data, dtype=bool))
        ccd.write(transit_image_file, overwrite=True)

if ccdproc_available:
    try:
        # Load the reduced science image (bias, dark, flat corrected).
        # Assume data might be in electrons, but gain/readnoise info is needed.
        print(f"Loading reduced transit image: {transit_image_file}")
        try:
             # Read gain/readnoise from header, provide defaults if missing
             hdr = fits.getheader(transit_image_file)
             gain_val = hdr.get('GAIN', 1.0) # Default gain 1 if missing
             readnoise_val = hdr.get('RDNOISE', 5.0) # Default readnoise 5 if missing
             # Read into CCDData, assuming units from header or specify
             transit_image_reduced = CCDData.read(transit_image_file, unit=hdr.get('BUNIT', 'electron'))
             # Ensure gain and readnoise have units for cosmicray_lacosmic
             gain = gain_val * u.electron / u.adu
             readnoise = readnoise_val * u.electron
        except FileNotFoundError:
             print(f"Warning: File {transit_image_file} not found, using dummy data.")
             gain = 1.5 * u.electron / u.adu
             readnoise = 5.0 * u.electron
             data = np.random.normal(loc=100.0, scale=3.0, size=(80, 80))
             data[20, 60] = 300 # Fake CR
             transit_image_reduced = CCDData(data, unit='electron', mask=np.zeros_like(data, dtype=bool))


        print(f"Applying L.A.Cosmic for cosmic ray rejection (Gain={gain}, RN={readnoise})...")
        # Apply cosmic ray rejection using L.A.Cosmic algorithm via ccdproc.
        # This function typically modifies the `.mask` attribute of the input
        # CCDData object, flagging pixels identified as cosmic rays.
        # It requires gain and readnoise values (with units) for its noise model.
        # sigclip: Threshold for initial detection based on Laplacian significance.
        # objlim: Contrast limit to distinguish CRs from bright object cores.
        # Setting cleantype='medmask' replaces CR pixels with local median AND masks them.
        # Using cleantype='none' or the default just updates the mask.
        cr_cleaned_image = ccdproc.cosmicray_lacosmic(
            transit_image_reduced,
            gain=gain,
            readnoise=readnoise,
            sigclip=4.5,      # Laplacian significance threshold
            objlim=5.0,       # Contrast limit compared to neighbors
            cleantype='none', # Only update the mask, do not replace pixel values
            verbose=True      # Print number of iterations and CRs found
        )

        # Check the number of pixels newly masked by the algorithm.
        num_cr_pixels = np.sum(cr_cleaned_image.mask) - np.sum(transit_image_reduced.mask) # Sum over boolean mask gives count
        print(f"\nIdentified and masked {num_cr_pixels} cosmic ray pixels.")
        if num_cr_pixels > 0:
             print(f"Masked pixel indices (first 10): {np.argwhere(cr_cleaned_image.mask)[:10].tolist()}")

        # Add history record.
        cr_cleaned_image.meta['HISTORY'] = f'Cosmic rays masked using L.A.Cosmic (sigclip=4.5, objlim=5.0)'

        # Save the image with the updated mask (optional - commented out for dummy data).
        # The data array itself is unchanged if cleantype='none'.
        # cr_cleaned_image.write(output_file, overwrite=True)
        print(f"(If successful, image with updated mask would be saved to {output_file})")


    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except ImportError:
         print("Error: astroscrappy (dependency for cosmicray_lacosmic) not found.")
    except Exception as e:
        print(f"An unexpected error occurred in the Exoplanetary CR example: {e}")
else:
     print("Skipping Exoplanetary CR example: ccdproc unavailable.")

```

This script addresses the critical task of cosmic ray identification in an image intended for exoplanet transit photometry, where precise flux measurements are paramount. It loads a science image assumed to have already undergone bias, dark, and flat correction, reading it into a `CCDData` object. The core of the example is the call to `ccdproc.cosmicray_lacosmic`, which implements the robust L.A.Cosmic algorithm. This function requires estimates of the detector gain and read noise (extracted from the header or provided) to build an accurate noise model. Key parameters like `sigclip` (threshold for Laplacian significance) and `objlim` (contrast threshold relative to neighbors) are set to typical values, controlling the sensitivity and specificity of the detection. The `cleantype='none'` argument ensures that identified cosmic ray pixels are only flagged in the `.mask` attribute of the output `CCDData` object, rather than being replaced by interpolated values, preserving the original data while marking affected pixels for exclusion during subsequent photometric analysis. The number of newly masked pixels is reported, providing immediate feedback on the algorithm's action.

**3.10.5 Galactic: Creating Master Flat for Milky Way Imaging**
Wide-field imaging of the Milky Way often aims to capture faint, extended structures like nebulae or stellar streams, requiring excellent flat-fielding to remove large-scale gradients and pixel-to-pixel variations that could otherwise obscure these features. Dome flats are a common source for calibration in this context. This example simulates the process of creating a master flat frame specifically for a narrowband filter (like H-alpha, common for Galactic imaging) from a series of individual dome flat exposures. It includes the necessary steps of bias and dark subtraction (assuming an appropriate master dark is available or dark current is negligible for the flat exposure time) before robustly combining the processed flats and normalizing the result to create the final master flat.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, CCDDataProcessingError
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Galactic example.")
    ccdproc_available = False
import astropy.units as u
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
n_flats = 15
filter_name = 'Halpha'
flat_exp_time_val = 20.0 # seconds
dome_flat_files = [f'domeflat_{i+1:03d}_{filter_name}.fits' for i in range(n_flats)]
master_bias_file = 'master_bias.fits'
# Assume dark current is low for this exposure time, use a zero dark or short dark master
master_dark_file_for_flats = f'master_dark_short.fits' # e.g., a 5s dark might suffice if dark ~linear
dark_for_flat_exposure_val = 5.0 # seconds (example)
output_master_flat_file = f'master_flat_{filter_name}.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    # Master Bias
    if not os.path.exists(master_bias_file):
        print(f"Creating dummy file: {master_bias_file}")
        fits.PrimaryHDU((np.ones((200, 200)) * 505).astype(np.float32)).writeto(master_bias_file, overwrite=True)
    # Master Dark (short exposure example)
    if not os.path.exists(master_dark_file_for_flats):
        print(f"Creating dummy file: {master_dark_file_for_flats}")
        hdr = fits.Header({'EXPTIME': dark_for_flat_exposure_val, 'BUNIT':'adu'})
        fits.PrimaryHDU(np.random.normal(2, 0.5, size=(200, 200)).astype(np.float32), header=hdr).writeto(master_dark_file_for_flats, overwrite=True)
    # Dome Flat files
    for fname in dome_flat_files:
        if not os.path.exists(fname):
            print(f"Creating dummy file: {fname}")
            # Simulate bias + flat signal + noise + dust motes
            bias_level = 505
            flat_level = 30000
            read_noise = 4
            # Add dust motes (dark spots)
            yy, xx = np.indices((200, 200))
            dust_mask = np.ones((200, 200))
            dust_locs = [(50, 60, 8), (150, 120, 5)] # y, x, radius
            for y, x, r in dust_locs:
                dist_sq = (xx - x)**2 + (yy - y)**2
                dust_mask[dist_sq < r**2] = 0.7 # Reduce sensitivity
            # Pixel variation
            prnu = np.random.normal(1.0, 0.015, size=(200, 200))
            # Combine
            raw_signal = flat_level * dust_mask * prnu
            raw_noisy_signal = bias_level + np.random.poisson(raw_signal) + np.random.normal(0, read_noise, size=(200, 200))
            hdr = fits.Header({'EXPTIME': flat_exp_time_val, 'FILTER': filter_name})
            fits.PrimaryHDU(raw_noisy_signal.astype(np.float32), header=hdr).writeto(fname, overwrite=True)

if ccdproc_available:
    try:
        # --- Load Master Bias and appropriate Master Dark ---
        print(f"Loading master bias: {master_bias_file}")
        master_bias = CCDData.read(master_bias_file, unit='adu')

        print(f"Loading master dark: {master_dark_file_for_flats}")
        master_dark_for_flats = CCDData.read(master_dark_file_for_flats)
        if master_dark_for_flats.unit is None: master_dark_for_flats.unit = 'adu'
        try:
            dark_exposure_time_master = master_dark_for_flats.header['EXPTIME'] * u.s
        except KeyError:
            print(f"Warning: EXPTIME missing in {master_dark_file_for_flats} header. Cannot scale dark.")
            # Assume dark is negligible or handle error
            dark_exposure_time_master = 0 * u.s # Assume zero dark if header missing
            master_dark_for_flats = CCDData(np.zeros_like(master_bias.data), unit='adu', meta=master_dark_for_flats.meta)

        flat_exposure_time = flat_exp_time_val * u.s
        scale_dark_needed = not np.isclose(flat_exposure_time.value, dark_exposure_time_master.value)
        if scale_dark_needed:
            if dark_exposure_time_master.value == 0:
                 print("Master dark exposure is zero, cannot scale. Assuming negligible dark.")
                 scale_dark_needed = False # Force no scaling if dark is zero
            else:
                 print(f"Note: Flat exposure {flat_exposure_time} != dark exposure {dark_exposure_time_master}. Scaling dark.")

        # --- Process and Combine Dome Flats ---
        print(f"Processing {n_flats} dome flats for filter {filter_name}...")
        processed_flats_list = []
        for f in dome_flat_files:
            try:
                flat_ccd = CCDData.read(f, unit='adu')
                if flat_ccd.header.get('FILTER', '') != filter_name: continue # Skip wrong filter
                current_flat_exp = flat_ccd.header.get('EXPTIME', -1) * u.s
                if not np.isclose(current_flat_exp.value, flat_exposure_time.value): continue # Skip wrong exposure

                flat_bias_sub = ccdproc.subtract_bias(flat_ccd, master_bias)
                # Use the loaded (potentially zero) master dark for flats
                flat_proc = ccdproc.subtract_dark(
                    flat_bias_sub, master_dark_for_flats,
                    dark_exposure=dark_exposure_time_master,
                    data_exposure=current_flat_exp,
                    exposure_unit=u.s, scale=scale_dark_needed
                )
                processed_flats_list.append(flat_proc)
            except Exception as read_err:
                print(f"Warning: Error processing flat file {f}: {read_err}. Skipping.")

        if not processed_flats_list:
            raise ValueError(f"No valid flat frames processed for filter {filter_name}.")

        print(f"Combining {len(processed_flats_list)} processed flats...")
        flat_combiner = ccdproc.Combiner(processed_flats_list)
        # Sigma clipping is often useful for flats to reject cosmic rays or defects
        flat_combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median, dev_func=np.ma.std)
        combined_flat = flat_combiner.average_combine() # Average combine often preferred for flats to maximize SNR

        # --- Normalize and Save Master Flat ---
        # Use nanmedian for robustness against masked pixels from clipping
        median_val = np.nanmedian(combined_flat.data)
        if median_val is None or np.isclose(median_val, 0):
            raise ValueError("Median of combined flat is invalid. Cannot normalize.")
        print(f"Normalizing combined flat by median value: {median_val:.2f}")
        master_flat_Ha = combined_flat.divide(median_val)
        master_flat_Ha.meta['FILTER'] = filter_name
        master_flat_Ha.meta['NCOMBINE'] = len(processed_flats_list)
        master_flat_Ha.meta['NORMVAL'] = (median_val, 'Median value used for normalization')

        # Save the master flat
        master_flat_Ha.write(output_master_flat_file, overwrite=True)
        print(f"Galactic imaging: Master flat for filter {filter_name} created and saved to {output_master_flat_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the Galactic example: {e}")
else:
     print("Skipping Galactic example: ccdproc unavailable.")
```

This script focuses on the specific task of generating a high-quality master flat-field frame, essential for accurate analysis of wide-field Galactic imagery, particularly in narrowband filters like H-alpha. It simulates the workflow using dome flat exposures. The code first loads the required master bias and an appropriate master dark (potentially a short-exposure dark if scaling is reliable or assuming negligible dark for the flat exposure time). It then iteratively processes each raw dome flat file: reading it as a `CCDData` object, applying bias and dark corrections using `ccdproc` functions (including logic for scaling the dark if exposure times mismatch). The processed flats are collected and then combined using `ccdproc.Combiner`, employing sigma clipping to reject outliers and typically using average combination to maximize the signal-to-noise ratio in the final flat. Finally, the crucial normalization step is performed by dividing the combined flat by its median value, producing the master flat with values fluctuating around unity, which is then saved to a FITS file, ready for application to science images taken with the same filter.

**3.10.6 Extragalactic: Combining Dithered Galaxy Exposures**
Deep imaging of faint extragalactic targets often involves taking multiple exposures with small spatial offsets (dithering) between them. This strategy serves multiple purposes: covering inter-chip gaps in mosaic detectors, improving the spatial sampling of the point spread function, and, critically for reduction, enabling robust cosmic ray rejection during image combination. After individual exposures are processed through bias, dark, and flat corrections, they need to be aligned to a common reference frame (using WCS information or detected sources) before being combined. This example demonstrates the final combination step using `ccdproc.Combiner`, assuming the input frames have already been reduced and *aligned*. Using median combination inherently rejects cosmic rays, which appear as outliers at a given sky position in the stack of aligned frames.

```python
import numpy as np
from astropy.nddata import CCDData, CCDDataProcessingError
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping Extragalactic example.")
    ccdproc_available = False
from astropy.io import fits # For dummy file creation
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
n_exp = 5 # Number of dithered exposures
filter_name = 'F814W'
# Assume these files are already reduced and ALIGNED to a common WCS
input_files = [f'galaxy_frame_{i+1:02d}_aligned_redux.fits' for i in range(n_exp)]
output_file = 'galaxy_combined_image.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    for i, fname in enumerate(input_files):
        if not os.path.exists(fname):
            print(f"Creating dummy file: {fname}")
            # Simulate background + faint galaxy + CR in some frames
            im_size = (100, 100)
            background = np.random.normal(50, 5, size=im_size)
            # Faint galaxy
            yy, xx = np.indices(im_size)
            galaxy_signal = 30 * np.exp(-0.5 * (((xx - 50)/15)**2 + ((yy - 50)/20)**2))
            data = background + galaxy_signal
            # Add a CR to frames 1 and 3 (indices 0 and 2) at different locations
            if i == 0:
                data[30, 70] = 500 # CR hit
            if i == 2:
                data[60, 25] = 600 # Another CR hit
            # Create CCDData object and write
            hdr = fits.Header({'FILTER': filter_name, 'BUNIT': 'electron'})
            ccd = CCDData(data.astype(np.float32), unit='electron', meta=hdr)
            ccd.write(fname, overwrite=True)


if ccdproc_available:
    try:
        # --- Load the ALIGNED, reduced individual exposures ---
        # It is CRITICAL that these frames are spatially aligned before combination.
        # Alignment is a separate step (e.g., using WCS, reproject, or source matching)
        # typically done between basic reduction and combination.
        print(f"Loading {n_exp} aligned, reduced galaxy frames...")
        galaxy_frames = []
        for f in input_files:
            try:
                # Assuming data is now in physically meaningful units like electrons or electrons/s
                ccd = CCDData.read(f) # Read units from header if present
                if ccd.unit is None: ccd.unit = 'electron' # Assume electrons if unit missing
                galaxy_frames.append(ccd)
            except FileNotFoundError:
                print(f"Warning: Input file {f} not found. Skipping.")
            except Exception as read_err:
                print(f"Warning: Error reading file {f}: {read_err}. Skipping.")

        if not galaxy_frames:
            raise ValueError("No valid input frames loaded for combination.")
        if len(galaxy_frames) < 3:
             print("Warning: Combining fewer than 3 frames provides limited cosmic ray rejection with median.")

        # --- Combine using Combiner ---
        print(f"Combining {len(galaxy_frames)} frames using median (with sigma clipping)...")
        combiner = ccdproc.Combiner(galaxy_frames)

        # Apply sigma clipping for additional outlier rejection before median.
        # This helps reject pixels that might be bad in multiple frames or have unusual noise.
        combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median, dev_func=np.ma.std)

        # Use median combination. The median is robust against outliers like
        # cosmic rays, assuming they occur at different pixel locations in the aligned stack.
        combined_galaxy_image = combiner.median_combine()

        # Update metadata
        combined_galaxy_image.meta['NCOMBINE'] = len(galaxy_frames)
        combined_galaxy_image.meta['COMBTYPE'] = 'Median (Sigma Clip)'
        combined_galaxy_image.meta['HISTORY'] = f'Combined {len(galaxy_frames)} aligned reduced exposures.'

        # Save the final combined image (optional - commented out for dummy data)
        # combined_galaxy_image.write(output_file, overwrite=True)
        print(f"Extragalactic combination complete. Result mean: {np.mean(combined_galaxy_image.data):.1f}")
        # Check CR locations in combined image (should be closer to background)
        cr1_val = combined_galaxy_image.data[30, 70]
        cr2_val = combined_galaxy_image.data[60, 25]
        print(f"Value at dummy CR1 location: {cr1_val:.1f} (Expected ~background)")
        print(f"Value at dummy CR2 location: {cr2_val:.1f} (Expected ~background)")
        print(f"(If successful, combined image would be saved to {output_file})")
        print("NOTE: Assumes input frames were accurately aligned prior to this script.")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except CCDDataProcessingError as e:
        print(f"Error during CCD processing: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the Extragalactic combination example: {e}")
else:
     print("Skipping Extragalactic example: ccdproc unavailable.")

```

This script demonstrates the crucial step of combining multiple, individually reduced, and spatially aligned exposures of an extragalactic target, a standard technique for achieving deep images and removing transient artifacts. It loads the pre-processed and aligned frames as `CCDData` objects. The core operation utilizes `ccdproc.Combiner` initialized with this list of frames. Sigma clipping is first applied via `combiner.sigma_clipping` to identify and flag pixels that are outliers across the stack even before the final combination, providing an extra layer of robustness. The final combination is performed using `combiner.median_combine`. Because cosmic rays affect different pixels in dithered, aligned frames, they appear as significant positive outliers in the stack of values at a given sky position and are effectively rejected by the median calculation. The resulting `combined_galaxy_image` represents a deeper view of the target with significantly reduced cosmic ray contamination, ready for scientific analysis like galaxy morphology studies or photometry of faint sources. Accurate prior alignment of the input frames is critical for this process to work correctly.

**3.10.7 Cosmology: Basic Cleaning of a Survey Tile**
Large cosmological imaging surveys, like DES, HSC, or the upcoming LSST, generate vast mosaics of the sky. Analyzing these images for subtle signals like weak gravitational lensing shear or detecting faint galaxy clusters requires careful image preparation beyond basic instrumental calibration. This often involves identifying and masking regions contaminated by bright stars, saturated pixels, satellite trails, image defects, and other artifacts that could bias measurements. This example simulates a basic cleaning step for such a survey tile, focusing on masking problematic pixels using a combination of a pre-existing bad pixel mask (BPM) and automated detection of bright sources using the `photutils` library.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
# photutils provides tools for source detection and background estimation
# Ensure installed: pip install photutils
try:
    from photutils.segmentation import detect_sources, make_source_mask
    from photutils.background import Background2D, MedianBackground
    photutils_available = True
except ImportError:
    print("photutils not found, skipping Cosmology example.")
    photutils_available = False
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
# Assume input is reduced science image tile
survey_tile_file = 'survey_tile_reduced.fits'
# Assume an existing Bad Pixel Mask file
bpm_file = 'survey_tile_bpm.fits'
# Output: FITS file containing the combined mask
output_mask_file = 'survey_tile_cleaned_mask.fits'
# Output: FITS file with updated mask in CCDData object (optional)
output_masked_ccd_file = 'survey_tile_masked.fits'


# Create dummy files if they don't exist
if photutils_available:
    # Reduced survey tile
    if not os.path.exists(survey_tile_file):
        print(f"Creating dummy file: {survey_tile_file}")
        im_size = (200, 200)
        # Simulate background + faint sources + bright stars
        background = np.random.normal(50.0, 5.0, size=im_size)
        data = background
        # Add faint sources
        n_faint = 100
        x_faint = np.random.uniform(0, im_size[1], n_faint)
        y_faint = np.random.uniform(0, im_size[0], n_faint)
        flux_faint = np.random.uniform(20, 50, n_faint)
        yy, xx = np.indices(im_size)
        psf_sigma = 1.5
        for x, y, flux in zip(x_faint, y_faint, flux_faint):
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Add bright stars that should be masked
        bright_stars = [(30, 40, 5000), (150, 160, 8000), (100, 80, 6000)]
        for y, x, flux in bright_stars:
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Create CCDData object and write
        hdr = fits.Header({'FILTER': 'i', 'BUNIT': 'ADU'})
        ccd = CCDData(data.astype(np.float32), unit='adu', meta=hdr)
        ccd.write(survey_tile_file, overwrite=True)
    # Bad Pixel Mask
    if not os.path.exists(bpm_file):
        print(f"Creating dummy file: {bpm_file}")
        bpm_data = np.zeros((200, 200), dtype=np.uint8)
        bpm_data[180:, 180:] = 1 # Mask a corner region
        bpm_data[50:55, :] = 1 # Mask some rows
        fits.PrimaryHDU(bpm_data).writeto(bpm_file, overwrite=True)

if photutils_available:
    try:
        # --- Load Reduced Survey Tile and Existing BPM ---
        print(f"Loading reduced survey tile: {survey_tile_file}")
        try:
            survey_tile_ccd = CCDData.read(survey_tile_file) # Read units from header or set default
            if survey_tile_ccd.unit is None: survey_tile_ccd.unit = 'adu'
        except FileNotFoundError:
             print(f"Warning: File {survey_tile_file} not found, using dummy data.")
             survey_tile_ccd = CCDData(np.random.normal(50, 5, size=(200, 200)), unit='adu')


        print(f"Loading bad pixel mask: {bpm_file}")
        try:
            bpm = fits.getdata(bpm_file).astype(bool) # Ensure mask is boolean
        except FileNotFoundError:
             print(f"Warning: File {bpm_file} not found, creating zero mask.")
             bpm = np.zeros(survey_tile_ccd.shape, dtype=bool)

        # Apply the initial BPM to the CCDData mask attribute
        survey_tile_ccd.mask = bpm
        print(f"Applied initial BPM, masking {np.sum(bpm)} pixels.")

        # --- Detect and Mask Bright Sources using photutils ---
        print("Detecting sources to mask...")
        # Estimate the 2D background and background noise (RMS)
        # Use robust estimators like MedianBackground and mask existing bad pixels
        try:
            # sigma_clip used within Background2D for robustness
            bkg_estimator = MedianBackground()
            bkg = Background2D(survey_tile_ccd.data, (32, 32), filter_size=(3, 3), # Adjust box/filter size as needed
                               mask=survey_tile_ccd.mask, bkg_estimator=bkg_estimator)
            background_map = bkg.background
            # Use background_rms_median for robustness against outliers in RMS map
            background_rms_map = bkg.background_rms_median
            print(f"Background estimated. Mean bkg={np.mean(background_map):.1f}, Mean RMS={np.mean(background_rms_map):.1f}")
        except Exception as bkg_err: # Handle potential errors in background estimation
            print(f"Warning: Background2D failed ({bkg_err}), using simple global background.")
            valid_pix = survey_tile_ccd.data[~survey_tile_ccd.mask]
            background_map = np.nanmedian(valid_pix)
            background_rms_map = np.nanstd(valid_pix)
            print(f"Using global median background: {background_map:.1f}, RMS: {background_rms_map:.1f}")


        # Detect sources significantly above the local background noise
        # threshold is typically N-sigma above the background RMS map
        detection_threshold = 5.0 * background_rms_map
        # Subtract background before detection
        data_subtracted = survey_tile_ccd.data - background_map
        # Use detect_sources to find connected pixels above threshold
        # npixels specifies the minimum number of connected pixels to be considered a source
        print(f"Detecting sources above 5 sigma threshold...")
        segment_map = detect_sources(data_subtracted, detection_threshold, npixels=5,
                                     mask=survey_tile_ccd.mask) # Provide mask to ignore bad pixels


        final_mask = survey_tile_ccd.mask # Start with the original BPM
        if segment_map:
            print(f"Detected {segment_map.nlabels} sources.")
            # Create a boolean mask covering the pixels belonging to detected sources
            source_mask = make_source_mask(segment_map, mask=survey_tile_ccd.mask)
            # Optional: Grow the source mask slightly to cover faint halos
            # from photutils.segmentation import SourceCatalog
            # from astropy.convolution import Gaussian2DKernel, convolve
            # kernel = Gaussian2DKernel(x_stddev=2) # Example kernel for growing
            # source_mask_grown = convolve(source_mask.astype(float), kernel) > 0.1 # Threshold convolution
            # source_mask = source_mask_grown

            # Combine the initial BPM with the new source mask
            # Use logical OR: mask if bad OR part of detected bright source
            final_mask = survey_tile_ccd.mask | source_mask
            num_newly_masked = np.sum(final_mask) - np.sum(survey_tile_ccd.mask)
            print(f"Created source mask, adding {num_newly_masked} newly masked pixels.")
        else:
            print("No significant sources detected above threshold to add to mask.")

        # Update the mask attribute of the CCDData object
        survey_tile_ccd.mask = final_mask
        print(f"Final combined mask applied. Total masked pixels: {np.sum(final_mask)}")

        # Save the combined mask to a separate FITS file (common practice)
        # Convert boolean mask to integer (0=good, 1=bad) for FITS saving
        mask_hdu = fits.PrimaryHDU(final_mask.astype(np.uint8))
        mask_hdu.header['MASKTYPE'] = 'BPM+SOURCES'
        # mask_hdu.writeto(output_mask_file, overwrite=True) # Don't write dummy data
        print(f"(If successful, combined mask would be saved to {output_mask_file})")

        # Optionally save the CCDData object with the updated mask
        # survey_tile_ccd.write(output_masked_ccd_file, overwrite=True)

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except ImportError:
         print("Error: photutils is required for this example but not found.")
    except Exception as e:
        print(f"An unexpected error occurred in the Cosmology example: {e}")
else:
     print("Skipping Cosmology example: photutils unavailable or dummy data missing.")

```

This final script addresses a common preliminary step in analyzing large cosmological survey images: identifying and masking regions unsuitable for sensitive measurements like weak lensing. It begins by loading a reduced survey image tile and a pre-existing bad pixel mask (BPM) into a `CCDData` object, applying the BPM to the `.mask` attribute. The core of the cleaning process utilizes the `photutils` library. First, `Background2D` is employed to create robust 2D estimates of the sky background and its RMS variation across the tile, accounting for the initial mask. Then, `detect_sources` identifies contiguous pixels significantly above the local background RMS (`threshold`), effectively segmenting bright stars and galaxies. The `make_source_mask` function converts this segmentation map into a boolean mask covering these detected bright objects. Finally, this source mask is combined (using logical OR) with the original BPM to create a `final_mask`, which flags pixels that are either inherently bad or contaminated by bright sources. This final mask is then typically saved separately or updated in the science file's mask attribute, ensuring these regions are excluded from subsequent cosmological analyses.

---

**References**

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project. The `astropy.nddata` subpackage (Section 3.2) and the affiliated `ccdproc` package (Section 3.9) are fundamental to the data structures and processing functions central to this chapter.

Craig, M. W., Crawford, S. M., Fowler, J., Tollerud, E. J., Greenfield, P., Droettboom, M., & Astropy Collaboration. (2017). ccdproc: CCD data reduction package. *Astrophysics Source Code Library*, record ascl:1708.009. https://ui.adsabs.harvard.edu/abs/2017ascl.soft08009C/abstract *(Note: ASCL entry)*
*   *Summary:* The ASCL entry for `ccdproc`, the primary software library utilized in this chapter's examples for performing bias subtraction, dark correction, flat fielding, and combining calibration frames (Sections 3.3, 3.4, 3.5, 3.6, 3.9, 3.10). Essential software reference.

Gillessen, S., Noderer, T., Quataert, E., Dexter, J., Pfuhl, O., Waisberg, I., Eisenhauer, F., Ott, T., Genzel, R., Habibi, M., Perrin, G., Brandner, W., Straubmeier, C., & Clénet, Y. (2020). Dynamic modeling of the Galactic center source G2. *The Astrophysical Journal, 901*(2), 116. https://doi.org/10.3847/1538-4357/abaf06
*   *Summary:* This paper, while focused on astrophysics, employs sophisticated data reduction techniques, including robust combination of calibration frames conceptually similar to methods discussed for creating master darks and flats (Sections 3.4, 3.5, 3.6). Illustrates general principles in practice.

Greenhouse, M., Egami, E., Dickinson, M., Finkelstein, S., Arribas, S., Ferruit, P., Giardino, G., Pirzkal, N., & Willott, C. (2023). The James Webb Space Telescope mission: Design reference information. *Publications of the Astronomical Society of the Pacific, 135*(1049), 078001. https://doi.org/10.1088/1538-3873/acdc58
*   *Summary:* Provides reference information on JWST instruments and their characteristics. The JWST pipeline handles corrections for effects analogous to bias, dark current, and flat-field variations (Section 3.1) for its IR detectors, offering a modern context for sophisticated calibration needs.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
*   *Summary:* The foundational paper for NumPy, which provides the `ndarray` class underpinning the image data representation discussed in Section 3.2 and used by `CCDData` and `ccdproc`.

Jia, S., Zhang, Z., Wang, J., & Bloom, J. S. (2023). Self-Supervised Learning for Astronomical Image Cleaning. *arXiv preprint arXiv:2310.14929*. https://doi.org/10.48550/arXiv.2310.14929
*   *Summary:* This preprint presents a recent self-supervised deep learning approach for identifying various artifacts, including cosmic rays (Section 3.8). It represents the state-of-the-art in applying ML to artifact detection, contrasting with traditional methods like L.A.Cosmic.

McCully, C., Crawford, S. M., & Sipőcz, B. (2018). astroscrappy: cosmic ray rejection. *Astrophysics Source Code Library*, record ascl:1801.007. https://ui.adsabs.harvard.edu/abs/2018ascl.soft01007M/abstract *(Note: ASCL entry)*
*   *Summary:* The ASCL entry for `astroscrappy`, the Python implementation of the L.A.Cosmic algorithm used by `ccdproc` for cosmic ray rejection, as discussed in Section 3.8 and demonstrated in Section 3.10.4. Essential software reference.

Prochaska, J. X., Hennawi, J. F., Westfall, K. B., Cooke, R., Wang, F., Hsyu, T., & Emg, D. (2020). PypeIt: The Python Spectroscopic Data Reduction Pipeline. *Journal of Open Source Software, 5*(54), 2308. https://doi.org/10.21105/joss.02308
*   *Summary:* Introduces the PypeIt pipeline. While focused on spectroscopy, its handling of 2D detector data involves steps analogous to image reduction (bias, dark, flat, CR rejection) discussed in this chapter, demonstrating common principles.

Rauscher, B. J. (2021). Fundamental limits to the calibration of astronomical detectors. *Journal of Astronomical Telescopes, Instruments, and Systems, 7*(4), 046001. https://doi.org/10.1117/1.JATIS.7.4.046001
*   *Summary:* Discusses physical limits on detector calibration accuracy, including non-linearity (Section 3.1) and other effects impacting the fidelity achievable through bias, dark, and flat correction steps (Sections 3.3, 3.4, 3.5).

Stefanescu, R.-A., Ré, P. M., Ferreira, L., Sousa, R., Amorim, A., Fernandes, C., Gaspar, M., & Correia, C. M. B. A. (2020). High-Performance CMOS Image Sensors for Scientific Imaging: Technology Development and Characterization. *Sensors, 20*(20), 5904. https://doi.org/10.3390/s20205904
*   *Summary:* Details performance characteristics of modern scientific CMOS sensors, providing context for the instrumental signatures (bias, dark, noise, PRNU - Section 3.1) encountered in data from these increasingly common detectors.

Vacca, W. D. (2021). Reduction and calibration of near-infrared spectra obtained with SpeX using Spextool. *Journal of Astronomical Telescopes, Instruments, and Systems, 7*(1), 018002. https://doi.org/10.1117/1.JATIS.7.1.018002
*   *Summary:* Describes the Spextool pipeline for reducing NIR spectra. The handling of calibration frames like flats and darks (Sections 3.4, 3.5, 3.6) shares principles with the image reduction techniques discussed in this chapter.

Weilbacher, P. M., Palsa, R., Streicher, O., Conseil, S., Bacon, R., Boogaard, L., Borisova, E., Brinchmann, J., Contini, T., Feltre, A., Guérou, A., Kollatschny, W., Krajnović, D., Maseda, M. V., Paalvast, M., Pécontal-Rousset, A., Pello, R., Richard, J., Roth, M. M., … Wisotzki, L. (2020). The MUSE data reduction pipeline. *Astronomy & Astrophysics, 641*, A28. https://doi.org/10.1051/0004-6361/202037985
*   *Summary:* Describes the reduction pipeline for the MUSE IFU. It details complex calibration procedures, including flat-fielding (Section 3.5) and bad pixel masking (Section 3.7) tailored for IFU data but illustrating principles applicable here.

Zhang, Z., & Bloom, J. S. (2020). DeepCR: Cosmic Ray Removal with Deep Learning. *The Astrophysical Journal, 889*(1), 49. https://doi.org/10.3847/1538-4357/ab6195
*   *Summary:* Presents a deep learning method for cosmic ray identification and removal, offering a modern machine learning alternative to algorithms like L.A.Cosmic discussed in Section 3.8. Indicates current research directions in artifact mitigation.
