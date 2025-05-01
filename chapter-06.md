Okay, here is the corrected Chapter 6, with the instructional paragraphs removed from Section 6.7, leaving only the proper introductory and concluding paragraphs around the code examples, while maintaining all other requirements.

---
# Chapter 6
# Image Analysis: Source Detection and Measurement
---

This chapter transitions from the preparatory stages of data reduction and calibration to the core scientific objective of extracting quantitative information from astronomical images. It provides a comprehensive overview of fundamental image analysis techniques essential for identifying celestial objects and measuring their basic physical properties, primarily focusing on source detection and photometric measurement. The discussion commences with methods for robustly estimating and subtracting the underlying image background or sky level, a critical prerequisite for accurate source characterization, employing techniques adaptable to varying background complexities. Subsequently, the chapter delves into algorithms designed for automated source detection, detailing methods based on thresholding relative to background noise and segmentation approaches that group connected pixels into distinct source entities, highlighting the functionality provided by standard libraries like `photutils`. Following detection, the principles and practices of aperture photometry are elaborated, explaining how to measure the integrated flux of sources within defined geometric apertures while carefully accounting for local background contributions. Recognizing the limitations of aperture photometry in crowded fields or for high-precision measurements, the chapter then introduces the more sophisticated technique of Point Spread Function (PSF) photometry, covering the essential steps of modeling the instrument's PSF and fitting this model to stellar images to derive accurate flux measurements and positions, particularly crucial in dense stellar environments. Basic morphological analysis techniques for characterizing the shapes, sizes, and orientations of detected sources are also briefly discussed. Finally, the chapter addresses the common requirement of cross-matching detected source catalogs with external astronomical databases based on celestial coordinates, a vital step for object identification, classification, and multi-wavelength studies. Throughout the chapter, practical implementations using the Python ecosystem, particularly the `photutils` and `astropy` libraries, are emphasized, culminating in illustrative examples demonstrating these techniques across diverse astronomical sub-disciplines.

---

**6.1 Background Estimation and Subtraction (`photutils`)**

Astronomical images invariably contain a background signal superimposed on the flux from celestial sources. This background arises from various sources, including diffuse emission from the Earth's atmosphere (airglow, scattered moonlight, light pollution for ground-based observations), zodiacal light, unresolved faint stars or galaxies contributing to a diffuse background glow, residual instrumental signatures (e.g., imperfectly subtracted bias or dark current), and thermal emission from the telescope and instrument, particularly significant in the infrared (Massey & Jacoby, 1992; Waters & Price, 2015). Accurately characterizing and subtracting this background is a critical prerequisite for nearly all subsequent image analysis tasks, particularly source detection and photometry. Failure to properly remove the background will lead to systematic errors in measured source fluxes and can significantly compromise the detection of faint objects. The background level is often not constant across an image; it can exhibit large-scale gradients due to scattered light or varying airglow, as well as smaller-scale fluctuations. Therefore, robust background estimation techniques are required.

The **`photutils`** package provides powerful tools for background estimation in Python (Bradley et al., 2023). A common approach involves dividing the image into a grid of smaller boxes or meshes and calculating robust statistical estimators of the background level within each box, thereby creating a 2D map representing the background variation across the image. The **`photutils.background.Background2D`** class implements this methodology effectively.

Key steps and considerations for 2D background estimation include:
1.  **Masking:** Pixels contaminated by actual astronomical sources must be excluded from the background calculation to avoid biasing the estimate upwards. This requires generating a mask that covers the sources. This mask can be created using preliminary source detection (Section 6.2), by thresholding the image significantly above the expected background noise, or by using iterative sigma-clipping within the background estimation algorithm itself. Pixels identified as bad in the Bad Pixel Mask (BPM, Section 3.7) should also be masked.
2.  **Grid Definition (`box_size`):** The image is divided into a grid of boxes. The size of these boxes (`box_size`) is a critical parameter. It must be large enough to contain sufficient "background" pixels for robust statistical estimation after source masking, but small enough to capture the scale of real background variations across the image. If the box size is too large, local background variations will be smoothed over; if too small (and heavily masked), the statistics within each box become poor. Typical box sizes might range from 20x20 to 100x100 pixels, depending on image size, source density, and background complexity.
3.  **Filtering (`filter_size`, `filter_threshold`):** Before calculating statistics within each box, `Background2D` can apply a filter (e.g., a median filter of size `filter_size`) to the grid of initial background estimates. This helps smooth the background map and reject outlier boxes that might be heavily contaminated by masked sources or defects. A `filter_threshold` can optionally exclude boxes where the filtered value differs too much from the original estimate.
4.  **Statistical Estimator (`bkg_estimator`, `bkgrms_estimator`):** Within each grid box (after filtering and masking source pixels), a robust statistical estimator is used to determine the local background level. Common choices provided by `photutils` include:
    *   `MedianBackground`: Calculates the median pixel value, which is robust against outliers (e.g., faint undetected sources or residual cosmic rays).
    *   `MeanBackground`: Calculates the mean, but requires effective sigma-clipping to remove outliers.
    *   `ModeEstimatorBackground` (e.g., using `MMMBackground` which implements a mode estimator): Attempts to find the peak of the pixel value distribution, often considered a good representation of the true background level in the presence of faint sources.
    Similarly, an estimator for the background noise (RMS) is calculated within each box (e.g., `StdBackground` with sigma clipping, `BiweightLocationBackground`, `MADStdBackground` which uses the Median Absolute Deviation for robust RMS estimation). `Background2D` typically performs iterative sigma clipping (`sigma_clip` attribute) within each box before applying the chosen estimators to remove outlier pixels.
5.  **Interpolation:** The background statistics calculated on the grid points are then interpolated (e.g., using bilinear or spline interpolation) back to the full image resolution, creating smooth 2D maps of the background level (`bkg.background`) and the background RMS noise (`bkg.background_rms`).

Once the 2D background map ($B(x, y)$) has been generated, it is subtracted pixel-by-pixel from the original science image ($I_{sci}(x, y)$) to produce a background-subtracted image ($I'_{sci}(x, y)$):
$I'_{sci}(x, y) = I_{sci}(x, y) - B(x, y)$
This background-subtracted image ideally has a mean level close to zero in source-free regions and represents the flux originating only from the celestial sources (plus noise). The background RMS map ($RMS_{bkg}(x, y)$) is crucial for subsequent source detection, as it quantifies the local noise level against which potential sources must be evaluated.

For simpler cases where the background is relatively flat across the image, a global scalar background value (e.g., the median or sigma-clipped mean of the entire image after source masking) might suffice. However, the 2D background estimation approach is generally more robust and recommended for accurate analysis, especially for wide-field images or images affected by gradients or complex structures. Careful choice of parameters (`box_size`, `filter_size`, estimators, sigma-clipping parameters) is necessary, often requiring some experimentation and visual inspection of the resulting background map and background-subtracted image to ensure satisfactory results.

The following Python code demonstrates using `photutils.background.Background2D` to estimate and subtract the 2D background from an astronomical image. It initializes the `Background2D` class with the image data, defines the grid box size and filter size, specifies robust estimators (Median and MAD-based standard deviation), and applies sigma clipping within the estimation. The resulting background map is then subtracted from the original image data.

```python
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip # For background estimation
# Requires photutils: pip install photutils
try:
    from photutils.background import Background2D, MedianBackground, MADStdBackground
    photutils_available = True
except ImportError:
    print("photutils not found, skipping background subtraction example.")
    photutils_available = False
import matplotlib.pyplot as plt # For visualization
import os # For dummy file creation/check

# --- Input/Output Files (using dummy data) ---
input_image_file = 'galaxy_field_image.fits' # Assume this is reduced data
output_bkgsub_file = 'galaxy_field_bkgsub.fits'
output_bkgmap_file = 'galaxy_field_bkgmap.fits' # Optional: save background map

# Create dummy file if it doesn't exist
if photutils_available:
    if not os.path.exists(input_image_file):
        print(f"Creating dummy file: {input_image_file}")
        im_size = (200, 200)
        # Simulate background gradient + sources
        yy, xx = np.indices(im_size)
        background_gradient = 50 + 0.1 * xx + 0.05 * yy # Simple linear gradient
        background_noise = np.random.normal(0, 5.0, size=im_size)
        data = background_gradient + background_noise
        # Add some sources
        n_src = 30
        x_src = np.random.uniform(0, im_size[1], n_src)
        y_src = np.random.uniform(0, im_size[0], n_src)
        flux_src = 10**(np.random.uniform(1.5, 3.0, n_src))
        psf_sigma = 1.8
        for x, y, flux in zip(x_src, y_src, flux_src):
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Create HDU and write
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['BUNIT'] = 'ADU'
        hdu.writeto(input_image_file, overwrite=True)

if photutils_available:
    try:
        # --- Load Image Data ---
        print(f"Loading image: {input_image_file}")
        try:
            image_data, header = fits.getdata(input_image_file, header=True)
            # Assume units or set a default if needed (not strictly required by Background2D)
            # image_unit = u.Unit(header.get('BUNIT', 'adu'))
        except FileNotFoundError:
            print(f"Error: File {input_image_file} not found. Cannot proceed.")
            exit()

        # --- Estimate 2D Background ---
        print("Estimating 2D background...")
        # Define sigma clipping object for robustness during estimation
        sigma_clip = SigmaClip(sigma=3.0, maxiters=5) # Clip at 3-sigma

        # Define robust background and RMS estimators
        bkg_estimator = MedianBackground() # Use Median for background level
        bkgrms_estimator = MADStdBackground() # Use Median Absolute Deviation for RMS

        # Define box size for gridding and filter size for smoothing grid
        box_size = (50, 50) # Size of grid boxes
        filter_size = (3, 3) # Size of median filter applied to grid points

        # Create Background2D instance
        # mask can be provided to exclude sources/bad pixels if known beforehand
        # exclude_percentile can be used for simple source masking internally
        bkg = Background2D(image_data, box_size, filter_size=filter_size,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                           bkgrms_estimator=bkgrms_estimator,
                           exclude_percentile=10.0 # Exclude top 10% pixels (crude source mask)
                          )

        # Access the calculated background map and RMS map
        background_map = bkg.background
        background_rms_map = bkg.background_rms

        print(f"Background estimation complete.")
        print(f"  Median Background Value: {bkg.background_median:.2f}")
        print(f"  Median Background RMS: {bkg.background_rms_median:.2f}")

        # --- Subtract Background ---
        print("Subtracting background map from image...")
        image_bkgsub = image_data - background_map

        # --- Save Results (Optional) ---
        # Save the background-subtracted image
        # hdu_bkgsub = fits.PrimaryHDU(data=image_bkgsub.astype(np.float32), header=header)
        # hdu_bkgsub.header['HISTORY'] = '2D Background Subtracted (photutils)'
        # hdu_bkgsub.writeto(output_bkgsub_file, overwrite=True)
        print(f"(If successful, background-subtracted image would be saved to {output_bkgsub_file})")

        # Save the background map itself for inspection
        # hdu_bkgmap = fits.PrimaryHDU(data=background_map.astype(np.float32), header=header)
        # hdu_bkgmap.header['HISTORY'] = '2D Background Map (photutils)'
        # hdu_bkgmap.writeto(output_bkgmap_file, overwrite=True)
        print(f"(If successful, background map would be saved to {output_bkgmap_file})")

        # --- Optional: Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(image_data, origin='lower', cmap='viridis', aspect='auto')
        axes[0].set_title('Original Image')
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(background_map, origin='lower', cmap='viridis', aspect='auto')
        axes[1].set_title('Background Map')
        fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(image_bkgsub, origin='lower', cmap='viridis', aspect='auto')
        axes[2].set_title('Background Subtracted')
        fig.colorbar(im2, ax=axes[2])
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Error: photutils library is required but not found.")
    except Exception as e:
        print(f"An unexpected error occurred during background estimation: {e}")
else:
     print("Skipping background subtraction example: photutils unavailable or dummy data missing.")

```

The Python script above effectively demonstrates the process of estimating and removing a potentially varying 2D background from an astronomical image using the `photutils` library. After loading the input image data, it initializes a `Background2D` object, specifying parameters like the `box_size` for gridding the image, the `filter_size` for smoothing the grid estimates, and robust statistical estimators (`MedianBackground`, `MADStdBackground`) along with `SigmaClip` to handle outlier pixels (like sources) within each box. The `Background2D` object computes `background_map` and `background_rms_map` representing the background level and its noise across the image. The core subtraction step simply involves subtracting the computed `background_map` from the original `image_data`. The resulting `image_bkgsub` represents the image with the background removed, making celestial sources more prominent against a near-zero background, suitable for subsequent source detection and photometry. The optional visualization clearly shows the original image, the derived smooth background map, and the final background-subtracted result.

**6.2 Source Detection Algorithms (`photutils.detection`)**

Once the image background has been estimated and subtracted, the next crucial step is to identify statistically significant detections that correspond to real astronomical sources (stars, galaxies, etc.), distinguishing them from residual background noise fluctuations or artifacts. This process, known as source detection or source finding, involves applying algorithms that locate localized peaks or connected regions of pixels whose brightness exceeds a specified threshold relative to the local noise level (Bertin & Arnouts, 1996; Annunziatella et al., 2023). The **`photutils.detection`** module provides several effective tools for this task.

Common approaches implemented in `photutils` include:
*   **Threshold-based Segmentation (`detect_sources`):** This is perhaps the most widely used method for detecting both point-like and extended sources. It operates on the background-subtracted image.
    1.  **Threshold Definition:** A detection threshold image is typically defined, usually as $N_\sigma \times RMS_{bkg}(x, y)$, where $RMS_{bkg}$ is the background RMS noise map (calculated in Section 6.1) and $N_\sigma$ is the significance level (e.g., 3.0 or 5.0). Pixels in the background-subtracted image exceeding this local threshold value are considered potential source pixels.
    2.  **Connectivity:** The `detect_sources` function identifies groups of connected pixels that are all above the specified threshold. Connectivity can be defined using 4-neighbor or 8-neighbor adjacency rules.
    3.  **Minimum Size (`npixels`):** A minimum number of connected pixels (`npixels`) is usually required for a detection to be considered valid. This helps reject isolated noise spikes that might randomly exceed the threshold but do not form spatially coherent structures typical of real sources convolved with the instrumental PSF.
    4.  **Output (Segmentation Map):** The primary output is typically a **segmentation map (`SegmentationImage`)**, an integer image where all pixels belonging to the same detected source are assigned a unique positive integer label, and background pixels are assigned zero. This map delineates the spatial extent of each detected source.
    This method is effective for a wide range of source morphologies but relies heavily on an accurate background RMS map and appropriate choices for the $N_\sigma$ threshold and `npixels` parameter. Too low a threshold increases spurious detections (noise peaks), while too high a threshold misses faint sources.
*   **Peak Finding (`find_peaks`):** This method is specifically designed to find local peaks (maxima) in an image, making it suitable for detecting point-like or marginally resolved sources, particularly stars. It identifies pixels that have a higher value than all their immediate neighbors (within a defined footprint or box size). Additional criteria can be applied, such as a minimum peak height above the local background (`threshold`) or a minimum separation between detected peaks (`min_separation`). `find_peaks` returns a table containing the pixel coordinates of the identified peaks. It is generally faster than segmentation but may struggle with resolving blended sources or detecting very extended, low surface brightness objects that lack a distinct peak.
*   **DAOStarFinder:** This algorithm, inspired by the `DAOFIND` algorithm from the IRAF `DAOPHOT` package (Stetson, 1987), is specifically optimized for finding stellar sources. It works by convolving the image with a kernel that approximates the shape of the PSF (typically a 2D Gaussian) and then searching for peaks in the convolved image that meet specified criteria (threshold above background, roundness/sharpness constraints). It performs iterative centroiding and optional profile fitting to refine source positions and properties. `photutils.detection.DAOStarFinder` provides an implementation of this approach. It is highly effective for stellar fields but less suited for detecting extended or irregular objects.
*   **IRAFStarFinder:** Similar to `DAOStarFinder` but based on the `STARFIND` task in IRAF, `photutils.detection.IRAFStarFinder` uses a thresholding algorithm on the image combined with marginal Gaussian fits to identify stellar sources and refine their centroids.

The choice of detection algorithm depends on the specific science goals and the characteristics of the sources being sought (point-like vs. extended, crowded vs. sparse field). For general-purpose detection of diverse source types, threshold-based segmentation (`detect_sources`) is often a good starting point. The segmentation map produced is a valuable input for subsequent measurement tasks like aperture photometry (Section 6.3) or morphological analysis (Section 6.5). Careful selection of detection parameters ($N_\sigma$, `npixels`, etc.) and visual inspection of the results are crucial to balance completeness (detecting faint real sources) and reliability (avoiding spurious detections).

The following Python code demonstrates source detection using `photutils.detection.detect_sources`. It utilizes the background-subtracted image and the background RMS map generated in the previous example (Section 6.1). A threshold is defined based on the RMS map, and `detect_sources` is called to identify connected pixels above this threshold, producing a segmentation map that labels each distinct detected source.

```python
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip # For background estimation
# Requires photutils: pip install photutils
try:
    from photutils.background import Background2D, MedianBackground, MADStdBackground
    from photutils.segmentation import detect_sources, SourceCatalog, SegmentationImage
    photutils_available = True
except ImportError:
    print("photutils not found, skipping source detection example.")
    photutils_available = False
import matplotlib.pyplot as plt # For visualization
import os # For dummy file creation/check

# --- Input Files (using dummy data from Sec 6.1) ---
input_image_file = 'galaxy_field_image.fits'
# Assumes background maps are available or re-calculated
output_segmap_file = 'galaxy_field_segmap.fits'
output_catalog_file = 'galaxy_field_source_catalog.ecsv' # Save catalog in ECSV format

# --- Re-create dummy data if needed ---
# (Code to re-create input_image_file from Sec 6.1 example if it doesn't exist)
if photutils_available:
    if not os.path.exists(input_image_file):
        print(f"Creating dummy file: {input_image_file}")
        im_size = (200, 200)
        yy, xx = np.indices(im_size)
        background_gradient = 50 + 0.1 * xx + 0.05 * yy
        background_noise = np.random.normal(0, 5.0, size=im_size)
        data = background_gradient + background_noise
        n_src = 30
        x_src = np.random.uniform(0, im_size[1], n_src)
        y_src = np.random.uniform(0, im_size[0], n_src)
        flux_src = 10**(np.random.uniform(1.5, 3.0, n_src))
        psf_sigma = 1.8
        for x, y, flux in zip(x_src, y_src, flux_src):
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['BUNIT'] = 'ADU'
        hdu.writeto(input_image_file, overwrite=True)


if photutils_available:
    try:
        # --- Load Image Data ---
        print(f"Loading image: {input_image_file}")
        try:
            image_data, header = fits.getdata(input_image_file, header=True)
        except FileNotFoundError:
            print(f"Error: File {input_image_file} not found. Cannot proceed.")
            exit()

        # --- Estimate Background and RMS (same as Sec 6.1) ---
        # (Re-run Background2D or load saved maps if available)
        print("Estimating background and RMS...")
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkgrms_estimator = MADStdBackground()
        try:
             bkg = Background2D(image_data, (50, 50), filter_size=(3, 3),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                                bkgrms_estimator=bkgrms_estimator, exclude_percentile=10.0)
             background_map = bkg.background
             background_rms_map = bkg.background_rms
        except Exception as bkg_err:
             print(f"Warning: Background2D failed ({bkg_err}), using global estimate.")
             background_map = np.nanmedian(image_data)
             background_rms_map = np.nanstd(image_data)

        # --- Perform Source Detection ---
        print("Detecting sources using detect_sources...")
        # Define the detection threshold: N-sigma above the background RMS
        n_sigma_threshold = 3.0 # Detect sources at 3-sigma significance
        threshold = background_map + (n_sigma_threshold * background_rms_map)
        # Alternatively, detect on background-subtracted image:
        # threshold = n_sigma_threshold * background_rms_map
        # data_to_detect = image_data - background_map

        # Define minimum number of connected pixels required for a detection
        npixels_min = 5 # Require at least 5 connected pixels

        # Use detect_sources to create a segmentation map
        # Provide the data to analyze (background-subtracted usually preferred)
        # Provide the threshold (can be scalar or 2D array)
        # mask can be used to ignore bad pixels during detection
        segment_map = detect_sources(image_data, threshold, npixels=npixels_min)
        # If detecting on background-subtracted data:
        # segment_map = detect_sources(data_to_detect, threshold, npixels=npixels_min)

        if segment_map:
            num_sources = segment_map.nlabels
            print(f"Detected {num_sources} sources (segments).")

            # --- Optional: Create Source Catalog ---
            # Calculate properties for each detected segment
            # Uses the original image data and the background map for calculations
            print("Calculating source properties...")
            cat = SourceCatalog(data=image_data, segment_img=segment_map, background=background_map)
            # Convert catalog to an Astropy Table for easier handling
            source_table = cat.to_table()
            # Print basic info about the first few sources
            print("Source Catalog (first 5 rows):")
            print(source_table[:5])

            # --- Save Results (Optional) ---
            # Save the segmentation map FITS file
            # segment_map.writeto(output_segmap_file, overwrite=True) # SegmentationImage has writeto method
            print(f"(If successful, segmentation map would be saved to {output_segmap_file})")
            # Save the source catalog table (e.g., in ECSV format for readability and metadata)
            # source_table.write(output_catalog_file, format='ascii.ecsv', overwrite=True)
            print(f"(If successful, source catalog would be saved to {output_catalog_file})")

            # --- Optional: Visualization ---
            # Display the segmentation map overlaid on the image
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            norm = plt.Normalize(vmin=np.percentile(image_data, 1), vmax=np.percentile(image_data, 99))
            ax.imshow(image_data, origin='lower', cmap='gray_r', norm=norm)
            # Use plot_contours for better visibility than direct segmap overlay sometimes
            segment_map.plot_contours(ax=ax, colors='red', linewidths=0.8)
            # Or display the colored segmentation map itself:
            # ax.imshow(segment_map.data, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
            ax.set_title(f"Detected Sources ({num_sources})")
            plt.show()

        else:
            print("No sources detected above the threshold.")


    except ImportError:
        print("Error: photutils library is required but not found.")
    except Exception as e:
        print(f"An unexpected error occurred during source detection: {e}")
else:
     print("Skipping source detection example: photutils unavailable or dummy data missing.")

```

The Python code presented implements a standard source detection workflow using the `photutils` library, building upon the background estimation performed previously. After loading the image and re-calculating or loading the 2D background and background RMS maps, it defines a detection threshold, typically as a multiple (e.g., 3 or 5 sigma) of the local background RMS. The core detection step uses `photutils.segmentation.detect_sources`, which identifies contiguous regions of pixels in the input image (often the background-subtracted image is preferred) that exceed this threshold value and meet a minimum area requirement (`npixels`). The primary output is a `SegmentationImage` object (`segment_map`), an integer array where each distinct detected source is assigned a unique label. The script further demonstrates using `photutils.segmentation.SourceCatalog` to calculate various properties (like centroid coordinates, area, morphology parameters) for each detected segment, compiling them into an Astropy `Table`. This catalog provides quantitative information about the detected sources, essential for subsequent analysis or cross-matching. The optional visualization overlays the boundaries of the detected segments onto the original image, allowing visual verification of the detection results.

**6.3 Aperture Photometry (`photutils.aperture`)**

Once sources have been detected and their positions accurately determined, **aperture photometry** provides a relatively simple and common method for measuring their brightness. The basic principle involves defining a geometric aperture (typically circular for stars, but potentially elliptical or rectangular for other shapes) centered on the source and summing the pixel values within that aperture, after carefully subtracting the contribution from the local background sky level (Howell, 1989; Mighell, 2005). The `photutils.aperture` module provides flexible tools for defining apertures and performing these calculations.

Key components of aperture photometry include:
1.  **Aperture Definition:** Define the shape and size of the aperture used to measure the source flux.
    *   **Circular Aperture (`CircularAperture`):** Most common for point sources (stars). Defined by the source center coordinates $(x_c, y_c)$ and a radius $r$.
    *   **Elliptical Aperture (`EllipticalAperture`):** Useful for measuring slightly extended or elongated sources (e.g., galaxies). Defined by center coordinates, semi-major axis length $a$, semi-minor axis length $b$, and orientation angle $\theta$.
    *   **Rectangular Aperture (`RectangularAperture`):** Less common for photometry but available. Defined by center, width $w$, height $h$, and angle $\theta$.
    The aperture size (radius $r$, or semi-axes $a, b$) is a critical choice. It should be large enough to encompass a significant fraction of the source's light (e.g., 2-3 times the PSF FWHM for stars) but small enough to minimize noise from the background and contamination from neighboring sources. Often, measurements are made with a relatively small, high-SNR aperture, and then an aperture correction (Section 5.4.3) is applied to estimate the total flux.
2.  **Source Positions:** Accurate coordinates ($x_c, y_c$) for the center of each source are required, typically obtained from the source detection step (Section 6.2) or from precise centroiding routines.
3.  **Local Background Subtraction:** The background level within the source aperture must be accurately estimated and subtracted. A common method uses a **sky annulus (`CircularAnnulus`, `EllipticalAnnulus`, `RectangularAnnulus`)** defined around the source aperture (e.g., an inner radius larger than the source aperture and an outer radius defining the annulus width). The background level per pixel is estimated robustly within this annulus (e.g., using the median or sigma-clipped mean of pixel values), excluding pixels belonging to the source itself or other contaminating objects. This per-pixel sky value is then multiplied by the area of the source aperture (in pixels) to estimate the total background contribution within the aperture, which is subtracted from the raw sum of pixel values inside the source aperture.
4.  **Flux Summation (`aperture_photometry`):** The `photutils.aperture.aperture_photometry` function performs the core calculation. It takes the image data, the defined aperture object(s), and optionally sky annulus objects or a pre-calculated background map as input. It calculates the sum of pixel values within each aperture, potentially performing background subtraction using the provided annulus or background map. The function supports various methods for handling fractional pixels at the aperture edge (e.g., `method='exact'`, `'center'`, `'subpixel'`). It can also optionally calculate uncertainties in the photometric measurement if an error array (representing pixel standard deviations) is provided, using standard error propagation rules for summation and subtraction.
5.  **Output:** The function returns an Astropy `Table` containing the results, typically including columns for the aperture sum (`aperture_sum`), source position, aperture area, and optionally the calculated local background and photometric error.

Aperture photometry is conceptually straightforward and widely used. However, its accuracy can be limited in crowded fields where apertures overlap, or for faint sources dominated by background noise. In such cases, PSF photometry (Section 6.4) often provides superior results. Furthermore, careful consideration of aperture size and background estimation methodology is required for reliable measurements.

The following Python code illustrates how to perform aperture photometry on detected sources using `photutils.aperture`. It assumes a source list (pixel coordinates) and the image data are available. It defines circular apertures for the sources and circular annuli for local sky background estimation. The `aperture_photometry` function is then called to compute the background-subtracted flux within each source aperture.

```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
# Requires photutils: pip install photutils
try:
    from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
    photutils_available = True
except ImportError:
    print("photutils not found, skipping aperture photometry example.")
    photutils_available = False
import os # For dummy file creation/check

# --- Input Data (using dummy data from Sec 6.2) ---
input_image_file = 'galaxy_field_bkgsub.fits' # Use background-subtracted image
# Use source positions from previous detection catalog (or re-detect)
# Example: Assume source_table exists from Sec 6.2 or load it
# If loading: source_table = Table.read('galaxy_field_source_catalog.ecsv')
# Get source positions (use 'xcentroid', 'ycentroid' if from SourceCatalog)
# Creating dummy positions if catalog doesn't exist
try:
    source_table = Table.read('galaxy_field_source_catalog.ecsv')
    positions = np.vstack((source_table['xcentroid'], source_table['ycentroid'])).T
except (FileNotFoundError, KeyError):
    print("Warning: Source catalog not found or missing columns. Using dummy positions.")
    n_src = 10
    positions = np.random.uniform(20, 180, size=(n_src, 2)) # Dummy x, y coordinates

# Create dummy background-subtracted image if needed
if photutils_available:
    if not os.path.exists(input_image_file):
        print(f"Creating dummy file: {input_image_file}")
        im_size = (200, 200)
        data = np.random.normal(0, 5.0, size=im_size) # Noise around zero
        yy, xx = np.indices(im_size)
        psf_sigma = 1.8
        for x, y in positions:
            flux = 10**(np.random.uniform(1.5, 2.5)) # Fainter sources now
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['BUNIT'] = 'ADU' # Or electron after gain correction
        hdu.writeto(input_image_file, overwrite=True)


if photutils_available:
    try:
        # --- Load Image Data ---
        print(f"Loading background-subtracted image: {input_image_file}")
        try:
            image_data, header = fits.getdata(input_image_file, header=True)
            image_unit = header.get('BUNIT', 'adu') # Get units if available
        except FileNotFoundError:
            print(f"Error: File {input_image_file} not found. Cannot proceed.")
            exit()

        # --- Define Apertures and Annuli ---
        print("Defining apertures and sky annuli...")
        # Define the radius for the source aperture (e.g., based on PSF FWHM)
        phot_radius = 4.0 # pixels
        # Define inner and outer radii for the background annulus
        sky_radius_in = 6.0 # pixels
        sky_radius_out = 9.0 # pixels

        # Create aperture objects for each source position
        apertures = CircularAperture(positions, r=phot_radius)
        # Create annulus objects for local background estimation
        sky_annuli = CircularAnnulus(positions, r_in=sky_radius_in, r_out=sky_radius_out)
        # Combine aperture and annulus for background subtraction method in aperture_photometry
        all_apers = [apertures, sky_annuli]

        # --- Perform Aperture Photometry with Background Subtraction ---
        print("Performing aperture photometry with annulus background subtraction...")
        # Use method='exact' for precise handling of pixels partially within apertures
        # error can be provided if an error map (pixel std devs) is available
        # mask can be provided to ignore bad pixels
        phot_table_result = aperture_photometry(image_data, all_apers, method='exact')
        # phot_table_result now contains columns like 'aperture_sum_0' (source)
        # and 'aperture_sum_1' (sky annulus sum)

        # --- Calculate Background-Subtracted Flux ---
        print("Calculating background-subtracted flux...")
        # Calculate background mean per pixel in the annulus
        # Need area of annulus to normalize
        sky_mask = sky_annuli.to_mask(method='center') # Use 'center' for area calc? Check docs.
        # Or use exact method and sum weights if needed for area
        # Let's calculate area directly from radii for simplicity here
        # Careful: annulus_apertures argument provides background subtraction directly
        # Check phot_table columns - newer versions might do subtraction internally
        # Alternative manual approach:
        # bkg_mean_per_pixel = phot_table_result['aperture_sum_1'] / sky_annuli.area

        # aperture_photometry with annulus_apertures often calculates background-subtracted sum directly
        # Let's assume 'aperture_sum' is already sky-subtracted when annulus provided:
        if 'aperture_sum' in phot_table_result.colnames:
             # If only one aperture type is passed, name is 'aperture_sum'
             # If multiple passed (as list), names are 'aperture_sum_0', 'aperture_sum_1', ...
             if len(all_apers) == 1:
                  bkg_subtracted_flux = phot_table_result['aperture_sum']
             elif 'aperture_sum_0' in phot_table_result.colnames:
                  # Assume sum in aperture 0 is background subtracted using annulus 1
                  bkg_subtracted_flux = phot_table_result['aperture_sum_0']
             else: # Fallback to manual subtraction if needed
                  print("Warning: Could not determine background subtracted column. Attempting manual.")
                  aperture_area = apertures.area # Use method='exact' for area calculation too
                  # Ensure sky_annuli has area attribute or calculate it
                  sky_area = np.pi * (sky_annuli.r_out**2 - sky_annuli.r_in**2) if hasattr(sky_annuli, 'area') else np.pi * (sky_radius_out**2 - sky_radius_in**2)
                  bkg_sum_in_aperture = (phot_table_result['aperture_sum_1'] / sky_area) * aperture_area
                  bkg_subtracted_flux = phot_table_result['aperture_sum_0'] - bkg_sum_in_aperture # Check indices
        else:
             raise KeyError("Could not find expected 'aperture_sum' column(s).")


        # Add calculated flux to the table
        phot_table_result['flux_net'] = bkg_subtracted_flux
        phot_table_result['flux_net'].info.unit = image_unit # Assign units

        # --- Print Results ---
        print("\nAperture Photometry Results (Net Flux):")
        # Select relevant columns to display
        print(phot_table_result[['id', 'xcenter', 'ycenter', 'flux_net']][:10]) # Show first 10

        # Can now convert 'flux_net' to instrumental magnitude:
        # exposure_time = header.get('EXPTIME', 1.0)
        # mag_instr = -2.5 * np.log10(phot_table_result['flux_net'] / exposure_time)

    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {input_image_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred during aperture photometry: {e}")
else:
     print("Skipping aperture photometry example: photutils unavailable or dummy data missing.")

```

This Python script demonstrates a standard aperture photometry workflow using the `photutils` library to measure the brightness of sources previously detected in an astronomical image. It begins by loading the background-subtracted image data and a list of source coordinates (e.g., from a `SourceCatalog` table generated in the detection step). Using `photutils.aperture`, it defines `CircularAperture` objects centered on each source position to measure the source flux, and corresponding `CircularAnnulus` objects surrounding each source aperture to estimate the local sky background. The core measurement is performed by the `aperture_photometry` function, which takes the image data and the defined aperture/annulus objects as input. By providing both source apertures and sky annuli, the function automatically calculates the median sky background per pixel within each annulus, scales it to the area of the source aperture, and subtracts this sky contribution from the total flux summed within the source aperture, yielding a net instrumental flux (background-subtracted counts) for each source. The results are conveniently returned in an Astropy `Table`.

**6.4 Point Spread Function (PSF) Photometry (`photutils.psf`)**

While aperture photometry is straightforward, it faces limitations in crowded stellar fields where apertures of different stars overlap, leading to mutual contamination and inaccurate flux measurements. Furthermore, aperture photometry can struggle to achieve the highest possible precision, especially for faint sources, as it doesn't optimally utilize information about the known shape of stellar images – the **Point Spread Function (PSF)**. The PSF describes the response of the combined telescope, instrument, and detector system (and atmosphere for ground-based observations) to a point source of light (King, 1971; Anderson & King, 2000). Due to diffraction, optical aberrations, detector effects, and atmospheric turbulence, stars do not appear as perfect points but are spread out into characteristic profiles described by the PSF.

**PSF photometry** leverages knowledge of the PSF shape to measure stellar brightness. Instead of simply summing pixels within an arbitrary aperture, it involves fitting a model of the PSF to the observed pixel data for each star. By adjusting the position and amplitude (flux) of the PSF model to best match the observed star image, PSF photometry can:
*   **Deconvolve Blended Sources:** In crowded fields, algorithms can simultaneously fit PSF models to multiple overlapping stars, disentangling their individual contributions much more effectively than aperture photometry.
*   **Improve Signal-to-Noise:** By weighting pixels according to their expected contribution based on the PSF model (similar in principle to optimal extraction for spectra), PSF fitting can achieve higher photometric precision, especially for faint stars dominated by background noise.
*   **Provide Precise Centroids:** The fitting process yields highly accurate sub-pixel coordinates for the center of each star.

PSF photometry typically involves two main stages: modeling the PSF itself and then fitting this model to the target stars. The **`photutils.psf`** module provides tools for both stages (Bradley et al., 2023).

*   **6.4.1 PSF Modeling**
    Accurate PSF photometry hinges on obtaining a reliable model of the PSF across the image. The PSF can vary with position on the detector (due to optical aberrations or detector effects) and potentially with time (e.g., changes in seeing). Several approaches exist for modeling the PSF:
    1.  **Analytical Models:** Simple mathematical functions like 2D Gaussian (`Gaussian2D`) or Moffat (`Moffat2D`) profiles are often used as approximations of the PSF core. These models are defined by parameters like amplitude, center coordinates ($x_0, y_0$), standard deviations or widths ($\sigma_x, \sigma_y$ or $\gamma$), and potentially ellipticity and orientation ($\theta$). While computationally convenient, simple analytical models often fail to capture the detailed structure in the wings or asymmetries present in real PSFs. `astropy.modeling` provides implementations of these functions.
    2.  **Empirical PSF (ePSF):** A more accurate approach involves constructing the PSF model empirically from the observed images of bright, relatively isolated "PSF stars" within the field. The process involves:
        *   Selecting suitable PSF stars (bright, unsaturated, isolated, well-distributed across the field if spatial variation is expected).
        *   Extracting small cutout images (thumbnails) centered on each selected PSF star.
        *   Carefully measuring the centroids of these stars to sub-pixel accuracy.
        *   Aligning and stacking these cutout images, often using iterative refinement and outlier rejection, to build a high signal-to-noise representation of the average PSF shape. Techniques exist to handle sub-pixel offsets during stacking, effectively creating an oversampled PSF model.
        *   Optionally fitting analytical functions or interpolation schemes to this stacked empirical PSF to create a smooth, continuous model.
        The **`photutils.psf.EPSFBuilder`** class provides a powerful tool for constructing empirical PSFs from stellar images, handling centering, normalization, outlier rejection, and oversampling. It produces an `EPSFModel` object that can be used for fitting.
    3.  **Spatially Varying PSF:** If the PSF shape changes significantly across the field of view, a single PSF model is insufficient. In such cases, multiple ePSFs can be constructed for different regions of the detector, or a more complex model can be built where the PSF parameters (e.g., widths, shape parameters) are functions of the pixel coordinates $(x, y)$. This often involves fitting polynomial functions to describe the variation of PSF characteristics derived from PSF stars across the field. Libraries like `PSFEx` (Bertin, 2011) or custom implementations are used for constructing spatially varying PSF models.

    The choice of PSF modeling technique depends on the required accuracy, the degree of PSF variation across the field, the crowding level, and the availability of suitable PSF stars. Empirical PSFs generally provide the most accurate representation of the true PSF shape.

*   **6.4.2 PSF Fitting for Source Measurement**
    Once a PSF model (analytical or empirical) is available, it can be fitted to the pixel data of target stars to measure their fluxes and precise positions. Several algorithms exist, often implemented within `photutils.psf`:
    1.  **Single Source Fitting:** For relatively isolated stars, the PSF model is centered at an initial guess position (e.g., from source detection), and its amplitude (flux) and potentially small position offsets ($\Delta x, \Delta y$) are adjusted using a least-squares minimization algorithm (e.g., Levenberg-Marquardt implemented in `scipy.optimize.least_squares` or `astropy.modeling.fitting`) to best match the observed pixel values within a small fitting box around the star. The fitted amplitude parameter directly corresponds to the star's estimated total flux (assuming the PSF model is normalized to unit volume/sum).
    2.  **Iterative Fitting (DAOPHOT Algorithm):** For moderately crowded fields, algorithms inspired by DAOPHOT (Stetson, 1987) often employ an iterative approach. Initial fits are performed on brighter stars. The fitted PSF models for these bright stars are then subtracted from the image, revealing fainter underlying sources, which are then fitted in subsequent iterations. This subtract-and-refit process helps deconvolve moderately blended sources. `photutils.psf.BasicPSFPhotometry` and `photutils.psf.IterativelySubtractedPSFPhotometry` provide implementations based on these concepts. These typically require providing initial guesses for source positions (from a detection step) and use the PSF model to refine positions and determine fluxes.
    3.  **Simultaneous Fitting (Group Fitting):** For very crowded fields where many stars significantly overlap, the most robust approach involves simultaneously fitting PSF models to all interacting sources within a defined group. This requires solving a larger, potentially complex, non-linear least-squares problem where the positions and fluxes of all stars in the group are optimized concurrently. This approach explicitly accounts for the flux contribution of neighboring stars at each pixel location. `photutils.psf.DAOPhotPSFPhotometry` implements algorithms capable of handling grouped PSF fitting, often involving sophisticated source grouping strategies.

    The choice of fitting algorithm depends on the degree of crowding and the desired precision. All methods require an accurate PSF model and robust background subtraction prior to fitting. The outputs typically include refined centroid coordinates ($x_{fit}, y_{fit}$), fitted flux ($F_{fit}$), uncertainties on these parameters derived from the covariance matrix of the fit, and goodness-of-fit statistics (e.g., $\chi^2$). PSF photometry, while computationally more intensive than aperture photometry, generally yields more accurate and precise results in crowded fields and is the preferred method for high-precision stellar photometry.

The following Python code provides a conceptual illustration of PSF photometry. It first simulates creating a simple empirical PSF model using `EPSFBuilder` from cutouts of bright, isolated "PSF stars" in an image. Then, it uses this ePSF model with `IterativelySubtractedPSFPhotometry` to perform photometry on all detected stars, demonstrating the basic workflow of building a PSF and using it for fitting.

```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import SigmaClip
# Requires photutils: pip install photutils
try:
    from photutils.detection import DAOStarFinder
    from photutils.background import Background2D, MedianBackground
    from photutils.psf import extract_stars, EPSFBuilder, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry, DAOPhotPSFPhotometry
    from photutils.datasets import make_gaussian_sources_image, make_noise_image
    photutils_available = True
except ImportError:
    print("photutils not found, skipping PSF photometry example.")
    photutils_available = False
from astropy.modeling.models import Gaussian2D # For simulation only
import matplotlib.pyplot as plt
import os # For dummy file creation/check
from astropy.modeling import fitting # Fitter needed

# --- Simulate Image Data with Stars ---
# Use photutils function for creating a more realistic test image
if photutils_available:
    sigma_psf = 2.0
    # Create sources table
    sources = Table()
    sources['flux'] = [1500, 3000, 1000, 2500, 800] # Fluxes
    sources['x_mean'] = [50, 80, 25, 110, 130] # x positions
    sources['y_mean'] = [40, 70, 60, 30, 85]  # y positions
    sources['x_stddev'] = sigma_psf
    sources['y_stddev'] = sigma_psf
    sources['theta'] = 0 # Rotation angle

    # Define image shape
    shape = (150, 150)
    # Make noise-free image with sources
    data_nonoise = make_gaussian_sources_image(shape, sources)
    # Add background and noise
    background_level = 50.0
    noise_sigma = 3.0
    noise = make_noise_image(shape, distribution='gaussian', mean=background_level,
                             stddev=noise_sigma, seed=1234)
    image_data = data_nonoise + noise
    print("Simulated image created.")

    # Define bright, isolated stars to use for building the ePSF
    # In practice, these would be selected carefully from the image
    psf_star_coords = np.array([[80.0, 70.0], [110.0, 30.0]]) # Use stars 1 and 3

    # --- Build Empirical PSF (ePSF) ---
    print("Building empirical PSF (ePSF)...")
    # Create initial guess positions for PSF stars (use true positions here)
    psf_stars_tbl = Table({'x': psf_star_coords[:, 0], 'y': psf_star_coords[:, 1]})
    # Extract cutout images (thumbnails) around the PSF stars
    # Requires estimating background or providing background-subtracted data
    # Use a larger size for extraction than fitting box later
    size = 25 # Size of cutout postage stamp
    hsize = (size - 1) / 2
    # Check boundary conditions carefully
    x = psf_stars_tbl['x']
    y = psf_stars_tbl['y']
    mask = ((x > hsize) & (x < (shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (shape[0] - 1 - hsize)))
    psf_stars_tbl_valid = psf_stars_tbl[mask]
    if len(psf_stars_tbl_valid) < 1:
        raise ValueError("No valid PSF stars found within image boundaries for cutout.")

    print(f"Extracting cutouts for {len(psf_stars_tbl_valid)} valid PSF stars...")
    # Background subtraction should ideally be done before this step
    # Here, subtract the known background level for simplicity
    extracted_stars = extract_stars(image_data - background_level, psf_stars_tbl_valid, size=size)

    # Use EPSFBuilder to create the ePSF model
    # oversampling=1 means no oversampling (for simplicity)
    # maxiters controls iterations for centering and rejection
    try:
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=3, progress_bar=False)
        # Build the ePSF from the extracted star cutouts
        epsf_model, fitted_stars = epsf_builder(extracted_stars)
        print("ePSF model built successfully.")

        # Visualize the ePSF model (optional)
        plt.figure()
        plt.imshow(epsf_model.data, origin='lower', interpolation='nearest')
        plt.title("Empirical PSF Model (ePSF)")
        plt.colorbar()
        plt.show()

    except Exception as epsf_err:
         print(f"Error building ePSF: {epsf_err}. Cannot proceed with PSF photometry.")
         epsf_model = None # Ensure model is None if build failed

    # --- Perform PSF Photometry ---
    if epsf_model:
        print("\nPerforming PSF Photometry using the ePSF model...")
        # Need initial guess positions for all stars we want to measure
        # Use DAOStarFinder to detect all sources first
        bkg_estimator = MedianBackground()
        bkg = Background2D(image_data, (20, 20), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        daofind = DAOStarFinder(fwhm=2*sigma_psf*np.sqrt(2*np.log(2)), # Convert sigma to FWHM
                                threshold=5.0*bkg.background_rms_median)
        sources_init = daofind(image_data - bkg.background)
        if sources_init is None:
            raise ValueError("No sources found by DAOStarFinder for initial positions.")
        init_guess_positions = Table({'x_0': sources_init['xcentroid'], 'y_0': sources_init['ycentroid']})
        print(f"Found {len(init_guess_positions)} sources for initial PSF fitting.")

        # Use IterativelySubtractedPSFPhotometry for potentially crowded fields
        # fitshape defines the size of the fitting box around each star
        # fitter specifies the fitting algorithm from astropy.modeling
        # niters controls number of subtraction iterations
        psf_phot = IterativelySubtractedPSFPhotometry(
            finder=daofind, # Use the same finder for subsequent iterations
            group_maker=None, # Use default grouping or provide one
            bkg_estimator=MedianBackground(), # Estimate background locally if needed
            psf_model=epsf_model, # The ePSF model we built
            fitter=fitting.LevMarLSQFitter(), # Least-squares fitter
            niters=2, # Number of iterations
            fitshape=(11, 11) # Fitting box size (pixels)
            # aperture_radius could be set for initial flux estimates
        )

        # Perform the photometry
        photometry_result = psf_phot(image=image_data, init_guesses=init_guess_positions)
        # Result is an Astropy Table with fitted parameters (x_fit, y_fit, flux_fit, etc.)

        print("\nPSF Photometry Results (first 5 rows):")
        print(photometry_result[['x_0', 'y_0', 'x_fit', 'y_fit', 'flux_fit', 'flux_unc']][:5])
        # Note: 'flux_fit' is the primary result for brightness measurement.

        # Can compare flux_fit to input fluxes for verification in simulation
        # ... comparison code ...

    else:
         print("Skipping PSF Photometry step as ePSF model was not built.")

else:
    print("Skipping PSF Photometry example: photutils unavailable.")
```

This Python script outlines the workflow for performing Point Spread Function (PSF) photometry, a technique crucial for accurate measurements in crowded stellar fields. It begins by simulating an astronomical image containing several stars. Then, it demonstrates the process of building an empirical PSF (ePSF) model: it selects bright, isolated stars from the image, extracts small cutout images centered on them using `photutils.psf.extract_stars`, and feeds these cutouts to `photutils.psf.EPSFBuilder`. The `EPSFBuilder` iteratively aligns and combines these cutouts to construct a high signal-to-noise model (`epsf_model`) representing the average observed PSF shape. With the ePSF model constructed, the script proceeds to perform photometry. It first detects all sources in the image using `DAOStarFinder` to get initial position guesses. Then, it utilizes `photutils.psf.IterativelySubtractedPSFPhotometry`, providing the ePSF model and the initial guesses. This function fits the PSF model to each detected star, potentially performing iterations where bright stars are fitted and subtracted to help measure fainter, nearby sources. The final output is an Astropy `Table` containing the refined positions (`x_fit`, `y_fit`) and, most importantly, the fitted fluxes (`flux_fit`) for each star, representing the PSF photometry measurements. The visualization of the residual image helps assess the fit quality.

**6.5 Basic Morphological Analysis (`photutils.morphology`)**

Beyond simply detecting sources and measuring their brightness, characterizing their shapes or morphology provides valuable physical insights, particularly for extended objects like galaxies or nebulae. While sophisticated galaxy morphology analysis involves fitting complex models (e.g., Sérsic profiles using tools like `statmorph` or `GalaFit` - Rodriguez-Gomez et al., 2019; Ding et al., 2023), basic morphological parameters can be readily derived from the source segmentation maps or image cutouts generated during earlier analysis steps. These parameters offer quantitative descriptions of source size, elongation, and orientation. The `photutils` package provides functions for calculating some of these fundamental properties, often integrated within the `SourceCatalog` class.

Common basic morphological parameters include:
*   **Centroid Coordinates ($x_c, y_c$):** The intensity-weighted center of the source, already calculated during detection or PSF fitting.
*   **Area:** The number of pixels belonging to the source segment (`area` attribute in `SourceCatalog`).
*   **Second Moments:** Measures related to the spatial distribution of flux within the source segment, analogous to moments of inertia. These are often calculated relative to the centroid:
    *   $M_{xx} = \sum (x - x_c)^2 I(x, y) / \sum I(x, y)$
    *   $M_{yy} = \sum (y - y_c)^2 I(x, y) / \sum I(x, y)$
    *   $M_{xy} = \sum (x - x_c)(y - y_c) I(x, y) / \sum I(x, y)$
    where $I(x, y)$ is the intensity (background-subtracted flux) at pixel $(x, y)$, and the sum is over all pixels in the source segment.
*   **Equivalent Radius ($r_{equiv}$):** The radius of a circle having the same area as the source segment: $r_{equiv} = \sqrt{\mathrm{Area} / \pi}$. Provides a basic size estimate.
*   **Semi-major and Semi-minor Axes ($a, b$):** Derived from the second moments, these represent the lengths of the principal axes of an ellipse that best approximates the source shape. They quantify the source's size and elongation.
*   **Ellipticity ($\epsilon$):** A measure of how elongated the source is, often defined as $\epsilon = 1 - b/a$. A value of 0 corresponds to a circular source, while values approaching 1 indicate highly elongated shapes.
*   **Position Angle ($\theta$):** The orientation angle of the major axis of the best-fit ellipse, typically measured counter-clockwise from the positive x-axis or North through East on the sky if WCS is available. Calculated from the second moments: $\tan(2\theta) = \frac{2 M_{xy}}{M_{xx} - M_{yy}}$.

These parameters can be calculated directly from the segmentation map and the image data using functions within `photutils.segmentation` (e.g., properties calculated by `SourceCatalog`) or related image processing libraries (`scipy.ndimage`, `skimage.measure`). While basic, they provide valuable quantitative descriptors for classifying sources (e.g., star vs. galaxy separation based on elongation), selecting samples based on size or shape, or parameterizing the overall structure of extended objects. For detailed galaxy structural analysis, more advanced techniques involving surface brightness profile fitting (e.g., Sérsic models - Sérsic, 1963) are necessary to derive parameters like effective radius, Sérsic index, disk/bulge decomposition, etc. (e.g., using `statmorph` - Rodriguez-Gomez et al., 2019).

The following Python code demonstrates calculating basic morphological properties using `photutils.segmentation.SourceCatalog`. It re-uses the segmentation map generated in Section 6.2 and calculates properties like area, semi-major/minor axes, ellipticity, and orientation for each detected source segment.

```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
# Requires photutils: pip install photutils
try:
    from photutils.segmentation import SourceCatalog, SegmentationImage
    from photutils.background import Background2D, MedianBackground # Needed if recalculating
    photutils_available = True
except ImportError:
    print("photutils not found, skipping morphological analysis example.")
    photutils_available = False
import os # For dummy file creation/check
import astropy.units as u # For orientation unit conversion

# --- Input Data (using outputs from Sec 6.1 & 6.2) ---
input_image_file = 'galaxy_field_image.fits' # Original (or bkg-subtracted) image
input_segmap_file = 'galaxy_field_segmap.fits' # Segmentation map from detect_sources
# Optional: background map needed by SourceCatalog if not using bkg-subtracted image
input_bkgmap_file = 'galaxy_field_bkgmap.fits'
output_morph_catalog_file = 'galaxy_field_morph_catalog.ecsv'

# --- Re-create dummy files if needed ---
# (Code to re-create input_image_file, input_segmap_file, input_bkgmap_file
# from previous examples if they don't exist)
if photutils_available:
    # Check/create image file
    if not os.path.exists(input_image_file):
        # (Simplified creation from Sec 6.1)
        print(f"Creating dummy file: {input_image_file}")
        im_size=(200,200); yy,xx=np.indices(im_size); data=50+0.1*xx+np.random.normal(0,5,size=im_size)
        fits.PrimaryHDU(data.astype(np.float32)).writeto(input_image_file, overwrite=True)
    # Check/create segmentation map file
    if not os.path.exists(input_segmap_file):
        # (Simplified detection from Sec 6.2, requires image)
        print(f"Creating dummy file: {input_segmap_file}")
        from photutils.segmentation import detect_sources
        from astropy.stats import SigmaClip
        data = fits.getdata(input_image_file)
        try: # Try background subtraction first
           bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=SigmaClip(3.0), bkg_estimator=MedianBackground())
           threshold = bkg.background + 3.0 * bkg.background_rms
        except Exception: threshold = np.median(data) + 3.0 * np.std(data)
        segm = detect_sources(data, threshold, npixels=5)
        if segm: segm.writeto(input_segmap_file, overwrite=True)
        else: fits.PrimaryHDU(np.zeros(data.shape, dtype=int)).writeto(input_segmap_file, overwrite=True) # Write empty if no sources
    # Check/create background map file
    if not os.path.exists(input_bkgmap_file):
         print(f"Creating dummy file: {input_bkgmap_file}")
         from photutils.background import Background2D, MedianBackground
         from astropy.stats import SigmaClip
         data = fits.getdata(input_image_file)
         try:
              bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=SigmaClip(3.0), bkg_estimator=MedianBackground())
              fits.PrimaryHDU(bkg.background.astype(np.float32)).writeto(input_bkgmap_file, overwrite=True)
         except Exception: fits.PrimaryHDU(np.ones(data.shape)*np.median(data)).writeto(input_bkgmap_file, overwrite=True) # Write median


if photutils_available:
    try:
        # --- Load Image, Segmentation Map, and Background Map ---
        print(f"Loading image data: {input_image_file}")
        try:
            image_data = fits.getdata(input_image_file)
        except FileNotFoundError:
            print(f"Error: File {input_image_file} not found.")
            exit()

        print(f"Loading segmentation map: {input_segmap_file}")
        try:
            # Read segmentation map using SegmentationImage.read if saved that way
            # Or directly if just a simple FITS image
            segment_map_data = fits.getdata(input_segmap_file)
            # Ensure it's integer type
            segment_map = SegmentationImage(segment_map_data.astype(int))
        except FileNotFoundError:
            print(f"Error: File {input_segmap_file} not found.")
            exit()

        print(f"Loading background map: {input_bkgmap_file}")
        try:
            background_map = fits.getdata(input_bkgmap_file)
        except FileNotFoundError:
            print(f"Warning: File {input_bkgmap_file} not found. Using zero background.")
            background_map = np.zeros_like(image_data) # Use zero if map missing

        # --- Calculate Morphological Properties using SourceCatalog ---
        print("Calculating morphological properties using SourceCatalog...")
        # Provide the original image data (NOT background-subtracted, unless background=None)
        # and the segmentation map. Provide the background map separately.
        # error can be provided for uncertainty calculation on properties.
        source_cat = SourceCatalog(data=image_data, segment_img=segment_map, background=background_map)

        # Convert the catalog to an Astropy Table
        properties_table = source_cat.to_table()

        # --- Print Selected Morphological Properties ---
        # Available properties depend on photutils version, check documentation
        # Common properties: area, equivalent_radius, semimajor_axis_sigma, semiminor_axis_sigma,
        # ellipticity, orientation (theta), Moments (moments_central), etc.
        print("\nSource Catalog with Morphological Properties (first 10 rows):")
        cols_to_show = ['label', 'xcentroid', 'ycentroid', 'area',
                        'equivalent_radius']
        # Check if moment-based properties are available (added in later versions)
        if 'semimajor_axis_sigma' in properties_table.colnames:
             cols_to_show.extend(['semimajor_axis_sigma', 'semiminor_axis_sigma',
                                  'ellipticity', 'orientation'])
             # Calculate b/a ratio for clarity as well
             properties_table['b_a_ratio'] = (properties_table['semiminor_axis_sigma'] /
                                             properties_table['semimajor_axis_sigma'])
             properties_table['b_a_ratio'].info.format = '.3f'
             cols_to_show.append('b_a_ratio')
        else:
             print("(Note: Sigma-based shape parameters might require newer photutils version)")

        # Print selected columns, formatting orientation to degrees
        if 'orientation' in properties_table.colnames:
             # Ensure orientation has units before converting
             if not isinstance(properties_table['orientation'], u.Quantity):
                  # Assuming radians if no unit
                  properties_table['orientation'] = properties_table['orientation'] * u.rad
             properties_table['orientation_deg'] = properties_table['orientation'].to(u.deg)
             properties_table['orientation_deg'].info.format = '.1f' # Format degrees
             cols_to_show.remove('orientation') # Remove original radian column if exists
             cols_to_show.append('orientation_deg')

        # Format other columns for display
        properties_table['xcentroid'].info.format = '.2f'
        properties_table['ycentroid'].info.format = '.2f'
        properties_table['equivalent_radius'].info.format = '.2f'
        if 'semimajor_axis_sigma' in properties_table.colnames:
             properties_table['semimajor_axis_sigma'].info.format = '.2f'
             properties_table['semiminor_axis_sigma'].info.format = '.2f'
             properties_table['ellipticity'].info.format = '.3f'

        # Ensure only existing columns are requested
        valid_cols_to_show = [col for col in cols_to_show if col in properties_table.colnames]
        print(properties_table[valid_cols_to_show][:10])

        # --- Save the Full Catalog (Optional) ---
        # properties_table.write(output_morph_catalog_file, format='ascii.ecsv', overwrite=True)
        print(f"\n(If successful, catalog with morphology would be saved to {output_morph_catalog_file})")


    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred during morphological analysis: {e}")
else:
     print("Skipping morphological analysis example: photutils unavailable or dummy data missing.")

```

This Python script demonstrates the calculation of basic morphological parameters for sources detected in an astronomical image, leveraging the `photutils` library. It assumes that the original image data, a corresponding segmentation map (output from `detect_sources`, labeling pixels belonging to each source), and an estimated background map are available. The core of the process utilizes `photutils.segmentation.SourceCatalog`, which is initialized with the image data, the segmentation map, and the background map. This class automatically calculates a variety of properties for each source segment defined in the segmentation map, including area, centroid coordinates, flux-related properties, and basic morphological parameters derived from image moments (such as semi-major and semi-minor axes based on standard deviations, ellipticity, and orientation angle). The script converts the resulting catalog into an Astropy `Table` for convenient access and printing, showcasing key morphological outputs that provide quantitative measures of the size, shape, and orientation of the detected sources.

**6.6 Catalog Cross-Matching (`astropy.coordinates`)**

A ubiquitous task in astronomical research is identifying detected sources by cross-matching their positions with entries in large astronomical catalogs or databases. This allows researchers to determine if a newly detected source corresponds to a known object (e.g., star, galaxy, quasar), retrieve additional information about that object from the catalog (e.g., magnitudes in different bands, redshift, spectral type, variability information), or combine information from different surveys or wavelength regimes (Allen et al., 2022; Budavári & Szalay, 2008). Effective cross-matching relies on having accurate celestial coordinates (RA, Dec) for both the detected sources (derived from astrometric calibration, Section 5.2) and the reference catalog entries.

The **`astropy.coordinates`** sub-package provides the fundamental tools for representing and working with celestial coordinates in Python, including powerful functions for performing spatial cross-matching. Key concepts and steps include:

1.  **Representing Coordinates:** Both the detected source positions and the reference catalog positions must be represented using `astropy.coordinates.SkyCoord` objects. These objects encapsulate the RA and Dec values along with their units (e.g., degrees) and the coordinate frame (e.g., ICRS - International Celestial Reference System, which is standard for modern catalogs like Gaia). `SkyCoord` objects can be created from various input formats (e.g., lists/arrays of RA/Dec values, string representations).
2.  **Reference Catalog Acquisition:** Coordinates from the reference catalog (e.g., Gaia, SDSS, Pan-STARRS, NED, SIMBAD) are typically obtained by querying the relevant online database, often programmatically using `astroquery` (Section 5.2.2), retrieving RA, Dec, and other desired parameters for objects within the region of interest. These are then converted into a `SkyCoord` object.
3.  **Matching Algorithm (`match_to_catalog_sky`):** The core matching operation is performed using the `SkyCoord.match_to_catalog_sky()` method. Given a `SkyCoord` object representing the list of detected sources (`coords_detected`) and another `SkyCoord` object representing the reference catalog sources (`coords_catalog`), this method efficiently finds the closest catalog source for *each* detected source on the celestial sphere.
    `idx, d2d, d3d = coords_detected.match_to_catalog_sky(coords_catalog)`
    The method returns:
    *   `idx`: An array of indices into the `coords_catalog` object. For each detected source $i$, `idx[i]` gives the index of the closest source in the catalog.
    *   `d2d`: An `astropy.coordinates.Angle` object representing the angular separation on the sky between each detected source and its closest catalog match.
    *   `d3d`: The 3D physical separation (requires distance information in the `SkyCoord` objects, often not available or needed for simple 2D matching).
4.  **Applying Match Threshold:** A crucial step is to filter the matches based on the angular separation `d2d`. Only matches where the separation is smaller than a specified tolerance or search radius (e.g., 0.5 or 1.0 arcseconds, depending on the astrometric accuracy of the detected sources and the catalog) are considered reliable identifications. Matches with larger separations are likely spurious alignments or correspond to objects present in one list but not the other.
5.  **Accessing Matched Information:** Using the indices `idx` and the filter mask derived from the separation threshold, one can link the properties of the detected sources (from the detection catalog) with the properties of their corresponding matches in the reference catalog (e.g., retrieve the catalog ID, standard magnitudes, redshift, etc. for the identified objects).

This process enables robust identification of detected sources and the fusion of information from different datasets. The efficiency of `astropy.coordinates`' matching algorithms, which utilize optimized spatial indexing techniques (like k-d trees), allows cross-matching of very large catalogs containing millions or billions of sources. Careful consideration of the appropriate matching radius, based on the astrometric uncertainties involved, is essential for achieving reliable results and minimizing false matches.

The following Python code demonstrates cross-matching a list of detected source coordinates (assumed to be derived from an image with accurate WCS) against the Gaia catalog retrieved via `astroquery`. It uses `SkyCoord` objects to represent coordinates and the `match_to_catalog_sky` method to find the nearest Gaia counterpart for each detected source, applying a separation threshold to identify reliable matches.

```python
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
# Requires astroquery for Gaia query: pip install astroquery
try:
    from astroquery.gaia import Gaia
    astroquery_available = True
except ImportError:
    print("astroquery not found, cross-matching example will use dummy catalog.")
    astroquery_available = False
import os # For dummy file creation/check

# --- Input Data ---
# Assume 'source_catalog' is an Astropy Table with RA/Dec columns
# (e.g., from detection + WCS application, or loading a previous catalog)
# Create a dummy source catalog:
n_det = 50
# Simulate some RA/Dec around a fictional field center
center_ra, center_dec = 180.0, -10.0
det_ra = np.random.normal(center_ra, 0.05, n_det) # Scatter within ~0.1 deg
det_dec = np.random.normal(center_dec, 0.05, n_det)
# Add some measured property, e.g., instrumental magnitude
det_mag = np.random.uniform(19, 23, n_det)
source_catalog = Table({'RA_detected': det_ra, 'DEC_detected': det_dec, 'Mag_instrumental': det_mag})
source_catalog['RA_detected'].unit = u.deg
source_catalog['DEC_detected'].unit = u.deg
print(f"Created dummy source catalog with {len(source_catalog)} detected objects.")

# --- Query Reference Catalog (Gaia) ---
print("\nQuerying reference catalog (Gaia DR3)...")
# Define query region based on detected source coordinates
query_radius = 0.1 * u.deg # Search radius around the field center
query_coord = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')

if astroquery_available:
    try:
        gaia_table = Gaia.query_object_async(coordinate=query_coord, radius=query_radius)
        if len(gaia_table) == 0:
            raise ValueError("No Gaia sources found in the query region.")
        print(f"Found {len(gaia_table)} Gaia sources.")
        # Create SkyCoord object for Gaia sources
        ref_catalog_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit='deg', frame='icrs')
        # Keep relevant Gaia columns (e.g., ID, G mag)
        gaia_table_subset = gaia_table['source_id', 'ra', 'dec', 'phot_g_mean_mag']
    except Exception as e:
        print(f"Warning: Gaia query failed: {e}. Using dummy catalog.")
        astroquery_available = False # Fallback to dummy

# Create dummy catalog if Gaia query failed or unavailable
if not astroquery_available:
    n_ref = 60
    ref_ra = np.random.normal(center_ra, 0.05, n_ref)
    ref_dec = np.random.normal(center_dec, 0.05, n_ref)
    ref_mag = np.random.uniform(18, 24, n_ref)
    ref_id = np.arange(n_ref) + 1000000
    gaia_table_subset = Table({'source_id': ref_id, 'ra': ref_ra, 'dec': ref_dec, 'phot_g_mean_mag': ref_mag})
    ref_catalog_coords = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg, frame='icrs')
    print("Using dummy reference catalog.")


# --- Perform Cross-Matching ---
print("\nPerforming cross-matching...")
# Create SkyCoord object for detected sources
detected_coords = SkyCoord(ra=source_catalog['RA_detected'], dec=source_catalog['DEC_detected'], unit='deg', frame='icrs')

# Use match_to_catalog_sky to find nearest neighbor in reference catalog
# Returns index into ref_catalog_coords, 2D separation, 3D separation
idx_ref, sep2d, sep3d = detected_coords.match_to_catalog_sky(ref_catalog_coords)

# --- Apply Separation Threshold ---
# Define maximum acceptable separation for a match
match_radius = 1.0 * u.arcsec # Example: 1 arcsecond tolerance
match_mask = sep2d <= match_radius
num_matches = np.sum(match_mask)
print(f"Found {num_matches} matches within {match_radius} ({num_matches / len(detected_coords) * 100:.1f}% of detected sources).")

# --- Combine Matched Information ---
print("\nCreating table of matched sources...")
# Create table of detected sources that have a match
matched_detected_sources = source_catalog[match_mask]

# Get the corresponding matched reference catalog entries using the indices
matched_ref_sources = gaia_table_subset[idx_ref[match_mask]]

# Add separation information to the detected source table (optional)
matched_detected_sources['Ref_ID'] = matched_ref_sources['source_id']
matched_detected_sources['Ref_GMag'] = matched_ref_sources['phot_g_mean_mag']
matched_detected_sources['Separation_arcsec'] = sep2d[match_mask].to(u.arcsec)

# Print the combined table for the first few matches
print("Combined Matched Catalog (first 10 rows):")
print(matched_detected_sources[:10])

# 'matched_detected_sources' now contains the original detected source info
# plus columns identifying the Gaia counterpart and its properties.

```

This Python script demonstrates the essential steps for cross-matching a catalog of detected astronomical sources against a reference catalog, exemplified using Gaia DR3 retrieved via `astroquery`. It begins by creating `astropy.coordinates.SkyCoord` objects representing the celestial positions (RA, Dec) of both the detected sources (from the input `source_catalog`) and the reference sources (from the queried `gaia_table`). The core matching logic is performed by the `detected_coords.match_to_catalog_sky(ref_catalog_coords)` method, which efficiently finds the nearest reference source in the sky for each detected source and returns the index of the best match (`idx_ref`) and the angular separation (`sep2d`). A critical step involves filtering these potential matches by applying a maximum separation threshold (`match_radius`), selecting only pairs that are closer than this tolerance to ensure reliable associations. Finally, the script combines information by selecting the subset of detected sources that passed the matching threshold and retrieving the corresponding reference source data (like Gaia source ID and magnitude) using the returned indices, creating a new table (`matched_detected_sources`) that links the detected objects to their known counterparts in the Gaia catalog.

**6.7 Examples in Practice (Python): Image Analysis Tasks**

This section provides illustrative Python examples applying the image analysis techniques discussed throughout the chapter to specific scientific scenarios across different subfields of astronomy. Each example focuses on a characteristic task, such as detecting sunspots, measuring planetary moon brightness, performing photometry in crowded stellar fields, extracting light curves from survey data, quantifying emission knots in nebulae, detecting faint galaxies, and cross-matching catalogs for cosmological studies. These examples utilize libraries like `photutils`, `astropy`, and potentially image data manipulation tools, showcasing the practical implementation of background subtraction, source detection, aperture photometry, PSF photometry, basic morphology, and catalog matching within realistic astronomical contexts.

**6.7.1 Solar: Detection and Area Measurement of Sunspots**
Analyzing solar images, such as continuum intensitygrams from SDO/HMI, often involves identifying and characterizing active regions like sunspots. Sunspots appear as dark areas compared to the surrounding photosphere. A common task is to detect these dark regions and measure their properties, such as area, which relates to magnetic flux and solar activity levels. This example simulates detecting dark sunspot regions by thresholding *below* the average photospheric intensity (after potentially removing large-scale limb darkening effects, which are ignored here for simplicity) and uses `photutils.segmentation` to identify connected dark pixels and measure their area.

```python
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
# Requires photutils: pip install photutils
try:
    from photutils.segmentation import detect_sources, SourceCatalog
    from photutils.background import Background2D, MedianBackground # Can use for photosphere level
    photutils_available = True
except ImportError:
    print("photutils not found, skipping Solar sunspot example.")
    photutils_available = False
import matplotlib.pyplot as plt
import os
from astropy.stats import mad_std # For robust RMS

# --- Input Data (Simulated HMI Intensitygram) ---
hmi_intensity_file = 'hmi_intensity_sample.fits'

# Create dummy file if it doesn't exist
if photutils_available:
    if not os.path.exists(hmi_intensity_file):
        print(f"Creating dummy file: {hmi_intensity_file}")
        im_size = (150, 150)
        # Simulate photosphere + noise + dark sunspots
        photosphere_level = 30000.0
        noise = np.random.normal(0, 100.0, size=im_size)
        data = photosphere_level + noise
        # Add sunspots (darker regions)
        spots = [(50, 60, 8, 0.6), (100, 100, 12, 0.4), (80, 30, 5, 0.7)] # y, x, radius, relative_intensity
        yy, xx = np.indices(im_size)
        for y, x, r, intensity_factor in spots:
            dist_sq = (xx - x)**2 + (yy - y)**2
            spot_mask = dist_sq < r**2
            data[spot_mask] *= intensity_factor # Reduce intensity within spot radius
        # Simple limb darkening approximation (cosine) - real correction is more complex
        center_x, center_y = im_size[1]/2.0, im_size[0]/2.0
        radius = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        max_radius = np.sqrt((im_size[1]/2.0)**2 + (im_size[0]/2.0)**2)
        limb_dark_factor = np.cos(np.arcsin(np.minimum(radius / max_radius, 1.0)))**0.5 # Simplified
        data *= limb_dark_factor
        # Ensure non-negative
        data = np.maximum(data, 0)
        # Create HDU and write
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['BUNIT'] = 'Intensity'
        hdu.writeto(hmi_intensity_file, overwrite=True)


if photutils_available:
    try:
        # --- Load Solar Image ---
        print(f"Loading solar intensity image: {hmi_intensity_file}")
        try:
            image_data, header = fits.getdata(hmi_intensity_file, header=True)
        except FileNotFoundError:
            print(f"Error: File {hmi_intensity_file} not found. Cannot proceed.")
            exit()

        # --- Estimate Photosphere Level (Background) ---
        # Use Background2D, but expect high values. Masking edges might be needed.
        # For simplicity, use a robust global median as the reference photosphere level.
        # In reality, limb darkening removal / local normalization is crucial first.
        print("Estimating overall photosphere intensity level (using median)...")
        photosphere_median = np.nanmedian(image_data)
        # Estimate noise - use MAD for robustness against dark spots
        photosphere_rms = mad_std(image_data, ignore_nan=True)
        print(f"  Median Photosphere Level: {photosphere_median:.1f}")
        print(f"  Photosphere RMS (MAD): {photosphere_rms:.1f}")

        # --- Detect Dark Sunspots ---
        # Detect by thresholding *below* the median level.
        # Threshold = median - N_sigma * RMS
        n_sigma_threshold = 3.0
        detection_threshold_value = photosphere_median - n_sigma_threshold * photosphere_rms
        print(f"Detecting regions darker than {detection_threshold_value:.1f} ({n_sigma_threshold}-sigma below median)...")

        # Use detect_sources, providing the threshold value.
        # Set 'data' to be negative of image to find 'peaks' in darkness, or use invert=True?
        # Or simply threshold directly: image_data < detection_threshold_value
        # Let's use direct thresholding to create a boolean mask first.
        dark_pixel_mask = image_data < detection_threshold_value

        # Use detect_sources on the boolean mask to find connected regions
        # npixels sets the minimum area (in pixels) for a valid sunspot detection
        npixels_min = 10
        segment_map = detect_sources(dark_pixel_mask, threshold=0.5, npixels=npixels_min) # Threshold=0.5 for boolean mask

        if segment_map:
            num_spots = segment_map.nlabels
            print(f"Detected {num_spots} potential sunspot regions.")

            # --- Measure Properties (Area) ---
            print("Calculating properties of detected regions...")
            # Use SourceCatalog on the original image data and the segmentation map
            # Background can be set to the estimated photosphere level
            source_cat = SourceCatalog(data=image_data, segment_img=segment_map,
                                       background=photosphere_median) # Use median as background reference
            properties_table = source_cat.to_table()

            # Extract and print relevant properties, especially area
            print("\nSunspot Region Properties (Area):")
            # Area is typically in pixels^2
            cols_to_show = ['label', 'xcentroid', 'ycentroid', 'area']
            print(properties_table[cols_to_show])

            # Can convert area to physical units if pixel scale is known (from WCS)
            # pixel_scale_arcsec = header.get('CDELT1', 0.6) # Example arcsec/pixel
            # area_arcsec_sq = properties_table['area'] * pixel_scale_arcsec**2

        else:
            print("No sunspot regions detected below the threshold.")
            num_spots = 0 # Initialize if no spots found

        # --- Optional: Visualization ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        norm = plt.Normalize(vmin=np.percentile(image_data, 1), vmax=np.percentile(image_data, 99))
        ax.imshow(image_data, origin='lower', cmap='afmhot', norm=norm) # 'afmhot' good for solar
        if segment_map:
            segment_map.plot_contours(ax=ax, colors='cyan', linewidths=1.0)
        ax.set_title(f"Detected Sunspot Regions ({num_spots})")
        plt.show()


    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {hmi_intensity_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred during sunspot detection: {e}")
else:
     print("Skipping Solar sunspot example: photutils unavailable or dummy data missing.")

```

The Python script above implements a simplified workflow for detecting and characterizing dark sunspot regions in a simulated solar intensitygram, utilizing the `photutils` library. After loading the image, it estimates the typical intensity level of the surrounding photosphere using a robust median calculation and estimates the noise level using the Median Absolute Deviation (MAD). Sunspot detection is achieved by identifying pixels significantly *darker* than this reference level, setting a threshold at N-sigma *below* the median. A boolean mask is created for pixels falling below this darkness threshold. `photutils.segmentation.detect_sources` is then used on this boolean mask to group connected dark pixels into distinct segments, requiring a minimum number of pixels (`npixels`) to qualify as a valid detection. Finally, `photutils.segmentation.SourceCatalog` calculates properties for each detected sunspot segment using the original image data and the segmentation map, specifically extracting the `area` (in pixels squared) as a key characteristic of sunspot size. The results are visualized by overlaying the detected segment boundaries onto the original solar image.

**6.7.2 Planetary: Aperture Photometry of Jupiter's Moons**
Monitoring the brightness variations of Jupiter's Galilean moons (Io, Europa, Ganymede, Callisto) can provide information about their surface properties, rotational periods, or mutual eclipse/occultation events. This requires performing photometry on the moons as they appear near the bright, extended disk of Jupiter. Aperture photometry is a suitable technique here. This example demonstrates measuring the instrumental flux of Jupiter's moons in a simulated image. It uses `photutils.aperture` to define circular apertures around the moons and background annuli nearby (carefully placed to avoid Jupiter's glare) to estimate and subtract the local background, yielding the net instrumental counts for each moon, which could then be used to generate light curves or calibrate magnitudes.

```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
# Requires photutils: pip install photutils
try:
    from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
    photutils_available = True
except ImportError:
    print("photutils not found, skipping Planetary moon photometry example.")
    photutils_available = False
import matplotlib.pyplot as plt
import os

# --- Input Data (Simulated Image of Jupiter and Moons) ---
jupiter_moons_file = 'jupiter_moons_image.fits'

# Create dummy file if it doesn't exist
if photutils_available:
    if not os.path.exists(jupiter_moons_file):
        print(f"Creating dummy file: {jupiter_moons_file}")
        im_size = (200, 200)
        # Simulate background + Jupiter (bright, saturated) + Moons (point sources)
        background = np.random.normal(50.0, 3.0, size=im_size)
        data = background
        # Jupiter (large, bright, likely saturated in core)
        jup_center_x, jup_center_y = 100, 100
        jup_radius_x, jup_radius_y = 30, 28
        jup_flux_scale = 50000.0 # Bright
        yy, xx = np.indices(im_size)
        dist_sq_jup = (((xx - jup_center_x)/jup_radius_x)**2 + ((yy - jup_center_y)/jup_radius_y)**2)
        jupiter_profile = jup_flux_scale * np.exp(-0.5 * dist_sq_jup**0.8) # Not really physical, just blob
        data += jupiter_profile
        # Add moons (point sources) at various positions
        # Assume pixel coords (x, y) are known or found via detection
        moon_coords = [(40.5, 110.2), (70.8, 95.5), (135.1, 105.8), (160.3, 88.1)] # Io, Europa, Ganymede, Callisto (example)
        moon_fluxes = [1500, 1200, 1800, 1600] # Relative brightness
        psf_sigma = 1.5
        for (x, y), flux in zip(moon_coords, moon_fluxes):
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Add noise and saturation
        saturation_limit = 40000.0
        data = np.random.poisson(np.maximum(data, 0)).astype(float)
        data = np.minimum(data, saturation_limit) # Apply saturation limit
        # Create HDU and write
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['OBJECT'] = 'Jupiter+Moons'
        hdu.header['BUNIT'] = 'Counts'
        hdu.writeto(jupiter_moons_file, overwrite=True)


if photutils_available:
    try:
        # --- Load Image Data ---
        print(f"Loading image: {jupiter_moons_file}")
        try:
            image_data, header = fits.getdata(jupiter_moons_file, header=True)
            image_unit = header.get('BUNIT', 'count') # Get units
        except FileNotFoundError:
            print(f"Error: File {jupiter_moons_file} not found. Cannot proceed.")
            exit()

        # --- Define Apertures for Moons and Sky Annuli ---
        # Use the known/detected positions of the moons
        print("Defining apertures and sky annuli for moons...")
        positions = moon_coords # Use the simulated positions
        psf_fwhm_approx = 2.355 * psf_sigma # Approximate FWHM if sigma known
        phot_radius = 2.0 * psf_fwhm_approx # Aperture radius based on FWHM
        # Define sky annuli CAREFULLY to avoid Jupiter's extended light
        # Place annulus further out or adjust size/position based on image inspection
        sky_radius_in = phot_radius + 5.0 # Start annulus well beyond aperture
        sky_radius_out = sky_radius_in + 6.0
        moon_apertures = CircularAperture(positions, r=phot_radius)
        sky_annuli = CircularAnnulus(positions, r_in=sky_radius_in, r_out=sky_radius_out)

        # --- Perform Aperture Photometry ---
        print("Performing aperture photometry on moons...")
        # Provide error= if uncertainty map available
        # Provide mask= if bad pixels/saturated Jupiter core needs masking
        # Create a simple mask for saturated Jupiter core (example)
        saturation_limit = 40000.0 # Re-define if needed
        mask_sat = image_data >= saturation_limit
        # Perform photometry, requesting calculation of local background median per pixel
        phot_table_moons = aperture_photometry(image_data, moon_apertures,
                                               local_bkg_annulus=sky_annuli, # Specify annulus for background
                                               mask=mask_sat, method='exact')
        # The 'local_bkg_annulus' argument tells photutils to calculate median background per pixel
        # in the annulus and subtract the scaled contribution from 'aperture_sum'

        # The primary result 'aperture_sum' should now be background-subtracted
        bkg_subtracted_flux = phot_table_moons['aperture_sum']

        # Add results to a more descriptive table
        results = Table()
        results['Moon_ID'] = ['Io', 'Europa', 'Ganymede', 'Callisto'] # Example names
        results['x_pix'] = phot_table_moons['xcenter'].value
        results['y_pix'] = phot_table_moons['ycenter'].value
        results['Net_Counts'] = bkg_subtracted_flux
        results['Net_Counts'].info.unit = image_unit

        # --- Print Results ---
        print("\nAperture Photometry Results (Jupiter's Moons):")
        print(results)

        # These net counts can be converted to instrumental magnitudes and potentially
        # calibrated if standard stars are also measured in the field (see Example 5.7.2).
        # Light curves can be generated by repeating this for multiple images over time.

        # --- Optional: Visualization ---
        plt.figure(figsize=(8, 8))
        norm = plt.Normalize(vmin=np.percentile(image_data[~mask_sat], 1), vmax=np.percentile(image_data[~mask_sat], 99.9)) # Avoid saturation in norm
        plt.imshow(image_data, origin='lower', cmap='bone', norm=norm)
        moon_apertures.plot(color='lime', lw=1.0)
        sky_annuli.plot(color='red', linestyle='--', lw=1.0)
        plt.title("Jupiter Moons Aperture Photometry")
        plt.show()


    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {jupiter_moons_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred during moon photometry: {e}")
else:
     print("Skipping Planetary moon photometry example: photutils unavailable or dummy data missing.")

```

This Python script simulates the measurement of instrumental brightness for Jupiter's Galilean moons using aperture photometry with `photutils`. After loading a simulated image containing Jupiter (potentially saturated) and its moons, it defines the pixel coordinates for the moons. `photutils.aperture.CircularAperture` objects are created centered on these coordinates to define the measurement regions, while `photutils.aperture.CircularAnnulus` objects define nearby sky regions, carefully positioned to avoid contamination from Jupiter's bright, extended light. The `photutils.aperture.aperture_photometry` function is then used with the `local_bkg_annulus` argument, which instructs the function to calculate the median background per pixel within the annulus and subtract the appropriately scaled background contribution from the flux summed within the source aperture. The script extracts the resulting background-subtracted net counts for each moon and presents them in a table. These instrumental measurements form the basis for constructing light curves or performing differential photometric calibration. The visualization overlays the defined apertures and annuli on the image for verification.

**6.7.3 Stellar: PSF Photometry in a Crowded Globular Cluster**
Globular clusters present extremely crowded stellar fields, where stars are densely packed and their images (PSFs) significantly overlap. Simple aperture photometry fails in such environments because apertures inevitably include contaminating flux from neighboring stars. PSF photometry is the required technique, involving fitting a model of the PSF to each star, often simultaneously for blended groups, to accurately measure individual fluxes and positions. This example demonstrates performing PSF photometry on a simulated crowded cluster image. It assumes a PSF model (either pre-built empirically or analytical) is available and uses `photutils.psf` tools (like `IterativelySubtractedPSFPhotometry` or `DAOPhotPSFPhotometry`) to fit the PSF to stars detected in the image, returning precise positions and PSF-fitted fluxes.

```python
import numpy as np
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.io import fits
# Requires photutils: pip install photutils
try:
    from photutils.detection import IRAFStarFinder # Good finder for crowded fields
    from photutils.background import Background2D, MedianBackground, MMMBackground
    from photutils.psf import IntegratedGaussianPRF, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry, DAOPhotPSFPhotometry
    from photutils.datasets import make_gaussian_sources_image, make_noise_image
    photutils_available = True
except ImportError:
    print("photutils not found, skipping PSF photometry (crowded) example.")
    photutils_available = False
from astropy.modeling import fitting # Fitter needed
import matplotlib.pyplot as plt
import os

# --- Simulate Crowded Cluster Image ---
if photutils_available:
    sigma_psf = 1.8 # PSF standard deviation
    fwhm_psf = gaussian_sigma_to_fwhm(sigma_psf)
    n_stars = 300 # More stars for crowding
    shape = (150, 150)
    # Generate random positions, concentrated towards center
    center_x, center_y = shape[1]/2.0, shape[0]/2.0
    pos_stddev = 25.0
    x_mean = np.random.normal(center_x, pos_stddev, n_stars)
    y_mean = np.random.normal(center_y, pos_stddev, n_stars)
    # Generate fluxes (luminosity function-like: more faint than bright)
    flux = 10**(np.random.uniform(1.0, 3.5, n_stars)) # Wider flux range
    # Clip positions to be within image boundaries
    x_mean = np.clip(x_mean, 0, shape[1]-1)
    y_mean = np.clip(y_mean, 0, shape[0]-1)

    sources = Table()
    sources['flux'] = flux
    sources['x_mean'] = x_mean
    sources['y_mean'] = y_mean
    sources['x_stddev'] = sigma_psf
    sources['y_stddev'] = sigma_psf
    sources['theta'] = 0

    data_nonoise = make_gaussian_sources_image(shape, sources)
    background_level = 80.0
    noise_sigma = 4.0
    noise = make_noise_image(shape, distribution='gaussian', mean=background_level,
                             stddev=noise_sigma, seed=4321)
    image_data = data_nonoise + noise
    print("Simulated crowded cluster image created.")

    # --- Define PSF Model ---
    # Use an analytical Gaussian model matching the simulation for simplicity
    # In practice, use an empirical PSF (ePSF) built from isolated stars if possible
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf) # Integrated Gaussian is better for flux
    # photutils psf models are typically normalized internally for use in photometry classes
    print("Using analytical IntegratedGaussianPRF as PSF model.")

    # --- Perform PSF Photometry ---
    try:
        # Estimate background
        # Use MMMBackground for potential robustness in crowded fields
        bkg_estimator = MMMBackground()
        bkg = Background2D(image_data, (25, 25), filter_size=(3, 3), bkg_estimator=bkg_estimator)

        # Find initial star positions using IRAFStarFinder (often good in crowds)
        # Requires FWHM and threshold relative to background
        starfind = IRAFStarFinder(fwhm=fwhm_psf, threshold=5.0*bkg.background_rms_median)
        sources_init = starfind(image_data - bkg.background) # Detect on bkg-subtracted
        if sources_init is None:
            raise ValueError("No initial sources found by IRAFStarFinder.")
        # Use 'xcentroid', 'ycentroid' and 'flux' as initial guesses
        init_guess_table = sources_init['xcentroid', 'ycentroid', 'flux']
        init_guess_table.rename_column('xcentroid', 'x_0')
        init_guess_table.rename_column('ycentroid', 'y_0')
        init_guess_table.rename_column('flux', 'flux_0') # Provide initial flux guess
        print(f"Found {len(init_guess_table)} initial source guesses.")


        # Choose PSF Photometry algorithm
        # BasicPSFPhotometry is faster but doesn't handle blends well
        # IterativelySubtractedPSFPhotometry handles moderate crowding
        # DAOPhotPSFPhotometry handles tighter groups but can be slower/complex
        # Let's use IterativelySubtracted for this example
        fitter = fitting.LevMarLSQFitter() # Define the fitter object
        photometry_engine = IterativelySubtractedPSFPhotometry(
            finder=starfind, # Finder for iterations
            group_maker=None, # Default grouping
            bkg_estimator=bkg_estimator, # Pass background estimator, not subtracted data
            psf_model=psf_model,
            fitter=fitter,
            niters=3, # Iterations of find-fit-subtract
            fitshape=(7, 7) # Fitting box size (pixels) should be odd and encompass PSF core
            # aperture_radius could be set for initial flux estimates
        )

        print("Performing iterative PSF photometry...")
        # Pass the original image data, not background-subtracted
        photometry_result = photometry_engine(image=image_data, init_guesses=init_guess_table)

        # Filter results? (e.g., remove sources with fit errors, flags)
        # ... filtering logic ...

        print("\nPSF Photometry Results (Crowded Field, first 10):")
        # Rename columns for clarity if needed
        # photometry_result.rename_column('x_fit', 'PSF_x')
        # photometry_result.rename_column('y_fit', 'PSF_y')
        # photometry_result.rename_column('flux_fit', 'PSF_Flux')
        cols_to_show = ['x_0', 'y_0', 'iter_detected', 'x_fit', 'y_fit', 'flux_fit', 'flux_unc']
        # Ensure columns exist before trying to show them
        valid_cols = [col for col in cols_to_show if col in photometry_result.colnames]
        print(photometry_result[valid_cols][:10])

        # --- Optional: Visualization ---
        # Show residual image after subtracting fitted PSFs
        residual_image = photometry_engine.get_residual_image()
        plt.figure(figsize=(8, 8))
        residual_norm = plt.Normalize(vmin=np.percentile(residual_image, 5), vmax=np.percentile(residual_image, 95))
        plt.imshow(residual_image, origin='lower', cmap='gray', norm=residual_norm)
        plt.title("Residual Image after PSF Subtraction")
        plt.colorbar()
        plt.show()

    except ImportError:
        print("Error: photutils library is required but not found.")
    except ValueError as e:
         print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during crowded field PSF photometry: {e}")
else:
    print("Skipping PSF Photometry (crowded) example: photutils unavailable.")

```

This Python script tackles the challenging problem of performing photometry in a simulated crowded stellar field, representative of a globular cluster, using PSF fitting techniques from `photutils`. After generating a test image with numerous overlapping Gaussian sources, it defines a PSF model (here, an analytical `IntegratedGaussianPRF` matching the simulation, but ideally an empirical PSF would be built as in Section 6.4.1). Accurate background estimation is performed using `Background2D`. Initial source detection and approximate positions are obtained using `IRAFStarFinder`, suitable for crowded fields. The core PSF photometry is then executed using `IterativelySubtractedPSFPhotometry`. This algorithm takes the PSF model, initial source guesses, and a background estimator, iteratively fits the PSF to stars (using a specified `fitter` like Levenberg-Marquardt and a defined `fitshape`), subtracts the models of fitted stars, and potentially detects fainter stars in subsequent iterations (`niters`). The final output is an Astropy `Table` containing the refined positions (`x_fit`, `y_fit`) and accurately measured PSF fluxes (`flux_fit`) for the stars, effectively disentangling the blended sources. Visualizing the residual image (original image minus all fitted PSF models) helps assess the quality of the fits and check for remaining unfitted sources or PSF mismatches.

**6.7.4 Exoplanetary: Aperture Photometry on TESS FFI Data**
While the primary data product for TESS transit searches is the pipeline-generated light curve (see Chapter 1, Example 1.7.4), researchers sometimes need to perform custom photometry on the TESS Full Frame Images (FFIs). FFIs are large images covering a wide field, taken at lower cadence (e.g., 10-30 minutes). Analyzing FFIs might be necessary to extract light curves for stars not included in the primary target lists, to use different photometric apertures, or to investigate instrumental effects. This example demonstrates performing aperture photometry on a cutout from a TESS FFI to extract a raw light curve for a specific target star. It involves defining an aperture around the target star (identified by its pixel coordinates on the FFI or cutout) and summing the flux within this aperture for each FFI cadence, while subtracting the local background estimated from an annulus.

```python
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
# Requires photutils: pip install photutils
try:
    from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
    photutils_available = True
except ImportError:
    print("photutils not found, skipping TESS FFI photometry example.")
    photutils_available = False
import matplotlib.pyplot as plt
import os

# --- Input Data (Simulated TESS FFI Cutout Time Series) ---
# Assume 'tess_ffi_cutout.fits' is a FITS file with a data cube (time, y, x)
# and corresponding time information (e.g., in a binary table extension)
ffi_cutout_file = 'tess_ffi_cutout.fits'

# Create dummy FFI cutout file if it doesn't exist
if photutils_available:
    if not os.path.exists(ffi_cutout_file):
        print(f"Creating dummy file: {ffi_cutout_file}")
        n_cadences = 100
        im_size = (20, 20) # Small cutout size
        # Simulate background + target star + noise
        background_level = 200.0
        noise_sigma = 5.0
        # Simulate target star at center with slight variability/noise
        target_x, target_y = im_size[1]/2.0 - 0.5, im_size[0]/2.0 - 0.5
        target_flux = np.random.normal(loc=5000.0, scale=50.0, size=n_cadences) # Base flux + noise
        # Add a fake transit dip
        target_flux[40:50] *= 0.99 # 1% dip
        yy, xx = np.indices(im_size)
        dist_sq = (xx - target_x)**2 + (yy - target_y)**2
        psf_sigma = 1.2 # TESS PSF is broad
        psf_profile = np.exp(-dist_sq / (2 * psf_sigma**2))
        # Create data cube (time, y, x)
        data_cube = (np.random.normal(background_level, noise_sigma, size=(n_cadences, *im_size)) +
                     target_flux[:, np.newaxis, np.newaxis] * psf_profile[np.newaxis, :, :])
        # Create Primary HDU (minimal)
        hdu0 = fits.PrimaryHDU()
        # Create Image HDU for the data cube
        hdr1 = fits.Header({'EXTNAME': 'FLUX', 'BUNIT': 'e-/s'}) # Example units
        hdu1 = fits.ImageHDU(data_cube.astype(np.float32), header=hdr1)
        # Create BinTable HDU for time stamps
        times_bjd = np.linspace(2458850.5, 2458850.5 + 100 * (10./1440.), n_cadences) # 10 min cadence approx
        col_time = fits.Column(name='TIME', format='D', array=times_bjd, unit='BJD')
        hdu2 = fits.BinTableHDU.from_columns([col_time], name='TIME')
        # Assemble and write
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        hdul.writeto(ffi_cutout_file, overwrite=True)


if photutils_available:
    try:
        # --- Load FFI Cutout Data Cube and Time Info ---
        print(f"Loading TESS FFI cutout data: {ffi_cutout_file}")
        try:
            with fits.open(ffi_cutout_file) as hdul:
                # Assume flux cube is in HDU 1, time in HDU 2
                flux_cube = hdul['FLUX'].data
                flux_unit = hdul['FLUX'].header.get('BUNIT', 'e-/s')
                time_table = Table(hdul['TIME'].data)
                times = Time(time_table['TIME'], format='jd', scale='tdb') # Use Astropy Time
        except FileNotFoundError:
            print(f"Error: File {ffi_cutout_file} not found. Cannot proceed.")
            exit()
        except KeyError as e:
            print(f"Error: Expected HDU ('FLUX' or 'TIME') not found: {e}")
            exit()

        n_frames, n_y, n_x = flux_cube.shape
        print(f"Loaded data cube with {n_frames} frames, shape ({n_y}, {n_x}).")

        # --- Define Aperture and Annulus for Target ---
        # Use target coordinates within the cutout (e.g., center)
        target_pos = (n_x / 2.0 - 0.5, n_y / 2.0 - 0.5) # Center of cutout
        phot_radius = 2.5 # pixels (choose based on TESS PSF and crowding)
        sky_radius_in = 4.0
        sky_radius_out = 6.0
        aperture = CircularAperture(target_pos, r=phot_radius)
        sky_annulus = CircularAnnulus(target_pos, r_in=sky_radius_in, r_out=sky_radius_out)
        print(f"Defined aperture (r={phot_radius}) and annulus (r_in={sky_radius_in}, r_out={sky_radius_out}).")

        # --- Perform Aperture Photometry on Each Frame ---
        print(f"Performing aperture photometry on {n_frames} frames...")
        # Create lists to store results
        net_fluxes = []
        flux_errs = [] # Placeholder for errors

        for i in range(n_frames):
            # Extract the single frame
            image_frame = flux_cube[i, :, :]
            # Perform photometry on this frame
            # Provide error= if per-frame error map is available
            # Use local_bkg_annulus for background subtraction
            phot_table_frame = aperture_photometry(image_frame, aperture,
                                                   local_bkg_annulus=sky_annulus,
                                                   method='exact')

            # Extract the net flux ('aperture_sum' is background-subtracted with local_bkg_annulus)
            net_flux = phot_table_frame['aperture_sum'][0]
            net_fluxes.append(net_flux)
            # Placeholder for error calculation (needs error array input to aperture_photometry)
            flux_errs.append(np.sqrt(max(net_flux, 0)) if flux_unit=='e-/s' else np.nan) # Very basic approx

            # Optional: Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Processed frame {i+1}/{n_frames}")

        # Convert results to numpy arrays with units
        net_fluxes = np.array(net_fluxes) * u.Unit(flux_unit)
        flux_errs = np.array(flux_errs) * u.Unit(flux_unit) # Apply same unit for now

        # --- Create Light Curve Table ---
        light_curve = Table({
            'time': times.tdb.jd, # Time in JD (TDB)
            'flux': net_fluxes,
            'flux_err': flux_errs
        })
        light_curve['time'].info.format = '.8f' # Display more precision
        print("\nGenerated Raw Light Curve Table (first 5 rows):")
        print(light_curve[:5])

        # --- Optional: Plot Light Curve ---
        plt.figure(figsize=(10, 4))
        plt.errorbar(light_curve['time'], light_curve['flux'].value, yerr=light_curve['flux_err'].value,
                     fmt='.', color='k', ecolor='gray', alpha=0.7)
        plt.xlabel(f"Time ({light_curve['time'].unit or 'JD'})")
        plt.ylabel(f"Raw Aperture Flux ({light_curve['flux'].unit})")
        plt.title("Raw TESS FFI Light Curve from Aperture Photometry")
        plt.grid(True, alpha=0.3)
        plt.show()

    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {ffi_cutout_file} not found.")
    except KeyError as e:
         print(f"Error accessing data from FITS HDU or Table column: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during TESS FFI photometry: {e}")
else:
     print("Skipping TESS FFI photometry example: photutils unavailable or dummy data missing.")

```

This Python script demonstrates how to extract a raw light curve for a target star from a time-series of TESS Full Frame Image (FFI) cutouts using aperture photometry with `photutils`. It begins by loading the data cube (dimensions: time, y, x) and corresponding time stamps from the input FITS file. A circular aperture (`CircularAperture`) is defined around the target star's position within the cutout, along with a surrounding sky annulus (`CircularAnnulus`) for background estimation. The script then iterates through each time slice (frame) of the data cube. In each iteration, `photutils.aperture.aperture_photometry` is called with the `local_bkg_annulus` argument, which calculates the background-subtracted flux within the source aperture based on the median sky level in the annulus. The resulting background-subtracted net flux for each frame is stored. Finally, these time-ordered flux measurements are combined with the corresponding time stamps into an Astropy `Table` and plotted, producing the raw light curve of the target star derived directly from the FFI data.

**6.7.5 Galactic: Detection and Photometry of H-alpha Knots**
Images of Galactic nebulae taken through narrowband filters like H-alpha often reveal intricate structures, including small, bright knots or clumps associated with ongoing star formation, shocks, or photoionization fronts. Identifying these knots and measuring their brightness is important for understanding the physical processes within the nebula. This example simulates detecting these H-alpha emission knots using `photutils.detection.detect_sources` (as they might be slightly extended or irregular) and then performing aperture photometry on the detected knots using `photutils.aperture` to quantify their H-alpha flux relative to the surrounding diffuse nebular background.

```python
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
# Requires photutils: pip install photutils
try:
    from photutils.background import Background2D, MedianBackground, MADStdBackground
    from photutils.segmentation import detect_sources, SourceCatalog, SegmentationImage
    from photutils.aperture import aperture_photometry, CircularAperture
    photutils_available = True
except ImportError:
    print("photutils not found, skipping Galactic knot example.")
    photutils_available = False
import matplotlib.pyplot as plt
import os

# --- Input Data (Simulated H-alpha Nebula Image) ---
halpha_image_file = 'halpha_nebula_image.fits'

# Create dummy file if it doesn't exist
if photutils_available:
    if not os.path.exists(halpha_image_file):
        print(f"Creating dummy file: {halpha_image_file}")
        im_size = (180, 180)
        # Simulate diffuse nebula background + noise + bright knots
        yy, xx = np.indices(im_size)
        diffuse_nebula = 50 * np.exp(-0.5 * (((xx - 90)/50)**2 + ((yy - 90)/60)**2)) # Elliptical nebula
        diffuse_nebula += 30 * np.sin(xx/15.0) * np.cos(yy/20.0) # Some structure
        background_noise = np.random.normal(0, 4.0, size=im_size)
        data = diffuse_nebula + background_noise + 20 # Add base level
        # Add knots (brighter, compact regions)
        knots = [(50, 70, 3.0, 150), (100, 110, 2.5, 200), (130, 60, 4.0, 120)] # y, x, sigma, peak_flux
        for y, x, sigma, flux in knots:
            dist_sq = (xx - x)**2 + (yy - y)**2
            data += flux * np.exp(-dist_sq / (2 * sigma**2))
        # Ensure non-negative
        data = np.maximum(data, 0)
        # Create HDU and write
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['FILTER'] = 'Halpha'
        hdu.header['BUNIT'] = 'Counts/s' # Example units
        hdu.writeto(halpha_image_file, overwrite=True)


if photutils_available:
    try:
        # --- Load H-alpha Image ---
        print(f"Loading H-alpha image: {halpha_image_file}")
        try:
            image_data, header = fits.getdata(halpha_image_file, header=True)
            image_unit = header.get('BUNIT', 'count/s')
        except FileNotFoundError:
            print(f"Error: File {halpha_image_file} not found. Cannot proceed.")
            exit()

        # --- Estimate Diffuse Nebular Background ---
        # Use Background2D to model the large-scale diffuse emission
        print("Estimating diffuse background...")
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground() # Median is good for diffuse structure
        bkgrms_estimator = MADStdBackground()
        # Use a relatively large box size to capture diffuse emission, small filter
        try:
             bkg = Background2D(image_data, (40, 40), filter_size=(3, 3),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                                bkgrms_estimator=bkgrms_estimator, exclude_percentile=20.0) # Exclude brighter parts
             background_map = bkg.background
             background_rms_map = bkg.background_rms
        except Exception as bkg_err:
             print(f"Warning: Background2D failed ({bkg_err}), using global median.")
             background_map = np.nanmedian(image_data)
             background_rms_map = mad_std(image_data, ignore_nan=True)


        # --- Detect Knots (Segmentation) ---
        # Detect compact sources above the diffuse background
        print("Detecting H-alpha knots...")
        n_sigma_threshold = 4.0
        threshold = background_map + (n_sigma_threshold * background_rms_map)
        npixels_min = 4 # Minimum knot size in pixels

        # Use detect_sources on original data with 2D threshold map
        segment_map = detect_sources(image_data, threshold, npixels=npixels_min)

        if not segment_map:
            print("No knots detected above threshold.")
            exit()

        num_knots = segment_map.nlabels
        print(f"Detected {num_knots} knots.")

        # --- Perform Aperture Photometry on Knots ---
        # Calculate knot centroids from segmentation map
        print("Performing aperture photometry on detected knots...")
        source_cat = SourceCatalog(data=image_data, segment_img=segment_map, background=background_map)
        # Use 'local_background' from SourceCatalog or perform annulus photometry
        # For simplicity, use fixed circular apertures centered on centroids
        positions = np.vstack((source_cat.xcentroid, source_cat.ycentroid)).T
        phot_radius = 3.0 # pixels (choose appropriate radius for knots)
        apertures = CircularAperture(positions, r=phot_radius)

        # Perform photometry using the background map estimated earlier
        # Note: Background is implicitly subtracted if `background` is provided to SourceCatalog,
        # but aperture_photometry needs explicit background handling if used standalone.
        # Here, we extract flux using SourceCatalog which includes background subtraction.
        # Or re-run aperture_photometry on background-subtracted data.
        # Let's use SourceCatalog's 'source_sum' which is background-subtracted.
        phot_table = source_cat.to_table()
        # Add units if known
        phot_table['source_sum'].unit = image_unit

        # --- Print Results ---
        print("\nAperture Photometry Results (H-alpha Knots):")
        cols_to_show = ['label', 'xcentroid', 'ycentroid', 'area', 'source_sum']
        # Ensure columns exist before printing
        valid_cols = [col for col in cols_to_show if col in phot_table.colnames]
        print(phot_table[valid_cols])

        # --- Optional: Visualization ---
        plt.figure(figsize=(8, 8))
        norm = plt.Normalize(vmin=np.percentile(image_data, 1), vmax=np.percentile(image_data, 99))
        plt.imshow(image_data, origin='lower', cmap='viridis', norm=norm)
        segment_map.plot_contours(ax=plt.gca(), colors='white', linewidths=0.8)
        apertures.plot(color='red', lw=0.5, alpha=0.7) # Show apertures used
        plt.title(f"Detected H-alpha Knots ({num_knots}) with Apertures")
        plt.show()

    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {halpha_image_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred during H-alpha knot analysis: {e}")
else:
     print("Skipping Galactic knot example: photutils unavailable or dummy data missing.")

```

This Python script demonstrates a workflow for identifying and measuring the flux of compact emission knots within a simulated H-alpha image of a Galactic nebula using `photutils`. It first estimates the large-scale, diffuse nebular emission using `Background2D`, treating it as the 'background' relative to the knots. Source detection is then performed using `detect_sources`, identifying connected pixels that rise significantly above this local diffuse background level plus noise, effectively segmenting the brighter knots. Subsequently, `SourceCatalog` is used with the original image data, the segmentation map, and the estimated diffuse background map. `SourceCatalog` calculates various properties, including the background-subtracted integrated flux (`source_sum`) within each segmented knot region. This provides a quantitative measure of the H-alpha emission originating specifically from each detected knot, suitable for studying the energetics or structure within the nebula. The visualization overlays the detected knot boundaries and the photometric apertures onto the H-alpha image.

**6.7.6 Extragalactic: Galaxy Detection and Flux Measurement**
Detecting and measuring the properties of faint, extended galaxies in deep field images is a core task in extragalactic astronomy. Unlike stars, galaxies have diverse morphologies and extended profiles, requiring detection algorithms sensitive to low surface brightness features and photometric methods appropriate for extended sources. This example simulates detecting galaxies using `photutils.segmentation.detect_sources` (often effective for extended objects) after background subtraction, and then measuring their total flux using elliptical apertures derived from basic morphological parameters (like semi-major/minor axes and orientation) calculated by `photutils.segmentation.SourceCatalog`.

```python
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
# Requires photutils: pip install photutils
try:
    from photutils.background import Background2D, MedianBackground, MADStdBackground
    from photutils.segmentation import detect_sources, SourceCatalog, SegmentationImage
    from photutils.aperture import EllipticalAperture, aperture_photometry
    photutils_available = True
except ImportError:
    print("photutils not found, skipping Extragalactic galaxy detection example.")
    photutils_available = False
import matplotlib.pyplot as plt
import os
import astropy.units as u # Needed for orientation conversion

# --- Input Data (Simulated Deep Field Image) ---
deep_field_file = 'deep_field_image.fits'

# Create dummy file if it doesn't exist
if photutils_available:
    if not os.path.exists(deep_field_file):
        print(f"Creating dummy file: {deep_field_file}")
        im_size = (250, 250)
        # Simulate background + faint galaxies (various shapes) + stars
        background = np.random.normal(30.0, 2.0, size=im_size)
        data = background
        # Add galaxies (elliptical profiles)
        n_galaxies = 25
        x_gal = np.random.uniform(0, im_size[1], n_galaxies)
        y_gal = np.random.uniform(0, im_size[0], n_galaxies)
        flux_gal = 10**(np.random.uniform(1.0, 2.5, n_galaxies))
        a_gal = np.random.uniform(3.0, 8.0, n_galaxies) # Semi-major axis
        ellip_gal = np.random.uniform(0.1, 0.7, n_galaxies) # Ellipticity (1-b/a)
        b_gal = a_gal * (1.0 - ellip_gal) # Semi-minor axis
        theta_gal = np.random.uniform(0, np.pi, n_galaxies) # Orientation (radians)
        yy, xx = np.indices(im_size)
        for x, y, flux, a, b, theta in zip(x_gal, y_gal, flux_gal, a_gal, b_gal, theta_gal):
            cost = np.cos(theta); sint = np.sin(theta)
            xt = (xx - x) * cost + (yy - y) * sint
            yt = -(xx - x) * sint + (yy - y) * cost
            dist_sq_scaled = (xt / a)**2 + (yt / b)**2
            # Use Gaussian for simplicity, though Sersic is more realistic
            data += flux * np.exp(-0.5 * dist_sq_scaled)
        # Add some stars
        n_stars = 50
        x_stars = np.random.uniform(0, im_size[1], n_stars)
        y_stars = np.random.uniform(0, im_size[0], n_stars)
        flux_stars = 10**(np.random.uniform(1.0, 3.0, n_stars))
        psf_sigma = 1.5
        for x, y, flux in zip(x_stars, y_stars, flux_stars):
             dist_sq = (xx - x)**2 + (yy - y)**2
             data += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
        # Ensure non-negative
        data = np.maximum(data, 0)
        # Create HDU and write
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header['FILTER'] = 'i'
        hdu.header['BUNIT'] = 'ADU'
        hdu.writeto(deep_field_file, overwrite=True)


if photutils_available:
    try:
        # --- Load Image ---
        print(f"Loading deep field image: {deep_field_file}")
        try:
            image_data, header = fits.getdata(deep_field_file, header=True)
            image_unit = header.get('BUNIT', 'adu')
        except FileNotFoundError:
            print(f"Error: File {deep_field_file} not found. Cannot proceed.")
            exit()

        # --- Estimate Background ---
        print("Estimating background...")
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        try:
             bkg = Background2D(image_data, (64, 64), filter_size=(3, 3),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
             background_map = bkg.background
             background_rms_map = bkg.background_rms
        except Exception as bkg_err:
             print(f"Warning: Background estimation failed ({bkg_err}). Using global.")
             background_map = np.nanmedian(image_data)
             background_rms_map = np.nanstd(image_data)

        # --- Detect Sources (Galaxies and Stars) ---
        print("Detecting sources (segmentation)...")
        n_sigma_threshold = 2.5 # Lower threshold for faint extended features
        threshold = background_map + (n_sigma_threshold * background_rms_map)
        npixels_min = 6 # Minimum size

        segment_map = detect_sources(image_data, threshold, npixels=npixels_min)
        if not segment_map:
            print("No sources detected.")
            exit()
        num_sources = segment_map.nlabels
        print(f"Detected {num_sources} sources.")

        # --- Calculate Properties including Morphology ---
        print("Calculating source properties (including morphology)...")
        # Use original image and background map
        source_cat = SourceCatalog(data=image_data, segment_img=segment_map, background=background_map)
        properties_table = source_cat.to_table()

        # Check if morphological properties needed for elliptical aperture are present
        if not all(col in properties_table.colnames for col in ['semimajor_axis_sigma', 'semiminor_axis_sigma', 'orientation']):
             print("Warning: Morphological parameters (a_sigma, b_sigma, orientation) not found in catalog. Cannot use elliptical apertures. Check photutils version or calculation.")
             # Could fallback to circular aperture based on equivalent_radius
             use_elliptical = False
             phot_radius = properties_table['equivalent_radius'] * 2.0 # Example fallback
             apertures = CircularAperture((properties_table['xcentroid'], properties_table['ycentroid']), r=phot_radius)
        else:
             use_elliptical = True
             # Define elliptical apertures based on measured shape parameters
             # Scale the sigma-based axes (usually by factor ~2-3) to encompass more light
             kron_radius_factor = 2.5 # Kron radius factor (k) often used for total mag
             a_phot = properties_table['semimajor_axis_sigma'] * kron_radius_factor
             b_phot = properties_table['semiminor_axis_sigma'] * kron_radius_factor
             # Ensure orientation has units (SourceCatalog usually returns Quantity)
             theta_phot = properties_table['orientation'] # Angle from SourceCatalog
             positions = (properties_table['xcentroid'], properties_table['ycentroid'])
             apertures = EllipticalAperture(positions, a=a_phot, b=b_phot, theta=theta_phot)
             print(f"Defined elliptical apertures based on morphology (scaled by {kron_radius_factor}).")


        # --- Perform Aperture Photometry using Defined Apertures ---
        print(f"Performing {'elliptical' if use_elliptical else 'circular'} aperture photometry...")
        # Use background-subtracted data for photometry if background map is reliable
        image_bkg_subtracted = image_data - background_map
        phot_table_aperture = aperture_photometry(image_bkg_subtracted, apertures, method='exact')
        # Result is in 'aperture_sum' column

        net_flux = phot_table_aperture['aperture_sum']
        net_flux.unit = image_unit # Add units

        # Add flux to the main properties table (match by label/ID if needed)
        properties_table['flux_aperture'] = net_flux

        # --- Print Results ---
        print("\nGalaxy Detection and Photometry Results (first 10):")
        cols_to_show = ['label', 'xcentroid', 'ycentroid', 'area', 'flux_aperture']
        if use_elliptical: cols_to_show.extend(['semimajor_axis_sigma', 'ellipticity', 'orientation'])
        valid_cols = [col for col in cols_to_show if col in properties_table.colnames]
        # Format orientation before printing
        if 'orientation' in valid_cols:
             properties_table['orientation_deg'] = properties_table['orientation'].to(u.deg)
             properties_table['orientation_deg'].info.format = '.1f'
             valid_cols.remove('orientation')
             valid_cols.append('orientation_deg')
        print(properties_table[valid_cols][:10])

        # --- Optional: Visualization ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        norm = plt.Normalize(vmin=np.percentile(image_data, 5), vmax=np.percentile(image_data, 98))
        ax.imshow(image_data, origin='lower', cmap='gray_r', norm=norm)
        segment_map.plot_contours(ax=ax, colors='blue', linewidths=0.5, alpha=0.6)
        apertures.plot(ax=ax, color='red', lw=0.8)
        ax.set_title(f"Detected Galaxies/Sources ({num_sources}) with Elliptical Apertures")
        plt.show()

    except ImportError:
        print("Error: photutils library is required but not found.")
    except FileNotFoundError:
        print(f"Error: Input file {deep_field_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred during galaxy detection/photometry: {e}")
else:
     print("Skipping Extragalactic galaxy detection example: photutils unavailable or dummy data missing.")
```

This Python script outlines a common workflow for detecting galaxies in deep field images and measuring their flux using `photutils`. After loading the image and estimating the 2D background, it uses `photutils.segmentation.detect_sources` with a relatively low threshold and minimum pixel count to segment both stars and potentially faint, extended galaxies. `photutils.segmentation.SourceCatalog` is then employed to calculate properties for each detected segment, crucially including morphological parameters like the semi-major axis (`semimajor_axis_sigma`), semi-minor axis (`semiminor_axis_sigma`), and orientation (`orientation`) based on second moments. These morphological parameters are then used to define `photutils.aperture.EllipticalAperture` objects tailored to the shape and orientation of each detected source, scaling the sigma-based axes appropriately (e.g., by a Kron-like factor) to encompass a significant fraction of the galaxy light. Finally, `aperture_photometry` is run on the background-subtracted image using these custom elliptical apertures to measure the integrated flux for each detected object, providing photometry adapted to the extended nature of galaxies. The visualization overlays the segmentation map and the derived elliptical apertures on the image.

**6.7.7 Cosmology: Cross-matching Optical and X-ray Catalogs**
Cosmological studies often involve identifying galaxy clusters, which are massive conglomerations of galaxies embedded in hot gas and dark matter. These clusters can be detected through various means, including optical surveys (finding overdensities of red galaxies) and X-ray surveys (detecting thermal emission from the hot intracluster medium). Cross-matching optical galaxy cluster catalogs with X-ray source catalogs is essential for confirming cluster candidates, studying the relationship between the galaxy population and the hot gas, and understanding cluster astrophysics. This example demonstrates cross-matching a simulated optical cluster catalog (containing RA, Dec) with a simulated X-ray source catalog using `astropy.coordinates.match_to_catalog_sky` to identify potential counterparts within a specified search radius.

```python
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
# Requires astroquery only if querying real catalogs, not needed for dummy data matching
import os

# --- Input Data (Simulated Catalogs) ---
# 1. Optical Galaxy Cluster Catalog (RA, Dec, Richness/Mass Estimate)
n_opt_clusters = 50
opt_ra = np.random.uniform(210, 215, n_opt_clusters) # RA range in degrees
opt_dec = np.random.uniform(10, 15, n_opt_clusters) # Dec range in degrees
opt_richness = np.random.uniform(20, 100, n_opt_clusters) # Example richness
opt_catalog = Table({'OptClusterID': np.arange(n_opt_clusters),
                     'RA_opt': opt_ra, 'DEC_opt': opt_dec,
                     'Richness': opt_richness})
opt_catalog['RA_opt'].unit = u.deg
opt_catalog['DEC_opt'].unit = u.deg
print(f"Created dummy optical cluster catalog with {len(opt_catalog)} entries.")

# 2. X-ray Source Catalog (RA, Dec, X-ray Flux/Luminosity)
n_xray_sources = 70
# Simulate X-ray sources, some coincident with optical clusters, some random
xray_ra = np.concatenate([
    np.random.normal(opt_ra[:30], 0.005, 30), # 30 sources near optical clusters
    np.random.uniform(210, 215, 40) # 40 random background sources
])
xray_dec = np.concatenate([
    np.random.normal(opt_dec[:30], 0.005, 30), # Near optical clusters
    np.random.uniform(10, 15, 40) # Random background
])
xray_flux = 10**(np.random.uniform(-15, -13, n_xray_sources)) # Example flux units
xray_catalog = Table({'XRayID': np.arange(n_xray_sources) + 1000,
                      'RA_xray': xray_ra, 'DEC_xray': xray_dec,
                      'Flux_X': xray_flux})
xray_catalog['RA_xray'].unit = u.deg
xray_catalog['DEC_xray'].unit = u.deg
xray_catalog['Flux_X'].unit = 'erg/s/cm2' # Example unit
print(f"Created dummy X-ray source catalog with {len(xray_catalog)} entries.")

# --- Perform Cross-Matching ---
print("\nPerforming cross-matching between optical clusters and X-ray sources...")
# Create SkyCoord objects for both catalogs
opt_coords = SkyCoord(ra=opt_catalog['RA_opt'], dec=opt_catalog['DEC_opt'], unit='deg', frame='icrs')
xray_coords = SkyCoord(ra=xray_catalog['RA_xray'], dec=xray_catalog['DEC_xray'], unit='deg', frame='icrs')

# Match optical clusters TO the X-ray catalog
# Finds the nearest X-ray source for each optical cluster
idx_xray, sep2d_opt_to_xray, _ = opt_coords.match_to_catalog_sky(xray_coords)

# Match X-ray sources TO the optical catalog
# Finds the nearest optical cluster for each X-ray source
idx_opt, sep2d_xray_to_opt, _ = xray_coords.match_to_catalog_sky(opt_coords)

# --- Apply Match Criterion (Bi-directional + Radius) ---
# Define a maximum search radius for a plausible match
# Cluster positions can be uncertain, X-ray positions too. Use generous radius.
max_separation = 1.0 * u.arcmin # Example: 1 arcminute matching radius

# Find matches within the radius from the opt -> xray perspective
matches1 = sep2d_opt_to_xray <= max_separation
# Find matches within the radius from the xray -> opt perspective
matches2 = sep2d_xray_to_opt <= max_separation

# Require a bi-directional ("best") match to increase reliability
# An optical cluster's best X-ray match must also have that optical cluster as its best match.
# Create indices for matching
opt_indices = np.arange(len(opt_catalog))
xray_indices = np.arange(len(xray_catalog))
# Check consistency: index of best X-ray match for opt[i] should point back to i
# i.e., idx_opt[idx_xray[i]] == i
consistent_match_mask = (idx_opt[idx_xray] == opt_indices)

# Final match mask: separation must be small AND match must be bi-directional best
# Apply consistency check only to pairs already within the separation limit from the primary catalog's perspective
final_match_mask = matches1 & consistent_match_mask # Check consistency for all opt clusters

# Apply the separation threshold to the primary catalog perspective again
final_match_mask = final_match_mask & (sep2d_opt_to_xray <= max_separation)


num_final_matches = np.sum(final_match_mask)
print(f"Found {num_final_matches} reliable bi-directional matches within {max_separation}.")

# --- Create Table of Matched Pairs ---
# Select the optical clusters that have a reliable match
matched_opt_clusters = opt_catalog[final_match_mask]
# Get the indices of the corresponding X-ray sources
matched_xray_indices = idx_xray[final_match_mask]
# Select the corresponding matched X-ray sources
matched_xray_sources = xray_catalog[matched_xray_indices]
# Get the separation for these final matches
matched_separation = sep2d_opt_to_xray[final_match_mask]

# Combine information into a single table
matched_table = Table()
matched_table['OptClusterID'] = matched_opt_clusters['OptClusterID']
matched_table['RA_opt'] = matched_opt_clusters['RA_opt']
matched_table['DEC_opt'] = matched_opt_clusters['DEC_opt']
matched_table['Richness'] = matched_opt_clusters['Richness']
matched_table['XRayID_match'] = matched_xray_sources['XRayID']
matched_table['RA_xray_match'] = matched_xray_sources['RA_xray']
matched_table['DEC_xray_match'] = matched_xray_sources['DEC_xray']
matched_table['Flux_X_match'] = matched_xray_sources['Flux_X']
matched_table['Separation_arcmin'] = matched_separation.to(u.arcmin)

print("\nTable of Matched Optical Clusters and X-ray Sources (first 10):")
# Format columns for better display
matched_table['RA_opt'].info.format = '.4f'
matched_table['DEC_opt'].info.format = '.4f'
matched_table['RA_xray_match'].info.format = '.4f'
matched_table['DEC_xray_match'].info.format = '.4f'
matched_table['Flux_X_match'].info.format = '.2e'
matched_table['Separation_arcmin'].info.format = '.3f'
print(matched_table[:10])

# This matched table can now be used for further analysis, e.g., studying
# the correlation between optical richness and X-ray luminosity for clusters.

```

This final Python script demonstrates a common cross-matching task relevant to cosmology and cluster studies: associating optically selected galaxy cluster candidates with sources detected in X-ray surveys. It starts by creating dummy Astropy `Table` objects representing the optical cluster catalog (with RA, Dec, richness) and the X-ray source catalog (RA, Dec, flux). `astropy.coordinates.SkyCoord` objects are created for both catalogs. The core matching logic uses the `match_to_catalog_sky` method twice: first to find the nearest X-ray source for each optical cluster (`idx_xray`, `sep2d_opt_to_xray`), and second to find the nearest optical cluster for each X-ray source (`idx_opt`). A reliable match is established by applying two criteria: the angular separation between the pair must be less than a defined `max_separation` (e.g., 1 arcminute), and the match must be bi-directional best (i.e., optical cluster A's closest X-ray match is B, and X-ray source B's closest optical match is A). This helps eliminate chance alignments. The script identifies pairs satisfying both criteria and constructs a final `matched_table` containing combined information (IDs, coordinates, properties, separation) for the reliably associated optical clusters and X-ray sources, enabling further scientific investigation of the matched sample.

---

**References**

Allen, A., Teuben, P., Paddy, K., Greenfield, P., Droettboom, M., Conseil, S., Ninan, J. P., Tollerud, E., Norman, H., Deil, C., Bray, E., Sipőcz, B., Robitaille, T., Kulumani, S., Barentsen, G., Craig, M., Pascual, S., Perren, G., Lian Lim, P., … Streicher, O. (2022). Astropy: A community Python package for astronomy. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.6514771
*   *Summary:* This Zenodo record archives a version of the Astropy package. Its `astropy.coordinates` module provides the `SkyCoord` object and matching functionality crucial for catalog cross-matching (Section 6.6).

Annunziatella, M., Marchesini, D., Stefanon, M., Zamorani, G., Caputi, K. I., Dickinson, M., Ferrara, A., Fontana, A., Grazian, A., Koekemoer, A. M., Nonino, M., Pacifici, C., Pentericci, L., Santini, P., & Weibel, A. (2023). The UV continuum slopes ($\beta$) of galaxies at $z \sim 9$–16 from the CEERS survey. *The Astrophysical Journal Letters, 958*(1), L6. https://doi.org/10.3847/2041-8213/ad0227
*   *Summary:* This study of high-redshift galaxies relies on robust source detection (Section 6.2) and photometry (Sections 6.3, 6.4) from deep JWST imaging to derive physical properties. It illustrates the application context of the techniques discussed.

Bertin, E. (2011). Automated Morphometry with SExtractor and PSFEx. *Astronomical Data Analysis Software and Systems XX*, 435. *(Note: ASCL entry/Conf proceeding reference, pre-2020)*
*   *Summary:* Describes PSFEx, a widely used code for building spatially varying PSF models from images. Although pre-2020, it's a key reference for advanced PSF modeling techniques mentioned in Section 6.4.1.

Bradley, L., Sipőcz, B., Robitaille, T., Tollerud, E., Vinícius, Z., Deil, C., Barbary, K., Wilson, T., Busko, I., Donath, A., Günther, H. M., Cara, M., Conseil, S., Bostroem, K. A., Droettboom, M., Bray, E., Andrés, J. C., Lim, P. L., Kumar, A., … D'Eugenio, F. (2023). photutils: Photometry and related tools for Python. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.8181181
*   *Summary:* This Zenodo record archives a version of `photutils`. It provides the core functionalities for background estimation (`Background2D`), source detection (`detect_sources`, `DAOStarFinder`), aperture photometry (`aperture_photometry`), PSF photometry (`EPSFBuilder`, `BasicPSFPhotometry`, etc.), and morphology (`SourceCatalog`) central to this chapter (Sections 6.1-6.5).

Ding, X., Lin, Z., Lin, L., Pan, H.-A., Gao, Z., & Kong, X. (2023). GalaFitM: A Galfit wrapper for robust and parallel fitting of galaxy morphology. *Astronomy and Computing, 44*, 100738. https://doi.org/10.1016/j.ascom.2023.100738
*   *Summary:* Introduces a tool for detailed galaxy morphology fitting (using GALFIT). This represents the advanced morphological analysis techniques that build upon the basic parameters discussed in Section 6.5.

Gaia Collaboration, Vallenari, A., Brown, A. G. A., Prusti, T., de Bruijne, J. H. J., Arenou, F., Babusiaux, C., Biermann, M., Creevey, O. L., Ducourant, C., Evans, D. W., Eyer, L., Guerra, R., Hutton, A., Jordi, C., Klioner, S. A., Lammers, U. L., Lindegren, L., Luri, X., … Zwitter, T. (2023). Gaia Data Release 3: Summary of the contents, processing, and validation. *Astronomy & Astrophysics, 674*, A1. https://doi.org/10.1051/0004-6361/202243940
*   *Summary:* The summary paper for Gaia DR3. Gaia serves as the primary reference catalog for high-accuracy positions and photometry used in astrometric calibration (Chapter 5) and catalog cross-matching (Section 6.6).

Hatt, D., Beaton, R. L., Freedman, W. L., Hoyt, T. J., Jang, I. S., Kang, J., Lee, M. G., Madore, B. F., Monson, A. J., Rich, J. A., Scowcroft, V., Seibert, M., & Tarantino, P. (2021). The Carnegie-Chicago Hubble Program. IX. The Tip of the Red Giant Branch distances to M66 and M96 of the Leo I Group. *The Astrophysical Journal, 912*(2), 118. https://doi.org/10.3847/1538-4357/abec75
*   *Summary:* This paper relies on precise photometry of individual stars (likely PSF photometry, Section 6.4) in nearby galaxies to measure distances using the TRGB method. It illustrates the application of high-precision photometry techniques.

Nardiello, D. (2023). High-precision photometry with the James Webb Space Telescope. The NIRCam case. *Astronomy & Astrophysics, 679*, A115. https://doi.org/10.1051/0004-6361/202347246
*   *Summary:* Focuses on achieving high-precision photometry with JWST/NIRCam, discussing complexities like spatially variable PSFs and sophisticated PSF modeling/fitting techniques (Section 6.4) required for state-of-the-art instruments.

Scolnic, D., Brout, D., Carr, A., Riess, A. G., Davis, T. M., Dwomoh, A., Jones, D. O., Ali, N., Clocchiatti, A., Filippenko, A. V., Foley, R. J., Hicken, M., Hinton, S. R., Kessler, R., Lidman, C., Möller, A., Nugent, P. E., Popovic, B., Setiawan, A. K., … Wiseman, P. (2022). Measuring the Hubble Constant with Type Ia Supernovae Observed by the Dark Energy Survey Photometric Calibration System. *The Astrophysical Journal, 938*(2), 113. https://doi.org/10.3847/1538-4357/ac8e7a
*   *Summary:* Details the photometric measurements of supernovae using difference imaging techniques, which inherently rely on accurate source detection (Section 6.2) and photometry (Sections 6.3 or 6.4) on both template and science images.

Weaver, J. R., Kauffmann, O. B., Ilbert, O., Toft, S., McCracken, H. J., Zalesky, L., Capak, P., Casey, C. M., Davidzon, I., Faisst, A. L., Glazebrook, K., Gould, K. M. T., Kartaltepe, J. S., Laigle, C., McPartland, C., Mehta, V., Mobasher, B., Moneti, A., Sanders, D. B., … Whitaker, K. E. (2023). COSMOS2020: A Panchromatic Catalog Covering UV, Optical, Near-infrared, and Mid-infrared Wavelengths Generated Using the Classic Technique. *The Astrophysical Journal Supplement Series, 265*(2), 53. https://doi.org/10.3847/1538-4365/acaea0
*   *Summary:* Describes the creation of the large, multi-wavelength COSMOS2020 catalog. This process involves sophisticated source detection (Section 6.2) and consistent photometry (Sections 6.3/6.4) across multiple bands and datasets, illustrating large-scale application.
