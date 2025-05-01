---
# Chapter 9
# Data Combination and Visualization Techniques
---
![imagem](imagem.png)

*This chapter focuses on essential methodologies for synthesizing information from multiple astronomical datasets and effectively communicating scientific results through impactful visualizations. It addresses techniques required to combine images spatially, such as aligning observations taken at different times or with slight pointing offsets, and methods for stacking these aligned images to enhance signal-to-noise ratios or reject transient artifacts. The construction of large-scale image mosaics by stitching together individual exposures covering extended areas of the sky is also detailed, highlighting the challenges involved in creating seamless final products. Furthermore, the chapter explores conceptual approaches and practical tools for fusing data obtained across different wavelengths or from different observational modalities, enabling a more holistic view of astronomical objects. A significant portion is dedicated to the principles and practices of effective scientific visualization, discussing the critical choices regarding data representation, colormap selection, contrast scaling, and plot annotation required to create clear, informative, and quantitatively accurate figures suitable for analysis and publication. Various visualization techniques are presented, including methods for handling multi-dimensional data like spectral cubes and an introduction to interactive visualization tools that facilitate data exploration. Practical Python examples utilizing libraries such as `reproject`, `astropy.visualization`, `matplotlib`, and others demonstrate the implementation of these data combination and visualization strategies across diverse astronomical research contexts.*

---

**9.1 Image Registration and Alignment Algorithms (`reproject`, `astropy.wcs`)**

Astronomical research frequently requires combining information from multiple images taken at different times, with different instruments, through different filters, or as part of a dithered observing sequence or mosaic pattern. Before these images can be directly compared or combined (e.g., stacked, subtracted), they must be **registered** and **aligned** onto a common spatial coordinate system and pixel grid. Image registration is the process of finding the geometric transformation (e.g., translation, rotation, scale change, distortion correction) that maps pixel coordinates in one image (the "source" image) to the corresponding pixel coordinates in another image (the "reference" or "target" image) or onto a common output World Coordinate System (WCS) frame. Alignment then involves applying this transformation, typically through resampling or **reprojection**, to create a new version of the source image that shares the same pixel grid and coordinate system as the reference. Accurate registration and alignment are fundamental for difference imaging (transient detection), image stacking (increasing depth), mosaic construction, and multi-wavelength analysis.

The approach to registration depends on whether accurate WCS information is available for the input images:

1.  **WCS-based Registration:** If both the source and target images possess reliable FITS WCS information (Section 5.1), registration is achieved by defining a target WCS and pixel grid (which could be derived from the reference image's WCS or defined independently, e.g., a standard tangent projection centered on the field). The transformation between the source image's pixel grid and the target grid is implicitly defined by the WCS information associated with each. The **`reproject`** package, an Astropy-affiliated library, is the standard tool for performing WCS-based reprojection (Astropy Collaboration et al., 2019, 2022).
    *   **Process:** `reproject` takes the source image data array and its associated WCS object, along with the target WCS definition (either as another WCS object or a target FITS header) and the desired output array shape. It then calculates, for each pixel in the target output grid, the corresponding location (often non-integer pixel coordinates) in the source image using the inverse WCS transformation of the source image.
    *   **Interpolation:** Since the target pixel grid generally does not map directly onto integer pixel locations in the source image, an **interpolation** scheme is required to estimate the flux value at the calculated source position based on the values of neighboring pixels in the source image. `reproject` supports several interpolation methods:
        *   `'nearest-neighbor'`: Assigns the value of the nearest source pixel. Fastest but not flux-conserving and can introduce blocky artifacts. Suitable for masks or categorical data.
        *   `'bilinear'`: Uses linear interpolation in 2D based on the four nearest source pixels. Relatively fast and smoother than nearest-neighbor, but not strictly flux-conserving.
        *   `'biquadratic'`, `'bicubic'`: Higher-order spline interpolation. Can produce smoother results but may introduce ringing artifacts near sharp features and can be computationally more expensive.
        *   **Flux-Conserving Methods:** Algorithms like `'flux-conserving'` (often based on spherical polygon intersection or related techniques) aim to preserve the total flux during reprojection, which is crucial for accurate photometry on the reprojected image. These methods consider the overlap area between source pixels and target pixels but are typically the most computationally intensive. The choice of interpolation method involves a trade-off between speed, smoothness, and flux conservation. For scientific analysis requiring accurate flux measurements, flux-conserving interpolation is generally preferred, while bilinear might suffice for visualization or alignment checks.
    *   **Advantages:** Conceptually straightforward if accurate WCS exists. Utilizes all pixel information. Can handle complex distortions defined in the WCS.
    *   **Disadvantages:** Entirely dependent on the accuracy of the input WCS information. Small errors in the WCS can lead to significant misalignments.

2.  **Feature-based Registration:** If accurate WCS information is absent or unreliable in one or both images, registration must rely on identifying common features (typically stars) present in both images.
    *   **Process:**
        1.  Detect sources independently in both the source and reference images (Section 6.2).
        2.  Cross-match the detected source lists based on their *pixel coordinate patterns*, typically using algorithms robust against translation, rotation, and scale differences (e.g., triangle matching algorithms). Libraries like **`astroalign`** (Beroiz et al., 2020) implement such feature-based matching routines.
        3.  Once a set of reliably matched source pairs ($(p_{x,src}, p_{y,src})\_i \leftrightarrow (p_{x,ref}, p_{y,ref})_i$) is identified, a geometric transformation model (e.g., affine transformation including translation, rotation, scale, shear; or a polynomial transformation for non-linear effects) is fitted to map the source pixel coordinates to the reference pixel coordinates using least squares.
        4.  Apply the fitted geometric transformation (including interpolation) to the source image data to align it with the reference image pixel grid.
    *   **Advantages:** Does not require prior WCS information. Can work even if images have significantly different orientations or scales (within limits).
    *   **Disadvantages:** Requires a sufficient number of well-distributed common features (stars) in both images. Accuracy depends on the precision of source detection/centroiding and the number/distribution of matched features. May struggle with images having very few stars or significantly different fields of view. Fitting only simple transformations (like affine) may not capture complex instrumental distortions accurately.

**Alignment (Reprojection):** Regardless of whether the transformation was derived from WCS or feature matching, the final step is to **resample** the source image onto the target pixel grid using the determined transformation and an appropriate interpolation algorithm. The `reproject` package can perform the reprojection step itself, even if the initial transformation was derived using feature matching (by constructing an effective WCS or directly providing the transformation function). Proper handling of image masks and uncertainty arrays during reprojection is also crucial and often supported by tools like `reproject` when operating on `NDData`-like objects. The result of successful registration and alignment is a set of images sharing a common pixel grid and coordinate system, ready for further combination or comparison.

**9.2 Image Stacking and Co-addition**

Combining multiple, aligned images of the same sky region is a fundamental technique used to significantly increase the effective depth of observations, enhance the signal-to-noise ratio (SNR) for faint sources, and mitigate the impact of transient artifacts like cosmic rays or cosmetic detector defects (Holwerda, 2021; Annis et al., 2014). This process is variously referred to as **stacking**, **co-addition**, or **image combination**. It leverages the fact that the astronomical signal from celestial sources is constant across the aligned frames, while random noise and transient events are uncorrelated between exposures.

The core principle involves calculating a statistically robust average pixel value at each position $(x, y)$ in the final combined image grid, based on the corresponding pixel values from the stack of $N$ input images ($I_i(x, y)$ for $i=1...N$) that have already been registered and aligned (Section 9.1). Several combination algorithms are commonly used:

1.  **Mean (Average) Combination:** The simplest method is to calculate the arithmetic mean of the pixel values across the stack:
    $\bar{I}\_{mean}(x, y) = \frac{1}{N} \sum_{i=1}^{N} I_i(x, y)$
    *   **SNR Improvement:** If the noise ($\sigma_{single}$) in each input image is random and uncorrelated, the noise in the mean-combined image is reduced by a factor of $\sqrt{N}$: $\sigma_{mean} = \sigma_{single} / \sqrt{N}$. This leads to a $\sqrt{N}$ improvement in SNR for background-limited faint sources.
    *   **Disadvantages:** The mean is highly sensitive to outlier pixel values. A single cosmic ray hit or unmasked bad pixel in one input frame can significantly contaminate the corresponding pixel in the final average image. Therefore, simple mean combination is rarely used without prior or simultaneous outlier rejection.

2.  **Median Combination:** Calculates the median value of the pixel stack at each position:
    $\bar{I}_{median}(x, y) = \mathrm{median} \{ I_i(x, y) \}$
    *   **Outlier Rejection:** The median is inherently robust against outliers. As long as fewer than 50% of the input frames are affected by an outlier (e.g., cosmic ray) at a given pixel, the median value will provide a reliable estimate of the true underlying signal. This makes median combination highly effective for rejecting cosmic rays and other transient artifacts without explicit prior masking, especially when $N \ge 5$.
    *   **SNR Improvement:** The noise reduction is slightly less efficient than the mean for purely Gaussian noise ($\sigma_{median} \approx 1.253 \sigma_{single} / \sqrt{N}$ for large $N$), but the robustness against outliers often makes it the preferred choice, particularly when dealing with cosmic rays.
    *   **Disadvantages:** Can slightly broaden the effective PSF compared to mean combination if images have slight registration errors. Computationally slightly more expensive than the mean.

3.  **Sigma-Clipped Mean Combination:** An attempt to gain the noise advantage of the mean while incorporating outlier rejection. At each pixel $(x, y)$, the mean and standard deviation of the stack values $\{ I_i(x, y) \}$ are calculated. Values deviating by more than a specified threshold (e.g., 3-sigma) are temporarily masked. The mean is then recalculated using only the remaining unmasked values. This process can be iterated.
    *   **Advantages:** Can achieve near $\sqrt{N}$ noise reduction if outliers are effectively removed.
    *   **Disadvantages:** Performance depends critically on the choice of sigma threshold and the underlying noise distribution. Can erroneously clip pixels belonging to faint real sources or the cores of bright stars if the noise model or threshold is incorrect. Requires careful tuning.

4.  **Weighted Mean Combination:** If the input images have different exposure times or varying quality (e.g., different background noise levels or seeing conditions), a weighted mean combination can produce a higher SNR result than a simple mean or median. Each image $I_i$ is assigned a weight $w_i$ (typically inversely proportional to its variance, $w_i \propto 1/\sigma_i^2$). The combined image is:
    $\bar{I}\_{weighted}(x, y) = \frac{\sum_{i=1}^{N} w_i I_i(x, y)}{\sum_{i=1}^{N} w_i}$
    Calculating appropriate weights requires accurate estimates of the variance (noise) in each input image, potentially including contributions from background noise, read noise, and Poisson noise from sources. Combining weighting with outlier rejection (e.g., sigma-clipped weighted mean) is often employed in sophisticated pipelines.

**Implementation (`ccdproc.Combiner`):** The **`ccdproc.Combiner`** class (introduced in Chapter 3 for calibration frames) is also ideally suited for combining science images. It takes a list of aligned `CCDData` objects as input. Users can configure various options:
*   `combine_method`: `'average'`, `'median'`, `'sum'`.
*   Outlier Rejection: Methods like `.sigma_clipping()`, `.minmax_clipping()` can be applied before combination.
*   Weighting: Weights can be assigned based on exposure time (`scale` keyword, often related to `EXPTIME`) or inverse variance (`weight` keyword, requires uncertainty information in input `CCDData`).
*   Memory Management: Options exist for handling large datasets that may not fit entirely in memory.

**Considerations:**
*   **Alignment Accuracy:** Sub-pixel alignment errors between input frames will effectively broaden the PSF in the final combined image, degrading resolution.
*   **Background Leveling:** If input images have slightly different background levels (e.g., due to varying sky conditions between exposures), these differences should ideally be removed or matched before combination to avoid introducing artifacts or large-scale gradients.
*   **Flux Scaling:** If images have different exposure times or transparency conditions, they must be scaled to a common flux level (e.g., counts per second) before combination.
*   **Uncertainty Propagation:** When combining images, the uncertainties must also be propagated correctly. `ccdproc.Combiner` can handle basic uncertainty propagation based on the combination method and input uncertainties (if provided in the `CCDData` objects). The variance in the final combined image depends on the combination algorithm and input variances. For example, for an average combine, $\sigma^2_{avg} = (\sum \sigma_i^2) / N^2$. For a median combine, estimating the output variance is more complex but often approximated.

Image stacking is a powerful technique for reaching fainter detection limits and improving measurement precision. Choosing the optimal combination algorithm depends on the data characteristics (number of frames, presence of outliers, varying quality) and scientific goals. Median combination is often a robust default choice for dealing with cosmic rays when sufficient frames ($N \ge 5$) are available.

The following Python code demonstrates stacking multiple aligned images using `ccdproc.Combiner`. It simulates reading several processed and aligned FITS files, combines them using median with sigma clipping for outlier rejection, and saves the resulting deeper, cleaner image.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping image stacking example.")
    ccdproc_available = False
import astropy.units as u
from astropy.stats import SigmaClip # For combiner
import os

# --- Input Files (List of Aligned, Reduced Images) ---
# Assume these files exist from previous reduction and alignment steps
input_files = [f'galaxy_frame_{i+1:02d}_aligned_redux.fits' for i in range(5)] # Use files from 6.7.6 example
output_stacked_file = 'galaxy_stacked_image.fits'

# --- Re-create dummy files if needed (from 6.7.6 example) ---
if ccdproc_available:
    # (Check and re-create dummy aligned files if they don't exist)
    # This part assumes the dummy file creation logic from Example 6.7.6 exists
    # or is executed beforehand. We'll just check for existence here.
    all_files_exist = True
    for fname in input_files:
        if not os.path.exists(fname):
            print(f"Warning: Input file {fname} not found. Stacking may fail or use fewer files.")
            all_files_exist = False
            # Optional: Add code here to re-generate the dummy file if desired for demo
            # break # Stop if any file is missing for simplicity

    if not all_files_exist and len(os.listdir('.')) < 10 : # Basic check if directory is empty
        print("Error: Input files for stacking not found. Please run Example 6.7.6 first or provide files.")
        ccdproc_available = False # Disable example if files missing


if ccdproc_available:
    try:
        # --- Load Aligned Images into CCDData List ---
        print(f"Loading {len(input_files)} aligned images for stacking...")
        ccd_list = []
        for f in input_files:
            try:
                # Read as CCDData, assuming units and potentially uncertainty/mask
                # If uncertainties are present, Combiner can propagate them
                ccd = CCDData.read(f) # Auto-detects units from BUNIT if present
                if ccd.unit is None: ccd.unit = u.electron # Assign default if needed
                ccd_list.append(ccd)
            except FileNotFoundError:
                print(f"Warning: Skipping missing file {f}")
            except Exception as read_err:
                print(f"Warning: Error reading file {f}: {read_err}. Skipping.")

        if not ccd_list:
            raise ValueError("No valid images loaded for stacking.")
        if len(ccd_list) < 3:
             print("Warning: Stacking fewer than 3 images offers limited outlier rejection.")

        # --- Combine Images using ccdproc.Combiner ---
        print(f"Combining {len(ccd_list)} images...")
        # Initialize Combiner with the list of CCDData objects
        combiner = ccdproc.Combiner(ccd_list)

        # Configure outlier rejection (optional but recommended)
        # Use sigma clipping before median/average combination
        sigma_clip = SigmaClip(low_thresh=3.0, high_thresh=3.0, func=np.ma.median, dev_func=np.ma.std)
        combiner.sigma_clipping(sigma_clip=sigma_clip)
        print("  (Sigma clipping enabled for outlier rejection)")

        # Choose combination method - median is robust for cosmic rays
        # Average combine could be used if CRs were perfectly masked beforehand
        print("  (Using median combine method)")
        stacked_image = combiner.median_combine()
        # Alternatively, for average combination:
        # stacked_image = combiner.average_combine()

        # The resulting stacked_image is a CCDData object.
        # It might contain propagated uncertainty and combined mask if inputs had them.

        # Update metadata in the stacked image header
        stacked_image.meta['NCOMBINE'] = len(ccd_list)
        stacked_image.meta['COMBTYPE'] = 'Median (Sigma Clip)'
        stacked_image.meta['HISTORY'] = f'Stacked {len(ccd_list)} aligned reduced exposures.'

        # --- Save the Stacked Image ---
        print(f"Saving stacked image...")
        # stacked_image.write(output_stacked_file, overwrite=True)
        print(f"(If successful, stacked image would be saved to {output_stacked_file})")
        print(f"Resulting stacked image shape: {stacked_image.shape}, Mean value: {np.mean(stacked_image.data):.2f}")

    except ImportError:
        print("Error: ccdproc library is required but not found.")
    except ValueError as e:
         print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during image stacking: {e}")
else:
     print("Skipping image stacking example: ccdproc unavailable or input files missing.")
```

This Python script demonstrates the process of combining multiple, previously reduced and spatially aligned images of the same sky field to create a deeper final image with enhanced signal-to-noise and reduced artifacts, using the `ccdproc` library. It begins by loading a list of input FITS files (representing the individual aligned exposures) into a list of `CCDData` objects. A `ccdproc.Combiner` object is then initialized with this list. To ensure robustness against remaining cosmic rays or other transient outliers, sigma clipping is enabled using `combiner.sigma_clipping`. The script then performs the image combination using the `combiner.median_combine` method, which is highly effective at rejecting outlier pixel values present in only a minority of the input frames. The resulting `stacked_image` is a `CCDData` object representing the combined, deeper view of the field, with metadata updated to reflect the number of combined frames and the method used. This stacked image is suitable for detecting fainter sources or performing more precise measurements than possible with individual exposures.

**9.3 Mosaic Construction for Large Sky Areas (`reproject`, `montage-wrapper`)**

Many astronomical surveys aim to map large contiguous areas of the sky, far exceeding the field of view of a single detector exposure. This is achieved by **mosaicking**, where multiple individual images (often overlapping or dithered) are combined to create a single, seamless large image covering the full survey region (Jacob et al., 2010; Descamps et al., 2015). Constructing high-quality mosaics involves several challenges beyond simple stacking, including precise relative and absolute astrometry across all input frames, handling varying background levels and photometric zero points between pointings, correcting for potentially complex geometric distortions that vary across detectors or wide fields, and resampling all images onto a common final projection and pixel grid.

Key steps in mosaic construction include:
1.  **Data Preparation:** All input images must undergo basic reduction (bias, dark, flat) and potentially initial cosmic ray cleaning. Accurate WCS information is crucial; often, a global astrometric solution is performed simultaneously across all input frames using tools like `SCAMP` (Bertin, 2006) to ensure high relative astrometric accuracy between overlapping images before reprojection. Photometric calibration (determining zero points, Section 5.4) for each input image might also be necessary for proper flux scaling during combination.
2.  **Defining the Output Frame:** A target FITS header defining the final mosaic's WCS (coordinate system, projection type, reference point, pixel scale) and dimensions must be created. The projection type (e.g., TAN, SIN, CAR) needs to be chosen carefully based on the sky area covered and the desired geometric properties (e.g., preserving angles locally vs. preserving area). The pixel scale determines the resolution of the final mosaic.
3.  **Reprojection:** Each individual input image must be reprojected onto the common output frame defined by the target header. This involves using the input image's WCS and the target WCS to transform and resample the image data, typically using flux-conserving interpolation algorithms provided by libraries like `reproject` (Section 9.1). This step transforms all input images onto the same pixel grid and coordinate system.
4.  **Background Matching/Leveling:** Since input images might have been taken under different sky conditions or have imperfect background subtraction, overlapping regions between reprojected images often show discontinuities or "seams." Background matching algorithms are employed to estimate and remove these residual background differences, often by fitting smooth functions to the differences in overlapping regions and applying corrections to ensure a seamless transition between adjacent frames.
5.  **Co-addition/Combination:** The reprojected, background-matched images are then combined to create the final mosaic image. In overlapping regions, pixel values from multiple input images contribute to the final pixel value. Similar to stacking (Section 9.2), combination methods like median or weighted mean (based on input image noise variance or exposure depth maps) are used. Median combination helps reject remaining outliers, while weighted averaging can optimize SNR. The combination method needs to handle pixels covered by only one image versus those covered by multiple images consistently. An output weight map, indicating the effective exposure depth or number of contributing frames at each pixel, is often generated alongside the science mosaic.

**Software Tools:**
*   **`reproject`:** Handles the core task of reprojecting individual images onto the target frame based on WCS information.
*   **`Montage`:** A widely used, dedicated software toolkit (originally developed by NASA/IPAC) specifically designed for astronomical image mosaicking (Berriman et al., 2003; Jacob et al., 2010). It includes modules for background modeling and rectification (`mBackground`, `mImgtbl`, `mProjExec`, `mDiff`, `mFitplane`), reprojection (`mProject`), and co-addition (`mAdd`). `Montage` is known for its robust background matching capabilities and handling of FITS WCS standards. Python wrappers like **`montage-wrapper`** (Astropy Project) provide convenient interfaces to run Montage tools from within Python scripts.
*   **SWarp:** Another popular command-line tool for image resampling, co-addition, and mosaicking, often used in survey pipelines (Bertin et al., 2002).

Creating large, high-quality mosaics requires careful handling of astrometry, photometry, background variations, and computational resources, often involving specialized software pipelines tailored to the specific survey data and scientific goals.

The following code provides a conceptual example using `reproject` to reproject two simulated, partially overlapping images onto a common output WCS frame, illustrating the first step towards creating a mosaic. It does not perform the background matching or final co-addition steps, which typically require more specialized tools like Montage.

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
# Requires reproject: pip install reproject
try:
    from reproject import reproject_interp # Choose interpolation method
    reproject_available = True
except ImportError:
    print("reproject not found, skipping mosaic reprojection example.")
    reproject_available = False
import matplotlib.pyplot as plt
import os

# --- Input: Simulate two overlapping images with WCS ---
image1_file = 'mosaic_img1.fits'
image2_file = 'mosaic_img2.fits'
output_mosaic_frame_file = 'mosaic_target_header.hdr' # Header defining output grid
output_repr1_file = 'mosaic_img1_reprojected.fits'
output_repr2_file = 'mosaic_img2_reprojected.fits'

if reproject_available:
    # Create Image 1
    if not os.path.exists(image1_file):
        print(f"Creating dummy file: {image1_file}")
        im_size = (100, 100)
        data1 = np.random.normal(10, 1, size=im_size)
        data1[30:70, 30:70] += 20 # Add a feature
        w1 = WCS(naxis=2)
        w1.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w1.wcs.crpix = [50.5, 50.5]
        w1.wcs.crval = [150.0, 20.0] # RA/Dec Center 1
        w1.wcs.cdelt = np.array([-0.0002, 0.0002]) # deg/pix
        hdu1 = fits.PrimaryHDU(data1.astype(np.float32), header=w1.to_header())
        hdu1.writeto(image1_file, overwrite=True)
    # Create Image 2 (partially overlapping)
    if not os.path.exists(image2_file):
        print(f"Creating dummy file: {image2_file}")
        im_size = (100, 100)
        data2 = np.random.normal(12, 1, size=im_size) # Slightly different background
        data2[40:80, 40:80] += 25 # Add another feature
        w2 = WCS(naxis=2)
        w2.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w2.wcs.crpix = [50.5, 50.5]
        w2.wcs.crval = [150.0 - 0.015, 20.0 + 0.005] # Offset center slightly
        w2.wcs.cdelt = np.array([-0.0002, 0.0002])
        hdu2 = fits.PrimaryHDU(data2.astype(np.float32), header=w2.to_header())
        hdu2.writeto(image2_file, overwrite=True)

    # --- Define Output Mosaic Frame ---
    # Create a target FITS header defining the output grid WCS and dimensions
    # Can be derived from input WCS or defined independently
    # For simplicity, use WCS of image 1 but expand dimensions
    print("Defining output mosaic frame...")
    from astropy.wcs.utils import fit_wcs_from_points, skycoord_to_pixel
    from reproject.mosaicking import find_optimal_celestial_wcs # Helper function

    # Use helper function to find optimal WCS covering both input WCS footprints
    # Or manually define a larger WCS grid centered between the inputs
    # Manual Example: Center between w1 and w2, make it larger
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    target_wcs.wcs.crval = [w1.wcs.crval[0] - 0.0075, w1.wcs.crval[1] + 0.0025] # Center point
    target_wcs.wcs.cdelt = w1.wcs.cdelt # Use same pixel scale
    target_shape = (150, 150) # Larger output dimensions
    target_wcs.wcs.crpix = [target_shape[1]/2.0 + 0.5, target_shape[0]/2.0 + 0.5]
    target_header = target_wcs.to_header()
    # Save target header (optional)
    # target_header.totextfile(output_mosaic_frame_file, overwrite=True)
    print(f"  Target Frame: Shape={target_shape}, Center={target_wcs.wcs.crval}, Scale={target_wcs.wcs.cdelt}")

    # --- Reproject Input Images onto Output Frame ---
    try:
        print(f"Reprojecting {image1_file}...")
        # Load image 1 data and WCS
        with fits.open(image1_file) as hdul1:
             data1, header1 = hdul1[0].data, hdul1[0].header
        # Reproject using bilinear interpolation (faster) or 'flux-conserving'
        array1_repr, footprint1 = reproject_interp((data1, header1), target_header,
                                                   shape_out=target_shape, return_footprint=True)
        # Save reprojected image 1
        # hdu_repr1 = fits.PrimaryHDU(data=array1_repr.astype(np.float32), header=target_header)
        # hdu_repr1.writeto(output_repr1_file, overwrite=True)
        print(f"  Reprojection of {image1_file} complete.")

        print(f"Reprojecting {image2_file}...")
        # Load image 2 data and WCS
        with fits.open(image2_file) as hdul2:
             data2, header2 = hdul2[0].data, hdul2[0].header
        # Reproject image 2
        array2_repr, footprint2 = reproject_interp((data2, header2), target_header,
                                                   shape_out=target_shape, return_footprint=True)
        # Save reprojected image 2
        # hdu_repr2 = fits.PrimaryHDU(data=array2_repr.astype(np.float32), header=target_header)
        # hdu_repr2.writeto(output_repr2_file, overwrite=True)
        print(f"  Reprojection of {image2_file} complete.")

        # --- Simple Combination (Ignoring Background Matching) ---
        # Conceptual: Combine using average where both contribute
        # Need weights or masks from footprints
        # combined_data = np.nanmean(np.stack([array1_repr, array2_repr]), axis=0) # Basic average
        print("\nConceptual Combination Step:")
        print("  (Skipping proper background matching and weighted co-addition)")
        print("  The arrays 'array1_repr' and 'array2_repr' are now on the same grid.")

        # --- Optional: Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, subplot_kw={'projection': target_wcs})
        axes[0].imshow(array1_repr, origin='lower', cmap='viridis', vmin=5, vmax=35)
        axes[0].set_title('Reprojected Image 1')
        axes[0].coords.grid(True, color='white', ls='dotted', alpha=0.5)
        axes[1].imshow(array2_repr, origin='lower', cmap='viridis', vmin=5, vmax=35)
        axes[1].set_title('Reprojected Image 2')
        axes[1].coords.grid(True, color='white', ls='dotted', alpha=0.5)
        # Visualize a simple combination (e.g., average) - NOTE: ignores background differences!
        combined_simple = np.nanmean(np.stack([np.where(footprint1>0, array1_repr, np.nan),
                                            np.where(footprint2>0, array2_repr, np.nan)]), axis=0)
        axes[2].imshow(combined_simple, origin='lower', cmap='viridis', vmin=5, vmax=35)
        axes[2].set_title('Simple Average (Conceptual)')
        axes[2].coords.grid(True, color='white', ls='dotted', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Error: reproject library is required but not found.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred during reprojection: {e}")
else:
     print("Skipping mosaic reprojection example: reproject unavailable or dummy data missing.")
```

This Python script illustrates the fundamental reprojection step involved in constructing an astronomical image mosaic, using the `reproject` library. It simulates two partially overlapping input images, each with its own distinct WCS information stored in a FITS header. A target output frame is defined, specifying the desired WCS (coordinate system, projection, pixel scale) and dimensions for the final mosaic. The core of the example uses the `reproject.reproject_interp` function (other interpolation methods like `reproject_exact` for flux conservation exist) to transform each input image onto this common output grid. The function takes the input image data and header (containing the WCS) and the target output header (or WCS object and shape) and performs the necessary coordinate transformations and interpolation to create new image arrays (`array1_repr`, `array2_repr`) that are aligned on the same pixel grid defined by the target WCS. While this example shows the reprojection, it notes that subsequent crucial steps for creating a science-ready mosaic, namely careful background matching between the reprojected images and weighted co-addition in overlapping regions (often handled by tools like Montage), are needed but not explicitly implemented here. The visualization displays the two reprojected images side-by-side on the common WCS grid.

**9.4 Multi-wavelength Data Fusion Concepts**

Astronomical objects often emit radiation across a wide range of the electromagnetic spectrum (and potentially via other messengers like gravitational waves or neutrinos), with different wavelengths tracing distinct physical components or processes. Combining data from multiple wavelengths provides a more complete and physically insightful picture than analyzing any single wavelength in isolation (Spinoglio et al., 2021; Padovani et al., 2023). For example, combining optical images showing stars and ionized gas with infrared images tracing cool dust and radio data revealing synchrotron emission or molecular gas allows comprehensive studies of star formation, galaxy structure, or AGN feedback. **Multi-wavelength data fusion**, in the context of imaging, primarily involves techniques for visually representing and spatially comparing aligned datasets obtained at different wavelengths.

Common visualization approaches include:
1.  **RGB Color Composites:** This is perhaps the most intuitive method for combining three different image datasets (typically representing different filters or wavelength ranges) into a single color image. Each input image, after being aligned to a common WCS and pixel grid (Section 9.1), is assigned to one of the three color channels: Red (R), Green (G), and Blue (B). The intensity scaling and stretching applied to each channel individually before combination significantly affect the final color balance and visibility of different features.
    *   **Process:**
        *   Align the three input images (e.g., representing red, green, and blue filter observations, or mapping different physical components like H$\alpha$ to red, [O III] to green, continuum to blue).
        *   Apply appropriate intensity scaling/stretching (e.g., linear, logarithmic, asinh) to each image independently to bring out features of interest and handle potentially large dynamic ranges.
        *   Combine the scaled images into an RGB array (e.g., using `numpy.stack` or dedicated functions).
    *   **Tools:** The **`astropy.visualization.make_lupton_rgb`** function provides a sophisticated implementation based on the algorithm developed for SDSS color images (Lupton et al., 2004). It allows flexible scaling (e.g., asinh stretch) and control over parameters like `minimum` intensity level and `stretch` factor to optimize the color balance and detail visibility. Other tools like `aplpy` also offer RGB plotting capabilities.
    *   **Interpretation:** Colors in the resulting image represent the relative brightness of the source in the different input bands. For example, regions bright in the red-assigned filter but faint in others will appear red. Interpretation requires knowing which physical tracer corresponds to each color channel.

2.  **Contour Overlays:** Useful for comparing a specific emission component (e.g., radio continuum, X-ray emission, molecular gas lines) with a broader background image (e.g., optical or infrared).
    *   **Process:**
        *   Align the two images (e.g., radio map and optical image) to the same WCS and pixel grid. One image serves as the base grayscale or color image.
        *   Generate contour levels from the second image (e.g., radio map), typically defined at specific multiples of the image RMS noise or at physically meaningful flux levels.
        *   Overlay these contours onto the base image using plotting library functions (e.g., `matplotlib.pyplot.contour`).
    *   **Tools:** `matplotlib.pyplot.contour` is the standard tool, often used in conjunction with plots generated using WCSAxes (as shown in Example 6.7.7). Libraries like `aplpy` also simplify contour overlays on astronomical images with WCS.
    *   **Interpretation:** Contours clearly delineate the spatial extent and morphology of the overlaid component relative to the features visible in the base image, highlighting correlations or anti-correlations.

3.  **Transparency/Alpha Blending:** One image can be overlaid on another using transparency (alpha channel) to visually blend the information. For example, a map of dust extinction could be overlaid with partial transparency onto an image of stars to show where the stars are obscured. The degree of transparency can be adjusted to emphasize one layer over the other. This is often achieved using the `alpha` parameter in plotting functions like `matplotlib.pyplot.imshow`.

4.  **Linked Views / Multi-Panel Plots:** Displaying aligned images from different wavelengths side-by-side in separate panels, potentially linked such that zooming or panning in one affects the others, allows for detailed comparison while preserving the original data representation of each image. Tools like `glue-viz` excel at this linked-view exploration.

Effective multi-wavelength data fusion requires careful alignment (Section 9.1) and thoughtful choices about how different datasets are represented visually (color mapping, scaling, contours, transparency) to best highlight the scientific question being addressed without creating misleading artifacts or obscuring important information.

**9.5 Principles of Effective Scientific Visualization**

Creating informative and quantitatively accurate visualizations is a critical skill in computational astrophysics, essential for both exploring data to gain insights and communicating results clearly and effectively to peers and the wider community (Rougier et al., 2014; Hunter, 2007; Christensen et al., 2022). A poorly designed plot can obscure important features, mislead the viewer, or hinder reproducibility. Adhering to established principles of effective scientific visualization ensures that plots are not only aesthetically pleasing but also scientifically rigorous and easily interpretable. Key principles include:

*   **9.5.1 Colormap Selection and Image Stretches (`matplotlib`)**
    The choice of colormap and the intensity scaling (stretch) applied significantly impact how features are perceived in images or heatmaps.
    *   **Colormaps:**
        *   **Avoid Rainbow Colormaps (e.g., `jet`):** Rainbow colormaps are perceptually non-uniform; equal steps in data value do not correspond to equal steps in perceived color brightness or hue. This can create false visual boundaries or gradients and obscure subtle features. They are also not friendly to individuals with color vision deficiencies and do not translate well to grayscale printing (Borland & Taylor, 2007).
        *   **Use Perceptually Uniform Colormaps:** Sequential colormaps like `viridis` (the `matplotlib` default), `plasma`, `inferno`, `magma`, or `cividis` are designed such that perceived brightness increases monotonically with the data value. This provides a more accurate representation of the data magnitude. For data where deviations from a central value are important (e.g., velocity fields, difference images), diverging colormaps like `coolwarm`, `RdBu`, or `seismic` (with a neutral color at the midpoint) are appropriate. Qualitative colormaps (`tab10`, `Set1`) are suitable for categorical data but not continuous data.
    *   **Intensity Scaling/Stretching:** Astronomical images often have a very large dynamic range (ratio of brightest to faintest features). Simple linear scaling often fails to reveal both bright cores and faint extended structures simultaneously. Non-linear scaling functions ("stretches") are used to compress the dynamic range for display:
        *   **Linear (`LinearStretch`):** Direct mapping of data values to color intensity. Good for data with limited dynamic range or when preserving linear relationships is paramount.
        *   **Logarithmic (`LogStretch`):** Compresses high values, enhancing faint features. Suitable for data spanning many orders of magnitude (e.g., galaxy images). $I_{display} \propto \log(a I_{data} + 1)$.
        *   **Square Root (`SqrtStretch`):** Intermediate compression between linear and log. $I_{display} \propto \sqrt{I_{data}}$.
        *   **Power Law (`PowerStretch`, `PowerDistStretch`):** $I_{display} \propto I_{data}^\gamma$. $\gamma < 1$ compresses bright end, $\gamma > 1$ expands bright end.
        *   **Arcsine Hyperbolic (`AsinhStretch`):** Behaves linearly for low values (preserving faint details and noise characteristics) and logarithmically for high values (compressing bright features). Often provides a good balance for typical astronomical images. $I_{display} \propto \mathrm{asinh}(I_{data}/\alpha)$.
        *   **Histogram Equalization (`HistEqStretch`):** Stretches intensity levels based on the image histogram to maximize contrast across all intensity levels. Can sometimes reveal subtle features but significantly alters the perceived relationship between pixel values.
    *   **Interval Selection:** In addition to the stretch function, selecting the minimum (`vmin`) and maximum (`vmax`) data values that map to the bottom and top of the colormap is crucial for contrast.
        *   **Min/Max:** Using the absolute minimum and maximum pixel values is often dominated by noise or saturated pixels.
        *   **Percentile Clipping:** Mapping, e.g., the 1st and 99th percentile values to the colormap range provides a more robust view.
        *   **ZScale (`ZScaleInterval`):** An algorithm commonly used in astronomy (implemented in IRAF and `astropy.visualization`) that iteratively calculates robust estimates of the median and standard deviation (or related scale) to determine optimal `vmin` and `vmax` values, effectively handling outliers and providing good contrast for typical astronomical images.
    The `astropy.visualization` module provides classes for these stretch functions (`LinearStretch`, `LogStretch`, `AsinhStretch`, etc.) and interval calculations (`ManualInterval`, `PercentileInterval`, `ZScaleInterval`) that integrate seamlessly with `matplotlib`'s `ImageNormalize` for applying these transformations during plotting with `imshow`.

*   **9.5.2 Publication-Quality Plot Generation (`matplotlib`)**
    Plots intended for publication require meticulous attention to detail to ensure clarity, reproducibility, and adherence to journal standards. `matplotlib` offers extensive customization capabilities:
    *   **Labels and Titles:** All axes must have clear, descriptive labels including physical units (e.g., "Wavelength (Å)", "Flux Density (erg s⁻¹ cm⁻² Å⁻¹)", "RA (deg)"). The plot should have an informative title. Legends are essential for plots with multiple datasets or symbols, clearly identifying each component.
    *   **Font Sizes:** Use sufficiently large font sizes for labels, titles, legends, and ticks so they remain legible when the figure is scaled down for publication. Consistency in font style and size is important.
    *   **Tick Marks and Grid:** Ensure tick marks are clearly visible and appropriately spaced. Minor ticks can improve readability. A subtle grid (`ax.grid(True, alpha=0.3, linestyle=':')`) can aid in reading values but should not dominate the plot.
    *   **Line Styles and Markers:** Use distinct line styles (solid, dashed, dotted), colors, and marker types (circles, squares, triangles) to differentiate multiple datasets clearly. Avoid using yellow or light colors that may be difficult to see or reproduce in print. Consider colorblind-friendly palettes.
    *   **Resolution and File Format:** Save figures in high-resolution vector formats (e.g., PDF, EPS, SVG) whenever possible. Vector formats scale without loss of quality and are preferred by most journals. For images or very complex plots where vector formats become too large, use high-resolution raster formats (e.g., PNG, TIFF) with sufficient DPI (e.g., 300 DPI or higher). Avoid lossy formats like JPG for scientific plots.
    *   **Aspect Ratio and Layout:** Choose an appropriate aspect ratio for the plot. Use `plt.tight_layout()` or `fig.subplots_adjust()` to prevent labels or titles from overlapping. For multi-panel figures (`plt.subplots`), ensure consistent axis limits and clear labeling for each panel.
    *   **Colorbars:** For images/heatmaps, always include a clearly labeled colorbar indicating the mapping between color and data value, including units.
    *   **Code Accessibility:** For reproducibility, consider making the plotting code available alongside the publication (see Chapter 16).

*   **9.5.3 Data Cube Visualization (3D Data)**
    Visualizing 3D datasets, such as those from Integral Field Unit (IFU) spectroscopy or numerical simulations, presents unique challenges. Standard 2D plots are insufficient to capture the full information content. Common techniques include:
    *   **Image Slices:** Displaying 2D slices through the cube along one axis (e.g., showing the spatial image at a single wavelength channel, or a position-velocity slice). Requires interactive tools or generating multiple panels.
    *   **Moment Maps:** Collapsing the spectral dimension to create 2D maps representing integrated properties. Common moment maps include:
        *   Moment 0: Integrated intensity map (sum of flux along the spectral axis).
        *   Moment 1: Intensity-weighted mean velocity map (reveals kinematics).
        *   Moment 2: Intensity-weighted velocity dispersion map (reveals turbulence or rotation).
    *   **Projections:** Projecting the 3D data onto a 2D plane (e.g., maximum intensity projection).
    *   **Volume Rendering:** Advanced techniques that treat the data cube as a semi-transparent volume, rendering integrated views that can reveal complex 3D structures.
    *   **Specialized Tools:** Libraries and applications are specifically designed for IFU/simulation data visualization:
        *   **`APLpy`:** Excellent for plotting FITS images and cube slices with WCS overlays, contours, etc. (Robitaille & Bressert, 2012).
        *   **`yt`:** A powerful toolkit primarily for analyzing and visualizing volumetric data from astrophysical simulations, offering slicing, projections, volume rendering, and quantitative analysis capabilities (Turk et al., 2011).
        *   **`glue-viz`:** A versatile, interactive linked-view visualization tool for exploring relationships within and between multiple datasets, including 3D cubes (Beaumont et al., 2015; Robitaille et al., 2017). Allows linked slicing and dicing of cubes alongside other plots (histograms, scatter plots).
        *   **Spectral Cube Viewers:** Dedicated applications like QFitsView, SAOImageDS9 (with cube capabilities), or CARTA (for radio cubes) provide interactive interfaces for exploring data cubes.

*   **9.5.4 Interactive Visualization Tools**
    While static plots generated by `matplotlib` are essential for publication, interactive visualizations offer powerful capabilities for data exploration and discovery. Interactive plots allow users to zoom, pan, hover over points to get information, select data subsets, and potentially link multiple plots together. Several Python libraries facilitate interactive plotting:
    *   **`Bokeh`:** Creates interactive plots and dashboards deployable in web browsers. Excellent for generating linked plots, custom interactions, and web applications (Bokeh Development Team, 2023).
    *   **`Plotly` (and `plotly.py`):** Produces rich, interactive charts (scatter, line, contour, 3D surface/scatter, etc.) suitable for web embedding and dashboards. Offers both online and offline capabilities (Plotly Technologies Inc., 2015).
    *   **`bqplot`:** A plotting library built specifically for the Jupyter Notebook environment, leveraging browser capabilities for interactivity (e.g., panning, zooming, selections) directly within notebooks (BQplot Development Team, 2021). Integrates well with `ipywidgets` for building interactive controls.
    *   **`ipympl`:** Enables interactive features (zoom, pan) for standard `matplotlib` plots directly within Jupyter environments.
    These tools can significantly enhance the data exploration process, allowing researchers to dynamically investigate relationships and features that might be missed in static visualizations.

By carefully considering colormaps, scaling, annotation, and choosing appropriate tools for the data dimensionality and desired level of interaction, astronomers can create visualizations that are both scientifically insightful and effectively communicate their findings.

**9.6 Examples in Practice (Python): Data Combination & Visualization**

The following examples demonstrate practical applications of the data combination and visualization techniques discussed in this chapter. They cover tasks such as creating composite solar images, mosaicking planetary data, stacking deep stellar images, visualizing exoplanet pixel data, generating multi-color astronomical images, overlaying radio and optical data for galaxies, and visualizing slices from cosmological simulations. These examples utilize libraries like `reproject`, `astropy.visualization`, `matplotlib`, `lightkurve`, and `yt` to illustrate common workflows.

**9.6.1 Solar: Composite Image Creation (SDO/HMI on AIA)**
Combining data from different instruments observing the Sun simultaneously provides complementary views of solar phenomena. For example, overlaying magnetic field contours from SDO/HMI onto an EUV image from SDO/AIA can reveal the relationship between magnetic structures and coronal loops or flares. This requires aligning the two datasets using their WCS information. This example demonstrates loading an AIA image and an HMI magnetogram (assumed to cover the same time and approximate region), aligning them based on their WCS using `reproject` (conceptually, as precise alignment might require specialized solar coordinate handling via `sunpy`), and plotting the HMI contours over the AIA image.

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
# Requires reproject (optional, if alignment needed): pip install reproject
try:
    from reproject import reproject_interp
    reproject_available = True
except ImportError:
    print("reproject not found, assuming images are already aligned for Solar example.")
    reproject_available = False
# Requires sunpy for plotting with coordinates: pip install sunpy
try:
    import sunpy.map
    sunpy_available = True
except ImportError:
    print("sunpy not found, plotting will use basic matplotlib.")
    sunpy_available = False
import matplotlib.pyplot as plt
import astropy.units as u
import os

# --- Input Files (Dummy SDO AIA and HMI) ---
aia_file = 'aia_composite_base.fits'
hmi_file = 'hmi_composite_contour.fits'

# Create dummy files if they don't exist (with simplified, aligned WCS)
if not os.path.exists(aia_file) or not os.path.exists(hmi_file):
    print("Creating dummy AIA and HMI files...")
    im_size = (100, 100)
    # Common WCS (Helioprojective)
    w = WCS(naxis=2)
    w.wcs.ctype = ['HPLN-TAN', 'HPLT-TAN']
    w.wcs.crpix = [im_size[1]/2.0 + 0.5, im_size[0]/2.0 + 0.5]
    w.wcs.crval = [100.0, -200.0] # Example arcsec offset
    w.wcs.cdelt = np.array([-0.6, 0.6]) * u.arcsec/u.pix
    w.wcs.dateobs = '2023-01-10T10:00:00'
    # Add minimal observer info if needed by sunpy.map
    from astropy.coordinates import SkyCoord
    # w.wcs.observer_coord = SkyCoord(0*u.AU, 0*u.AU, 1*u.AU, frame='heliocentrictrueecliptic', obstime=w.wcs.dateobs)
    header_base = w.to_header()
    # AIA image data (loops)
    yy, xx = np.indices(im_size)
    aia_data = 500 * np.exp(-0.5 * (((xx - 50)/15)**2 + ((yy - 60)/30)**2)**0.5)
    aia_data += np.random.normal(100, 20, size=im_size)
    hdr_aia = header_base.copy()
    hdr_aia['INSTRUME'] = 'AIA'
    hdr_aia['WAVELNTH'] = 171
    fits.PrimaryHDU(aia_data.astype(np.float32), header=hdr_aia).writeto(aia_file, overwrite=True)
    # HMI magnetogram data (bipolar region)
    hmi_data = 500 * np.exp(-0.5 * (((xx - 40)/8)**2 + ((yy - 55)/8)**2))
    hmi_data -= 400 * np.exp(-0.5 * (((xx - 60)/8)**2 + ((yy - 65)/8)**2))
    hmi_data += np.random.normal(0, 30, size=im_size)
    hdr_hmi = header_base.copy()
    hdr_hmi['INSTRUME'] = 'HMI'
    hdr_hmi['CONTENT'] = 'MAGNETOGRAM'
    fits.PrimaryHDU(hmi_data.astype(np.float32), header=hdr_hmi).writeto(hmi_file, overwrite=True)


# --- Load Data ---
print("Loading AIA and HMI data...")
try:
    # Load AIA image using sunpy.map for coordinate-aware plotting
    if sunpy_available:
        aia_map = sunpy.map.Map(aia_file)
        print("Loaded AIA data using sunpy.map.")
    else: # Fallback to basic FITS read
        with fits.open(aia_file) as hdul:
             aia_data = hdul[0].data
             aia_wcs = WCS(hdul[0].header)
             print("Loaded AIA data using astropy.io.fits.")

    # Load HMI data (just need data and header/WCS for contours)
    with fits.open(hmi_file) as hdul:
        hmi_data = hdul[0].data
        hmi_wcs = WCS(hdul[0].header)
        print("Loaded HMI data using astropy.io.fits.")

    # --- Alignment Check/Reprojection (Conceptual) ---
    # In reality, solar data often requires specialized alignment using sunpy
    # coordinates transformations beyond simple WCS reprojection if times differ
    # or coordinate systems need careful handling (e.g., rotation).
    # For this example, we *assume* the dummy WCS are already aligned.
    # If reprojection were needed (and standard WCS sufficient):
    # if reproject_available and not wcs_are_aligned(aia_wcs, hmi_wcs):
    #    print("Reprojecting HMI to AIA frame (Conceptual)...")
    #    hmi_data_aligned, _ = reproject_interp((hmi_data, hmi_wcs), aia_wcs, shape_out=aia_map.data.shape)
    # else:
    hmi_data_aligned = hmi_data # Assume aligned
    print("Assuming images are aligned (no reprojection performed).")

except FileNotFoundError as e:
    print(f"Error: Input file not found - {e}")
    exit()
except ImportError as e:
    print(f"Error: Missing required library (sunpy or reproject?) - {e}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# --- Create Composite Plot ---
print("Creating composite plot...")
try:
    fig = plt.figure(figsize=(8, 8))
    # Use sunpy map's plot method if available for coordinate grid
    if sunpy_available:
        ax = fig.add_subplot(111, projection=aia_map)
        aia_map.plot(axes=ax, cmap='sdoaia171', # Standard AIA colormap
                     norm=plt.Normalize(vmin=np.percentile(aia_map.data, 1),
                                        vmax=np.percentile(aia_map.data, 99.9)))
        # Plot contours using ax.contour, providing data and WCS transform
        contour_levels = [-500, -200, 200, 500] # Example levels in Magnetogram units (e.g., Gauss)
        ax.contour(hmi_data_aligned, levels=contour_levels, colors=['blue', 'blue', 'red', 'red'],
                   linewidths=0.8, transform=ax.get_transform(hmi_wcs))
        ax.set_title("SDO/HMI Contours on AIA 171 Image")
    else: # Basic matplotlib plot if sunpy unavailable
        ax = fig.add_subplot(111, projection=aia_wcs) # Use WCSAxes
        norm = plt.Normalize(vmin=np.percentile(aia_data, 1), vmax=np.percentile(aia_data, 99.9))
        ax.imshow(aia_data, origin='lower', cmap='sdoaia171', norm=norm)
        contour_levels = [-500, -200, 200, 500]
        # Need to ensure hmi_data_aligned is used if reprojection occurred
        ax.contour(hmi_data_aligned, levels=contour_levels, colors=['blue', 'blue', 'red', 'red'],
                   linewidths=0.8) # WCS projection handles alignment
        ax.set_xlabel("HPLN (arcsec)")
        ax.set_ylabel("HPLT (arcsec)")
        ax.set_title("SDO/HMI Contours on AIA 171 Image (Matplotlib)")
        ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)


    plt.show()

except Exception as e:
    print(f"An error occurred during plotting: {e}")

```

This Python script demonstrates the creation of a composite solar image, overlaying magnetic field contours from SDO/HMI onto an SDO/AIA EUV image to visualize the relationship between magnetic structures and coronal features. It begins by loading both the AIA and HMI data, preferably using `sunpy.map.Map` for AIA to leverage its coordinate-aware plotting capabilities. While this example assumes the input dummy data are already aligned via their WCS for simplicity, it notes that in a real scenario, precise alignment using solar-specific coordinate transformations (often handled by `sunpy` functions beyond simple WCS reprojection) would be crucial, especially if observation times differ. The core visualization uses `matplotlib` (either directly through `sunpy.map.plot` or `WCSAxes`). The AIA image is displayed as the base map using an appropriate colormap. The `ax.contour` function is then used to draw contour lines from the HMI magnetogram data directly onto the AIA map axes. The `transform=ax.get_transform(hmi_wcs)` argument (used implicitly if plotting directly on `WCSAxes` with aligned data) ensures the contours, defined by the HMI data array, are correctly placed according to the shared heliographic coordinate system defined by the WCS, resulting in a scientifically meaningful overlay showing magnetic field concentrations relative to coronal structures.

**9.6.2 Planetary: Mosaicking Mars Orbiter Images (`reproject`)**
Creating large-scale maps of planetary surfaces, like Mars, often requires mosaicking numerous individual images taken by orbiters, such as the Context Camera (CTX) or High Resolution Imaging Science Experiment (HiRISE) on the Mars Reconnaissance Orbiter (MRO). These images need to be accurately georeferenced (possess WCS defining map projection coordinates) and then reprojected onto a common map grid before being combined. This example demonstrates using the `reproject` library to reproject two simulated, partially overlapping Mars image strips onto a common target map projection frame (e.g., Sinusoidal or Equirectangular), representing a key step in the mosaicking pipeline. It focuses on the reprojection aspect, assuming input WCS and a target frame definition are available.

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
# Requires reproject: pip install reproject
try:
    from reproject import reproject_interp
    from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd # Higher level mosaic tools
    reproject_available = True
except ImportError:
    print("reproject not found, skipping Mars mosaicking example.")
    reproject_available = False
import matplotlib.pyplot as plt
import os

# --- Input Data (Simulated Mars Image Strips with Map Projection WCS) ---
mars_strip1_file = 'mars_strip1.fits'
mars_strip2_file = 'mars_strip2.fits'
output_repr1_file = 'mars_strip1_reprojected.fits'
output_repr2_file = 'mars_strip2_reprojected.fits'

# Create dummy files if needed
if reproject_available:
    # Create Strip 1
    if not os.path.exists(mars_strip1_file):
        print(f"Creating dummy file: {mars_strip1_file}")
        im_size = (200, 50) # Long, narrow strip
        data1 = np.random.normal(1000, 50, size=im_size)
        data1[50:150, 10:40] += 500 # Add a feature
        w1 = WCS(naxis=2)
        w1.wcs.ctype = ['OLON-SIN', 'OLAT-SIN'] # Orthographic Longitude/Latitude, Sinusoidal proj
        w1.wcs.crpix = [im_size[1]/2.0 + 0.5, im_size[0]/2.0 + 0.5]
        w1.wcs.crval = [180.0, 0.0] # Lon/Lat Center 1 (deg)
        w1.wcs.cdelt = np.array([-0.001, 0.001]) # deg/pix scale
        hdu1 = fits.PrimaryHDU(data1.astype(np.float32), header=w1.to_header())
        hdu1.writeto(mars_strip1_file, overwrite=True)
    # Create Strip 2 (Overlapping/Adjacent)
    if not os.path.exists(mars_strip2_file):
        print(f"Creating dummy file: {mars_strip2_file}")
        im_size = (200, 50)
        data2 = np.random.normal(1100, 50, size=im_size) # Different background level
        data2[70:170, 5:35] += 600 # Different feature
        w2 = WCS(naxis=2)
        w2.wcs.ctype = ['OLON-SIN', 'OLAT-SIN']
        w2.wcs.crpix = [im_size[1]/2.0 + 0.5, im_size[0]/2.0 + 0.5]
        # Shift center slightly westward for overlap/adjacency
        w2.wcs.crval = [180.0 + (im_size[1]-10)*w1.wcs.cdelt[0], 0.0] # Shift Lon based on width & overlap
        w2.wcs.cdelt = np.array([-0.001, 0.001])
        hdu2 = fits.PrimaryHDU(data2.astype(np.float32), header=w2.to_header())
        hdu2.writeto(mars_strip2_file, overwrite=True)

    # --- Define Target Output Frame for Mosaic ---
    # Use find_optimal_celestial_wcs to determine a WCS and shape covering both inputs
    print("Determining optimal output mosaic frame...")
    # Need to provide paths or HDULists to the function
    # Assumes celestial (RA/Dec-like) WCS, might need adapter for planetary OLON/OLAT if func strict
    # For simplicity, manually define target frame covering both roughly
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.ctype = ['OLON-SIN', 'OLAT-SIN']
    # Calculate center and extent needed
    center_lon = (w1.wcs.crval[0] + w2.wcs.crval[0]) / 2.0
    center_lat = (w1.wcs.crval[1] + w2.wcs.crval[1]) / 2.0
    width_deg = (im_size[1] * abs(w1.wcs.cdelt[0])) * 1.8 # Approx width covering both + overlap
    height_deg = im_size[0] * abs(w1.wcs.cdelt[1]) * 1.1 # Approx height covering both
    target_shape = (int(height_deg / abs(w1.wcs.cdelt[1])), int(width_deg / abs(w1.wcs.cdelt[0])))
    target_wcs.wcs.crval = [center_lon, center_lat]
    target_wcs.wcs.cdelt = w1.wcs.cdelt
    target_wcs.wcs.crpix = [target_shape[1]/2.0 + 0.5, target_shape[0]/2.0 + 0.5]
    target_header = target_wcs.to_header()
    print(f"  Target Frame: Shape={target_shape}, Center=({center_lon:.3f}, {center_lat:.3f}), Scale={target_wcs.wcs.cdelt}")


    # --- Reproject Individual Strips ---
    input_data_list = [mars_strip1_file, mars_strip2_file]
    reprojected_arrays = []
    print("\nReprojecting input strips to target frame...")
    for i, infile in enumerate(input_data_list):
        try:
            print(f"  Reprojecting {infile}...")
            # Use reproject_interp (or reproject_exact for flux conservation)
            # Provide input data+header tuple, and target header + shape
            array_repr, footprint = reproject_interp((infile, 0), target_header, # Use HDU index 0
                                                      shape_out=target_shape, return_footprint=True)
            reprojected_arrays.append(array_repr)
            # Save reprojected strip (optional)
            # hdu_repr = fits.PrimaryHDU(data=array_repr.astype(np.float32), header=target_header)
            # hdu_repr.writeto(f'mars_strip{i+1}_reprojected.fits', overwrite=True)
        except FileNotFoundError:
             print(f"  Error: Input file {infile} not found.")
        except Exception as reproj_err:
             print(f"  Error during reprojection of {infile}: {reproj_err}")

    # --- Conceptual Mosaic Combination (using reproject helper) ---
    # The reproject package also has higher-level mosaicking functions
    # that can handle reprojection AND coaddition, including background matching.
    # Example using reproject_and_coadd (requires more setup for background matching etc.)
    if len(reprojected_arrays) == len(input_data_list):
         print("\nConceptual combination using reproject_and_coadd (simplified)...")
         # This function needs input HDUs/paths, target WCS/header, etc.
         # It can perform weighted averaging based on footprints or specified weights.
         # Background subtraction/matching is a separate concern usually handled before/during.
         # For a simple average visualization (ignoring weights/backgrounds):
         # Stack valid pixels and average
         valid_pixels_mask = np.isfinite(reprojected_arrays[0]) & np.isfinite(reprojected_arrays[1])
         combined_image_simple_avg = np.nanmean(np.stack(reprojected_arrays), axis=0)

         # --- Optional: Visualization ---
         fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, subplot_kw={'projection': target_wcs})
         norm = plt.Normalize(vmin=800, vmax=1800) # Adjust vmin/vmax
         im0 = axes[0].imshow(reprojected_arrays[0], origin='lower', cmap='gray', norm=norm)
         axes[0].set_title('Reprojected Strip 1')
         axes[0].coords.grid(True, color='white', ls='dotted', alpha=0.5)
         im1 = axes[1].imshow(reprojected_arrays[1], origin='lower', cmap='gray', norm=norm)
         axes[1].set_title('Reprojected Strip 2')
         axes[1].coords.grid(True, color='white', ls='dotted', alpha=0.5)
         im2 = axes[2].imshow(combined_image_simple_avg, origin='lower', cmap='gray', norm=norm)
         axes[2].set_title('Simple Average Mosaic (Conceptual)')
         axes[2].coords.grid(True, color='white', ls='dotted', alpha=0.5)
         plt.tight_layout()
         plt.show()

    else:
         print("Skipping combination and visualization due to reprojection errors.")


else:
     print("Skipping Mars mosaicking example: reproject unavailable or dummy data missing.")

```

This Python script illustrates the core reprojection step involved in creating a mosaic map of a planetary surface, using Mars orbiter images as an example and leveraging the `reproject` library. It simulates two overlapping image "strips," each with its own WCS defining its location and orientation within a specific map projection (e.g., Sinusoidal). A target output frame is defined, specifying the WCS parameters (projection, center, scale) and pixel dimensions intended for the final, larger mosaic covering the area of both strips. The key operation uses `reproject.reproject_interp` (or potentially a flux-conserving alternative like `reproject_exact`) to resample each input image strip onto this common target grid. The function takes the input data and WCS (read from the FITS header) and the target output header (or WCS object and shape) and computes the corresponding pixel values in the output grid via interpolation. The resulting arrays (`reprojected_arrays`) represent the individual image strips now aligned on the same map projection grid, ready for subsequent steps like background matching and weighted co-addition (using tools like `reproject.mosaicking.reproject_and_coadd` or external packages like Montage) to produce the final seamless mosaic. The visualization shows the two reprojected strips and a conceptual simple average combination.

**9.6.3 Stellar: Stacking Deep Exposures of Stellar Stream**
Detecting faint, diffuse structures like stellar streams in the Galactic halo requires very deep imaging, often achieved by stacking multiple long exposures of the same field. This process enhances the signal-to-noise ratio sufficiently to reveal the low surface brightness stream against the background noise and foreground stars. This example demonstrates stacking several simulated deep exposures of a faint stellar stream after aligning them (assuming prior reduction and alignment). It uses `ccdproc.Combiner` with median combination to effectively reject cosmic rays and other transient artifacts while summing the faint stream signal.

```python
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
# Requires ccdproc: pip install ccdproc
try:
    import ccdproc
    ccdproc_available = True
except ImportError:
    print("ccdproc not found, skipping stellar stream stacking example.")
    ccdproc_available = False
import astropy.units as u
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
import os

# --- Input Files (Simulated aligned, reduced images of a faint stream) ---
n_exp = 8 # Number of exposures to stack
input_files = [f'stream_exp_{i+1:02d}_aligned.fits' for i in range(n_exp)]
output_stacked_file = 'stream_stacked_image.fits'

# Create dummy files if they don't exist
if ccdproc_available:
    # Check/create dummy aligned files
    all_files_exist = True
    if not all(os.path.exists(f) for f in input_files):
         print("Creating dummy deep exposure files...")
         im_size = (150, 150)
         # Simulate background + faint stream + stars + CRs
         background_level = 200.0
         noise_sigma = 10.0 # Higher noise per frame
         # Faint stream (e.g., a diagonal band)
         yy, xx = np.indices(im_size)
         stream_signal = 15.0 * np.exp(-0.5 * ((yy - xx - 10) / 15.0)**2) # Diagonal gaussian band
         # Add some foreground stars
         n_stars = 20
         x_stars = np.random.uniform(0, im_size[1], n_stars)
         y_stars = np.random.uniform(0, im_size[0], n_stars)
         flux_stars = 10**(np.random.uniform(2.5, 4.5, n_stars))
         psf_sigma = 1.8
         star_field = np.zeros_like(stream_signal)
         for x, y, flux in zip(x_stars, y_stars, flux_stars):
              dist_sq = (xx - x)**2 + (yy - y)**2
              star_field += flux * np.exp(-dist_sq / (2 * psf_sigma**2))
         # Create individual frames
         for i, fname in enumerate(input_files):
              data = np.random.normal(background_level, noise_sigma, size=im_size) + stream_signal + star_field
              # Add random cosmic rays to each frame
              n_crs = 5
              cr_x = np.random.randint(0, im_size[1], n_crs)
              cr_y = np.random.randint(0, im_size[0], n_crs)
              cr_flux = np.random.uniform(500, 2000, n_crs)
              data[cr_y, cr_x] += cr_flux
              # Write file
              hdu = fits.PrimaryHDU(data.astype(np.float32))
              hdu.header['BUNIT'] = 'electron'
              hdu.writeto(fname, overwrite=True)
         all_files_exist = True # Mark as created


if ccdproc_available and all_files_exist:
    try:
        # --- Load Aligned Deep Exposures ---
        print(f"Loading {n_exp} aligned deep exposures...")
        ccd_list = []
        for f in input_files:
            try:
                # Read as CCDData, assuming units are electrons
                ccd = CCDData.read(f, unit='electron')
                # If uncertainty available (e.g., from earlier processing), load it too
                # ccd = CCDData.read(f, unit='electron', hdu_uncertainty='ERR')
                ccd_list.append(ccd)
            except Exception as read_err:
                print(f"Warning: Error reading file {f}: {read_err}. Skipping.")

        if not ccd_list:
            raise ValueError("No valid images loaded for stacking.")
        if len(ccd_list) < 5: # Median combination needs sufficient frames for CR rejection
             print("Warning: Stacking fewer than ~5 images may provide poor cosmic ray rejection with median.")

        # --- Combine Images using Median ---
        print(f"Combining {len(ccd_list)} images using median stack...")
        # Initialize Combiner
        combiner = ccdproc.Combiner(ccd_list)

        # Use median combine for robust outlier/cosmic ray rejection
        # Sigma clipping could be added but median is often sufficient for CRs with enough frames
        # combiner.sigma_clipping(low_thresh=3, high_thresh=3) # Optional
        stacked_stream_image = combiner.median_combine()

        # Update metadata
        stacked_stream_image.meta['NCOMBINE'] = len(ccd_list)
        stacked_stream_image.meta['COMBTYPE'] = 'Median'
        stacked_stream_image.meta['HISTORY'] = f'Median stacked {len(ccd_list)} aligned deep exposures.'

        # --- Save Stacked Image ---
        print("Saving stacked image...")
        # stacked_stream_image.write(output_stacked_file, overwrite=True)
        print(f"(If successful, stacked image would be saved to {output_stacked_file})")

        # --- Optional: Visualization ---
        plt.figure(figsize=(8, 8))
        # Use robust normalization to see faint stream
        norm = plt.Normalize(vmin=np.percentile(stacked_stream_image.data, 5),
                             vmax=np.percentile(stacked_stream_image.data, 98))
        plt.imshow(stacked_stream_image.data, origin='lower', cmap='gray', norm=norm)
        plt.title(f"Median Stacked Image ({len(ccd_list)} Exposures) - Stellar Stream")
        plt.colorbar(label=f"Flux ({stacked_stream_image.unit})")
        plt.show()


    except ImportError:
        print("Error: ccdproc library is required but not found.")
    except ValueError as e:
         print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during stream image stacking: {e}")
else:
     print("Skipping stellar stream stacking example: ccdproc unavailable or input files missing.")
```

This Python script demonstrates the stacking of multiple deep exposures to reveal faint structures like stellar streams, using `ccdproc`. It simulates several individual image frames of a region containing a faint stream, foreground stars, background noise, and crucially, different random cosmic ray hits in each frame. Assuming these frames have already been processed through basic reduction (bias, dark, flat) and accurately aligned spatially (a critical prerequisite not shown here), they are loaded as a list of `CCDData` objects. The `ccdproc.Combiner` class is then used to combine these images. The `median_combine` method is chosen specifically for its robustness in rejecting outlier pixel values; cosmic rays, which affect different pixels in each aligned frame, are effectively removed by the median calculation when combining a sufficient number of images (typically 5 or more). The resulting `stacked_stream_image` is a significantly deeper image where the random noise is reduced and cosmic rays are suppressed, allowing the faint stellar stream signal to become visible against the background. The visualization displays the final stacked image, highlighting the revealed stream.

**9.6.4 Exoplanetary: Visualizing TESS Pixel Data with Aperture**
Analyzing data from TESS Target Pixel Files (TPFs) sometimes requires visual inspection of the pixel data itself, for example, to verify the quality of the photometric aperture used by the pipeline, check for contamination from nearby stars, or understand instrumental effects. The `lightkurve` package provides convenient tools for this. This example uses `lightkurve` to download a TPF for a TESS target, extracts the pixel data cube, defines or retrieves a photometric aperture mask, and visualizes a single frame (cadence) of the pixel data with the chosen aperture overlaid. This allows assessment of which pixels are included in the photometry.

```python
import numpy as np
# Requires lightkurve: pip install lightkurve
try:
    import lightkurve as lk
    lightkurve_available = True
except ImportError:
    print("lightkurve not found, skipping TESS pixel visualization example.")
    lightkurve_available = False
import matplotlib.pyplot as plt

# --- Target: A TESS Object of Interest (e.g., planet host) ---
# Example: Pi Mensae (TIC 261136679), hosts transiting planet Pi Men c
target_tic = 'TIC 261136679'
sector_num = 1 # Specify sector (or search all)

if lightkurve_available:
    try:
        # --- Search and Download TESS Target Pixel File (TPF) ---
        print(f"Searching for TESS TPF for {target_tic}, Sector {sector_num}...")
        # Search for TPF files from the TESS-SPOC pipeline
        search_result = lk.search_targetpixelfile(target_tic, sector=sector_num, author='SPOC')
        if len(search_result) == 0:
            raise ValueError(f"No TPF found for {target_tic} in Sector {sector_num}.")

        print(f"Found {len(search_result)} TPF product(s). Downloading the first one...")
        tpf = search_result.download()
        # Alternatively, download specific cadence if needed: download(quality_bitmask='default', download_dir='.')

        print("TPF downloaded successfully.")
        print(tpf) # Print basic info about the TPF object

        # --- Visualize Pixel Data and Aperture ---
        # Select a specific frame (cadence) to visualize, e.g., the first frame
        frame_index = 0
        print(f"\nVisualizing frame {frame_index} with pipeline aperture...")

        # Use the TPF object's plot method for convenient visualization
        # It automatically handles pixel coordinates, flux scaling, and aperture overlay
        # By default, it plots the pipeline aperture mask stored within the TPF
        ax = tpf.plot(frame=frame_index, aperture_mask='pipeline', show_colorbar=True)
        ax.set_title(f"TESS Pixel Data (Frame {frame_index}) with Pipeline Aperture")
        plt.show()

        # --- Example: Define and Plot a Custom Aperture ---
        print("\nVisualizing frame with a custom circular aperture...")
        # Create a custom aperture mask, e.g., a simple circular aperture
        # Need target position within the cutout (often near center)
        cutout_shape = tpf.shape[1:] # Get (y, x) shape of cutout
        center_x = cutout_shape[1] / 2.0 - 0.5
        center_y = cutout_shape[0] / 2.0 - 0.5
        custom_aperture_radius = 2.0 # pixels
        # Create a boolean mask for the aperture
        custom_mask = tpf.create_threshold_mask(threshold=0) # Start with empty mask
        yy, xx = np.indices(cutout_shape)
        custom_mask = (xx - center_x)**2 + (yy - center_y)**2 < custom_aperture_radius**2

        # Plot the frame again, overlaying the custom mask
        ax_custom = tpf.plot(frame=frame_index, aperture_mask=custom_mask, mask_color='red', show_colorbar=True)
        ax_custom.set_title(f"TESS Pixel Data (Frame {frame_index}) with Custom Aperture (r={custom_aperture_radius})")
        plt.show()

    except ValueError as e:
         print(f"Value Error: {e}")
    except Exception as e:
        # Catch errors during search, download, or plotting
        print(f"An unexpected error occurred in the TESS pixel visualization example: {e}")
else:
    print("Skipping TESS pixel visualization example: lightkurve unavailable.")

```

This Python script utilizes the `lightkurve` library to facilitate the visualization of data stored within a TESS Target Pixel File (TPF). It begins by searching for and downloading a TPF for a specified target star and observing sector using `lk.search_targetpixelfile` and `search_result.download()`. The downloaded `tpf` object encapsulates the time series of pixel data (flux cube), time stamps, and associated metadata, including the standard photometric aperture mask used by the TESS pipeline. The script then leverages the `tpf.plot()` method, a convenient built-in function, to display a single frame (specified by `frame_index`) from the pixel data cube. Crucially, setting `aperture_mask='pipeline'` automatically overlays the standard pipeline aperture onto the image, allowing visual inspection of which pixels were included in the default photometry. The script further demonstrates how to define a custom aperture mask (e.g., a simple circular mask created using NumPy array logic) and use `tpf.plot()` again with the custom `aperture_mask` specified to visualize this alternative aperture choice overlaid on the pixel data.

**9.6.5 Galactic: Three-Color Image Creation of Orion Nebula (`astropy.visualization`)**
Creating visually appealing and informative color composite images is a common way to represent multi-band observations of striking Galactic objects like the Orion Nebula. By assigning images taken through different filters (e.g., tracing different emission lines or continuum ranges) to the Red, Green, and Blue color channels, the resulting image highlights the spatial distribution of different physical components. This example demonstrates creating an RGB composite image from three simulated input FITS files (representing R, G, B channels, which could correspond to H-alpha, [O III], and continuum filters, for example), using the `astropy.visualization.make_lupton_rgb` function, which implements a sophisticated algorithm for optimal color balancing and scaling based on asinh stretches.

```python
import numpy as np
from astropy.io import fits
# Requires astropy.visualization for RGB creation
try:
    from astropy.visualization import make_lupton_rgb, AsinhStretch, ManualInterval
    astropy_vis_available = True
except ImportError:
    print("astropy.visualization not found, skipping RGB image example.")
    astropy_vis_available = False
import matplotlib.pyplot as plt
import os

# --- Input Data (Simulated R, G, B images of Orion region) ---
# Assume files 'orion_r.fits', 'orion_g.fits', 'orion_b.fits' exist and are aligned
r_file = 'orion_r.fits'
g_file = 'orion_g.fits'
b_file = 'orion_b.fits'

# Create dummy files if they don't exist
if astropy_vis_available:
    all_files_exist = True
    for fname, channel_name in zip([r_file, g_file, b_file], ['R', 'G', 'B']):
        if not os.path.exists(fname):
            print(f"Creating dummy file: {fname}")
            im_size = (200, 200)
            yy, xx = np.indices(im_size)
            # Simulate base structure (e.g., diffuse cloud)
            base_signal = 50 * np.exp(-0.5 * (((xx - 100)/60)**2 + ((yy - 100)/80)**2))
            # Add channel-specific features
            if channel_name == 'R': # e.g., H-alpha dominant features
                feature = 150 * np.exp(-0.5 * (((xx - 80)/15)**2 + ((yy - 120)/20)**2))
            elif channel_name == 'G': # e.g., [O III] dominant features
                feature = 180 * np.exp(-0.5 * (((xx - 120)/25)**2 + ((yy - 90)/18)**2))
            else: # B channel (e.g., continuum or other lines)
                feature = 60 * np.exp(-0.5 * (((xx - 100)/50)**2 + ((yy - 70)/40)**2))
            data = base_signal + feature + np.random.normal(5, 2, size=im_size)
            data = np.maximum(data, 0)
            hdu = fits.PrimaryHDU(data.astype(np.float32))
            hdu.writeto(fname, overwrite=True)
            if not os.path.exists(fname): all_files_exist = False


if astropy_vis_available and all_files_exist:
    try:
        # --- Load R, G, B Images ---
        print("Loading R, G, B image files...")
        try:
            image_r = fits.getdata(r_file)
            image_g = fits.getdata(g_file)
            image_b = fits.getdata(b_file)
        except FileNotFoundError as e:
            print(f"Error: Input file not found - {e}")
            exit()

        # --- Create RGB Image using make_lupton_rgb ---
        print("Creating RGB composite image using make_lupton_rgb...")
        # Parameters for make_lupton_rgb control the scaling and stretching:
        # minimum: Background level subtracted before scaling (scalar or per channel)
        # stretch: Controls the nonlinearity of the stretch (higher values = more linear)
        # Q: Controls the softening factor for the asinh stretch
        # These often require experimentation to get a visually pleasing result.

        # Estimate minimum background level (use a robust percentile)
        bg_level = np.percentile(np.concatenate((image_r.flatten(), image_g.flatten(), image_b.flatten())), 5)
        print(f"  Estimated minimum background level: {bg_level:.1f}")
        # Define stretch and Q parameters (typical starting points)
        stretch_val = 0.5 # Adjust for contrast
        Q_val = 8       # Adjust for asinh softening

        # Generate the RGB image array
        rgb_image = make_lupton_rgb(image_r, image_g, image_b,
                                    minimum=bg_level,
                                    stretch=stretch_val,
                                    Q=Q_val)
        print("RGB image array created.")

        # --- Display the RGB Image ---
        print("Displaying the RGB image...")
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_image, origin='lower')
        plt.title("Orion Nebula RGB Composite (Simulated)")
        plt.xlabel("X pixels")
        plt.ylabel("Y pixels")
        # Turn off axis ticks for cleaner image display if desired
        # plt.xticks([])
        # plt.yticks([])
        plt.show()

    except ImportError:
        print("Error: astropy.visualization is required but not found.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred during RGB image creation: {e}")
else:
     print("Skipping RGB image example: astropy.visualization unavailable or input files missing.")

```

This Python script demonstrates the creation of a three-color composite image, often used to visualize multi-band observations of Galactic nebulae like Orion, using `astropy.visualization`. It loads three separate FITS images, assumed to be spatially aligned, representing the data acquired in the Red, Green, and Blue channels respectively (these could correspond to specific filters like H-alpha, [O III], continuum, or standard broadband filters). The core function `astropy.visualization.make_lupton_rgb` is then employed to combine these three single-channel images into a single RGB image array suitable for display. This function implements the Lupton et al. (2004) algorithm, which uses an arcsinh stretch (`AsinhStretch`) to handle the large dynamic range typical of astronomical images effectively. Key parameters like `minimum` (background level), `stretch` (controlling contrast), and `Q` (controlling the asinh softening) are set to produce a visually informative image where colors represent the relative intensity in the different input bands, highlighting features with different emission characteristics. The resulting `rgb_image` array is then displayed using `matplotlib.pyplot.imshow`.

**9.6.7 Cosmology: Visualization of Simulation Slice (`yt`)**
Large-scale cosmological simulations produce massive datasets representing the distribution of dark matter, gas, and galaxies within a 3D volume evolving over time. Visualizing this complex, multi-field, 3D data is essential for understanding the formation of cosmic structures like the cosmic web, galaxy clusters, and filaments. The `yt` toolkit is specifically designed for analyzing and visualizing such volumetric astrophysical simulation data. This example provides a conceptual outline of how `yt` might be used to load a cosmological simulation dataset (often stored in specialized formats like Enzo HDF5, Gadget snapshots, etc.) and create a 2D slice plot showing the density or temperature distribution within a specific plane through the simulation box.

```python
# Conceptual Example: Visualizing a slice through a cosmological simulation cube using yt
# Requires yt: pip install yt
# Requires loading actual simulation data (not generated here)

try:
    import yt
    yt_available = True
except ImportError:
    print("yt not found, skipping Cosmology simulation visualization example.")
    yt_available = False
import matplotlib.pyplot as plt # yt uses matplotlib for output

# --- Input: Path to Simulation Dataset ---
# Replace with the actual path to your simulation output file/directory
# Supported formats include Enzo, Gadget, RAMSES, ART, Cholla, etc.
# Example using a generic dataset name
simulation_dataset_path = "simulation_output/GasSloshing/sloshing_low_res_hdf5_plt_cnt_0300" # Example path structure
# Need actual data for this to run. Example files available at yt-project.org

if yt_available:
    # Check if the dummy path exists (as a stand-in for real data check)
    # In a real scenario, check accessibility of the actual dataset
    # For this example, we'll just proceed conceptually if yt is installed.
    data_exists_conceptually = True # Assume data is present for concept demo
    if not data_exists_conceptually: # Replace with actual os.path.exists(simulation_dataset_path) check
        print(f"Error: Simulation dataset not found at {simulation_dataset_path}.")
        print("Download sample data from http://yt-project.org/data/ to run this example.")
        # exit() # Exit if data truly missing

    print("Conceptual Example: Visualizing Simulation Slice with yt")
    print(f"Attempting to load dataset (if path were real): {simulation_dataset_path}")

    try:
        # --- Load Simulation Dataset using yt ---
        # yt automatically detects the simulation format and loads the data hierarchy
        # This line would load the actual data if the path was valid
        # ds = yt.load(simulation_dataset_path) # This is the core loading step

        # For demonstration without real data, create a dummy ds object with basic attributes
        # In a real run, remove this dummy object creation
        class DummyDS: # Minimal dummy object to allow code structure to run
            domain_center = [0.5, 0.5, 0.5] * yt.units.code_length
            domain_width = [1.0, 1.0, 1.0] * yt.units.code_length
            domain_left_edge = domain_center - domain_width / 2
            domain_right_edge = domain_center + domain_width / 2
            current_time = 0.0 * yt.units.Gyr
            def __repr__(self): return f"DummyDataset (Time: {self.current_time})"
        ds = DummyDS() # Use dummy object if data loading is skipped/fails

        print(f"Dataset loaded conceptually: {ds}")
        # Print basic dataset info (would work with real ds)
        # print(ds.domain_width)
        # print(ds.domain_center)
        # print(ds.current_time)

        # --- Create a Slice Plot ---
        # yt provides convenient objects for generating visualizations
        # Create a SlicePlot object: ds, axis, field
        # 'axis': The axis normal to the slice plane ('x', 'y', or 'z')
        # 'field': The physical field to plot (e.g., ('gas', 'density'), ('gas', 'temperature'))
        print("Creating a slice plot of gas density through the center...")
        # This line creates the plot object using the loaded dataset (ds)
        # slc = yt.SlicePlot(ds, 'z', ('gas', 'density')) # Example: Slice through z-axis showing density

        # --- Customize the Plot (using SlicePlot methods) ---
        # Annotate grid, velocity vectors, contours, etc.
        # slc.annotate_grids() # Show refinement grids if AMR simulation
        # slc.annotate_title(f"Gas Density at z=0.5 (Time = {ds.current_time:.2f})")
        # Set colormap, colorbar limits (zlim), log scale
        # slc.set_cmap(field=('gas', 'density'), cmap='viridis')
        # slc.set_zlim(field=('gas', 'density'), zmin=1e-30, zmax=1e-26) # Example density range
        # slc.set_log(field=('gas', 'density'), log=True)

        print("Conceptual plot object created and customized.")
        print("  (Plotting would typically show gas density sliced along the z-axis)")

        # --- Save or Display the Plot ---
        # Save the plot to a file
        # output_plot_file = 'simulation_slice_density.png'
        # slc.save(output_plot_file)
        # print(f"Slice plot saved conceptually to {output_plot_file}")
        # Or display interactively (might require specific backends)
        # slc.show()

        # Example of plotting Temperature instead
        # print("\nCreating a slice plot of gas temperature...")
        # slc_temp = yt.SlicePlot(ds, 'z', ('gas', 'temperature'))
        # slc_temp.set_cmap(field=('gas', 'temperature'), cmap='magma')
        # slc_temp.set_log(field=('gas', 'temperature'), log=True)
        # slc_temp.save('simulation_slice_temperature.png')
        print("\n(Similar process could create temperature slice plot)")

    except ImportError:
        print("Error: yt library is required but not found.")
    except Exception as e:
        print(f"An unexpected error occurred during yt visualization: {e}")
else:
     print("Skipping Cosmology simulation visualization example: yt unavailable.")

```

This Python script provides a conceptual illustration of how the `yt` toolkit is used to visualize data from large-scale cosmological simulations. It begins by defining the path to a simulation dataset (noting that actual data is required for execution). The core step involves loading this dataset using `yt.load()`, which automatically parses the simulation format and creates a dataset object (`ds`) containing the data hierarchy and metadata. To create a visualization, the script demonstrates the instantiation of a `yt.SlicePlot` object. This requires specifying the dataset (`ds`), the axis perpendicular to the desired slice (e.g., `'z'`), and the physical field to be plotted (e.g., `('gas', 'density')` or `('gas', 'temperature')`). The `SlicePlot` object provides numerous methods for customization, such as setting colormaps (`set_cmap`), colorbar limits (`set_zlim`), applying logarithmic scaling (`set_log`), adding titles (`annotate_title`), or overlaying simulation grid structures (`annotate_grids`). Finally, the plot can be saved to a file using `.save()` or displayed interactively using `.show()`, providing a 2D view through the complex 3D simulation volume. (Note: This example uses a dummy dataset object as real simulation data is needed to run `yt.load` and generate actual plots.)

---

**References**

Allen, A., Teuben, P., Paddy, K., Greenfield, P., Droettboom, M., Conseil, S., Ninan, J. P., Tollerud, E., Norman, H., Deil, C., Bray, E., Sipőcz, B., Robitaille, T., Kulumani, S., Barentsen, G., Craig, M., Pascual, S., Perren, G., Lian Lim, P., … Streicher, O. (2022). Astropy: A community Python package for astronomy. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.6514771
*   *Summary:* This Zenodo record archives a version of the Astropy package. Its sub-packages `astropy.wcs` (Section 9.1), `astropy.visualization` (Sections 9.4, 9.5.1, 9.6.5), and affiliated packages `reproject` (Section 9.1, 9.3) and `ccdproc` (Section 9.2) provide fundamental tools for alignment, visualization, reprojection, and image combination discussed in this chapter.

Annis, J., Soares-Santos, M., Strauss, M. A., Becker, M. R., Dodelson, S., Fan, X., Gunn, J. E., Hao, J., Ivezić, Ž., Joffre, P., Johnston, D. E., Kubik, D., Lahav, O., Lin, H., Lupton, R. H., McKay, T. A., Plazas, A. A., Roe, N. A., Sheldon, E., … Yanny, B. (2014). The Sloan Digital Sky Survey Coadd: 275 deg$^2$ of deep Sloan Digital Sky Survey imaging. *The Astrophysical Journal, 794*(2), 120. https://doi.org/10.1088/0004-637X/794/2/120 *(Note: Pre-2020, but details large-scale stacking)*
*   *Summary:* Although pre-2020, this paper describes the creation of the SDSS "Stripe 82" coadd, a widely used deep dataset created by stacking multiple SDSS scans. It details practical challenges and techniques involved in large-scale image stacking (Section 9.2).

Astropy Collaboration. (2019). Astropy/reproject: 0.6. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.3355866 *(Note: Software release reference, pre-2020)*
*   *Summary:* This Zenodo record archives a specific version (0.6) of the `reproject` package. `reproject` is the key Astropy-affiliated library for performing WCS-based image reprojection needed for alignment and mosaicking (Sections 9.1, 9.3).

Astropy Project. (n.d.). *montage-wrapper*. Retrieved from https://github.com/astropy/montage-wrapper *(Note: Software repository)*
*   *Summary:* This is the GitHub repository for `montage-wrapper`, a Python package providing an interface to the Montage mosaicking toolkit. It is relevant to the discussion of mosaic construction tools in Section 9.3.

Beaumont, C., Goodman, A., Greenfield, P., Lim, P. L., & Robitaille, T. (2015). Glue Visualization. *Astronomical Data Analysis Software and Systems XXIV*, 495, 387. *(Note: Conference proceeding reference, pre-2020)*
*   *Summary:* This conference proceeding describes the `glue-viz` application. While pre-2020, `glue` remains a powerful tool for linked-view interactive exploration of multi-dimensional data, including cubes (Section 9.5.3).

Berriman, G. B., Good, J. C., Laity, A. C., & Jacob, J. C. (2003). Montage: A Grid Portal for Sustained Processing of Terabyte-Scale Data Sets. *Astronomical Data Analysis Software and Systems (ADASS) XII*, 295, 453. *(Note: Foundational Montage paper, pre-2020)*
*   *Summary:* This early paper describes the Montage toolkit, a foundational software package specifically designed for astronomical image mosaicking (Section 9.3), addressing challenges like background matching and reprojection across large datasets.

Bokeh Development Team. (2023). *Bokeh: Python library for interactive visualization*. https://bokeh.org *(Note: Software website)*
*   *Summary:* The official website for the Bokeh library. Bokeh is mentioned in Section 9.5.4 as a powerful tool for creating interactive web-based visualizations in Python, suitable for data exploration.

BQplot Development Team. (2021). *bqplot: Plotting library for IPython/Jupyter notebooks*. https://github.com/bqplot/bqplot *(Note: Software repository)*
*   *Summary:* The GitHub repository for `bqplot`. This library is mentioned in Section 9.5.4 as a tool specifically designed for creating interactive plots directly within Jupyter notebooks, leveraging browser capabilities.

Christensen, A. J., Alden, K., & Rogers, K. K. (2022). Accessible Visualization for Astronomy: Raising the Floor on Scientific Data Visualization. *Bulletin of the AAS, 54*(2), e002. https://doi.org/10.3847/25c2cfeb.5d0a61e2
*   *Summary:* This article discusses the importance of creating accessible scientific visualizations, considering factors like color blindness. It provides modern context and best practices relevant to the principles of effective visualization discussed in Section 9.5.

Holwerda, B. W. (2021). Stacking astronomical images: methods and pitfalls. *Raspberry Pi for Computer Vision*, 319–343. [Chapter in a Book/Manual]. https://benneholwerda.files.wordpress.com/2021/09/stackingchapter_holwerda.pdf
*   *Summary:* This book chapter provides a practical guide to image stacking techniques in astronomy. It covers methods like mean and median combination (Section 9.2) and discusses common issues and best practices.

Jacob, J. C., Katz, D. S., Berriman, G. B., Good, J., Laity, A. C., Deelman, E., Kesselman, C., Singh, G., Su, M.-H., Prince, T. A., & Williams, R. D. (2010). Montage: A Petascale Image Montaging Service for the Era of the Virtual Observatory. *PLoS ONE, 5*(2), e9181. https://doi.org/10.1371/journal.pone.0009181 *(Note: Key Montage science paper, pre-2020)*
*   *Summary:* Although pre-2020, this paper details the science capabilities and architecture of the Montage mosaicking service. It provides essential background on the challenges and solutions involved in large-scale mosaic construction (Section 9.3).

Padovani, P., Giommi, P., & Resconi, E. (2023). Multi-messenger astrophysics: Status, challenges and opportunities. *Progress in Particle and Nuclear Physics, 131*, 104034. https://doi.org/10.1016/j.ppnp.2023.104034
*   *Summary:* This review discusses multi-messenger astrophysics, which intrinsically involves combining data from different sources. While broader than just imaging, it highlights the scientific driver for data fusion techniques discussed conceptually in Section 9.4.

Plotly Technologies Inc. (2015). *Collaborative data science*. Plotly Technologies Inc. https://plotly.com *(Note: Software website)*
*   *Summary:* The official website for Plotly. Plotly is mentioned in Section 9.5.4 as a popular library for creating rich, interactive scientific visualizations suitable for web embedding and dashboards.

Robitaille, T., & Bressert, E. (2012). APLpy: Astronomical Plotting Library in Python. *Astrophysics Source Code Library*, record ascl:1208.017. https://ui.adsabs.harvard.edu/abs/2012ascl.soft08017R/abstract *(Note: ASCL entry, pre-2020)*
*   *Summary:* The ASCL entry for APLpy. While pre-2020, APLpy remains a widely used Python library specifically designed for creating publication-quality plots of astronomical images and data cubes, including WCS projections and overlays (Section 9.5.3).

Robitaille, T. P., Tollerud, E. J., Greenfield, P., Droettboom, M., Bray, E., Aldcroft, T., Davis, M., Ginsburg, A., Price-Whelan, A. M., Kerzendorf, W. E., & Astropy Collaboration. (2017). Glue Visualization (Version v0.11) [Computer software]. https://doi.org/10.5281/ZENODO.834288 *(Note: Software release reference, pre-2020)*
*   *Summary:* This Zenodo record archives a version of `glue-viz`. While pre-2020, Glue is a powerful, interactive, linked-view visualization tool mentioned in Section 9.5.3 as being particularly useful for exploring relationships within multi-dimensional datasets like data cubes.

Spinoglio, L., Fernández-Ontiveros, J. A., Pereira-Santaella, M., Malkan, M. A., & Dasyra, K. M. (2021). Synergy between infrared spectroscopy and other multiwavelength observations of AGN. *Rendiconti Lincei. Scienze Fisiche e Naturali, 32*(1), 131–141. https://doi.org/10.1007/s12210-021-00983-0
*   *Summary:* Discusses the scientific value of combining infrared spectroscopy with other multi-wavelength data for AGN studies. This highlights the motivation for data fusion techniques (Section 9.4) in astrophysical research.

Turk, M. J., Smith, B. D., Oishi, J. S., Skory, S., Skillman, S. W., Abel, T., & Norman, M. L. (2011). yt: A Multi-code Analysis Toolkit for Astrophysical Simulation Data. *The Astrophysical Journal Supplement Series, 192*(1), 9. https://doi.org/10.1088/0067-0049/192/1/9 *(Note: Foundational yt paper, pre-2020)*
*   *Summary:* The foundational paper introducing the `yt` toolkit. Although pre-2020, `yt` remains a primary tool for analyzing and visualizing large volumetric astrophysical simulation data, including generating slices and projections as mentioned in Section 9.5.3 and demonstrated conceptually in Example 9.6.7.
