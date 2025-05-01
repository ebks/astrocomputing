---
# Appendix B
# Quick Reference for Key Libraries
---
![imagem](imagem.png)

*This appendix provides a concise quick reference (or "cheatsheet") for frequently used functions, modules, and concepts within the core Python libraries essential for astrocomputing, as discussed throughout this book. It is not exhaustive but aims to serve as a rapid reminder of common syntax and functionality for libraries like NumPy, SciPy, Matplotlib, Astropy (including its key sub-packages like `io.fits`, `table`, `units`, `coordinates`, `wcs`, `modeling`, `visualization`), Photutils, Specutils, Lightkurve, and Astroquery. Users are encouraged to consult the detailed official documentation for each library for comprehensive information, advanced features, and specific options.*

---

**B.1 NumPy (`numpy`)**

*Fundamental package for numerical array computations.*

*   **Import:** `import numpy as np`
*   **Array Creation:**
    *   `np.array([1, 2, 3])`: Create from list/tuple.
    *   `np.zeros((3, 4))`: Array of zeros with shape (3, 4).
    *   `np.ones((3, 4))`: Array of ones.
    *   `np.arange(start, stop, step)`: Evenly spaced values within interval.
    *   `np.linspace(start, stop, num)`: `num` evenly spaced values over interval.
    *   `np.logspace(start, stop, num)`: `num` log-spaced values.
    *   `np.random.rand(d0, d1, ...)`: Random values [0, 1).
    *   `np.random.randn(d0, d1, ...)`: Random values (standard normal).
    *   `np.random.normal(loc, scale, size)`: Normal distribution samples.
    *   `np.random.poisson(lam, size)`: Poisson distribution samples.
*   **Array Attributes:**
    *   `arr.shape`: Tuple of array dimensions.
    *   `arr.ndim`: Number of dimensions.
    *   `arr.size`: Total number of elements.
    *   `arr.dtype`: Data type of elements (e.g., `float64`, `int32`).
*   **Indexing & Slicing:**
    *   `arr[i]`: Access element at index `i`.
    *   `arr[i, j]`: Access element at row `i`, column `j` (2D).
    *   `arr[start:stop:step]`: Slice along an axis.
    *   `arr[[idx1, idx2]]`: Indexing with a list of indices.
    *   `arr[bool_mask]`: Boolean mask indexing (select elements where mask is True).
*   **Mathematical Operations (Element-wise):** `+`, `-`, `*`, `/`, `**` (power), `np.sin()`, `np.cos()`, `np.exp()`, `np.log()`, `np.sqrt()`, etc.
*   **Reductions:**
    *   `np.sum(arr)`, `arr.sum()`: Sum of elements (can specify `axis`).
    *   `np.mean(arr)`, `arr.mean()`: Mean.
    *   `np.median(arr)`: Median.
    *   `np.std(arr)`, `arr.std()`: Standard deviation.
    *   `np.min(arr)`, `arr.min()`: Minimum value.
    *   `np.max(arr)`, `arr.max()`: Maximum value.
    *   `np.argmin(arr)`, `np.argmax(arr)`: Indices of min/max value.
    *   `np.percentile(arr, q)`: Calculate q-th percentile.
*   **Linear Algebra (`np.linalg`):** `np.dot()`, `np.linalg.inv()`, `np.linalg.solve()`, `np.linalg.eig()`, `np.linalg.svd()`, etc.
*   **File I/O:**
    *   `np.save('file.npy', arr)`: Save array to binary `.npy` file.
    *   `np.load('file.npy')`: Load array from `.npy` file.
    *   `np.savetxt('file.txt', arr)`: Save array to text file.
    *   `np.loadtxt('file.txt')`: Load array from text file.
    *   `np.genfromtxt('file.txt', ...)`: More robust text file loading.

**B.2 SciPy (`scipy`)**

*Collection of scientific algorithms built on NumPy.*

*   **Import:** `import scipy` (Submodules imported separately, e.g., `from scipy import stats`)
*   **Statistics (`scipy.stats`):**
    *   Distributions: `stats.norm`, `stats.poisson`, `stats.chi2`, `stats.uniform`, etc. (methods: `.pdf()`, `.cdf()`, `.ppf()`, `.rvs()`, `.fit()`).
    *   Tests: `stats.ttest_ind()`, `stats.ks_2samp()`, `stats.chi2_contingency()`.
    *   Descriptive: `stats.describe()`, `stats.iqr()`, `stats.skew()`, `stats.kurtosis()`.
    *   Regression: `stats.linregress()`.
*   **Optimization (`scipy.optimize`):**
    *   Curve Fitting: `optimize.curve_fit(func, xdata, ydata, p0, sigma)`.
    *   Minimization: `optimize.minimize(func, x0, method='...')`.
    *   Root Finding: `optimize.root_scalar()`, `optimize.root()`.
*   **Interpolation (`scipy.interpolate`):** `interpolate.interp1d()`, `interpolate.interp2d()`, `interpolate.griddata()`, spline functions.
*   **Signal Processing (`scipy.signal`):** `signal.convolve()`, `signal.correlate()`, `signal.fftconvolve()`, filtering functions (`signal.medfilt`, `signal.wiener`), peak finding (`signal.find_peaks`).
*   **Image Processing (`scipy.ndimage`):** Filters (`ndimage.gaussian_filter`, `ndimage.median_filter`), morphology (`ndimage.binary_erosion`), measurements (`ndimage.label`, `ndimage.center_of_mass`).
*   **Integration (`scipy.integrate`):** Numerical integration: `integrate.quad()`, `integrate.dblquad()`, `integrate.tplquad()`. ODE solvers: `integrate.solve_ivp()`.

**B.3 Matplotlib (`matplotlib.pyplot`)**

*Core plotting library.*

*   **Import:** `import matplotlib.pyplot as plt`
*   **Basic Plotting:**
    *   `plt.figure(figsize=(w, h))`: Create figure.
    *   `plt.plot(x, y, format_string, label='...')`: Line/marker plot (format e.g., `'ro-'`).
    *   `plt.scatter(x, y, s=size, c=color, alpha=alpha, label='...')`: Scatter plot.
    *   `plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', label='...')`: Plot with error bars.
    *   `plt.imshow(image_data, cmap='...', norm=..., origin='...', aspect='...')`: Display 2D image.
    *   `plt.contour(X, Y, Z, levels, colors='...')`: Contour plot.
    *   `plt.hist(data, bins=n, range=(min,max))`: Histogram.
    *   `plt.xlabel('Label (Unit)')`, `plt.ylabel('Label (Unit)')`: Axis labels.
    *   `plt.title('Plot Title')`: Plot title.
    *   `plt.legend()`: Display legend (requires `label=` in plot commands).
    *   `plt.grid(True)`: Add grid.
    *   `plt.xlim(min, max)`, `plt.ylim(min, max)`: Set axis limits.
    *   `plt.xscale('log')`, `plt.yscale('log')`: Set logarithmic scale.
    *   `plt.colorbar(mappable, label='...')`: Add colorbar for `imshow` or `scatter`.
    *   `plt.savefig('filename.png', dpi=300)`: Save figure to file.
    *   `plt.show()`: Display plot window.
*   **Object-Oriented Interface (Recommended for complex plots):**
    *   `fig, ax = plt.subplots()`: Create figure and single axes object.
    *   `fig, axes = plt.subplots(nrows=N, ncols=M)`: Create figure with grid of axes.
    *   Use `ax.` methods instead of `plt.` (e.g., `ax.plot()`, `ax.set_xlabel()`, `ax.set_title()`, `ax.legend()`, `ax.grid()`, `ax.imshow()`).
*   **Common Colormaps (`cmap=`):** `viridis`, `plasma`, `inferno`, `magma`, `cividis` (sequential), `gray`, `gray_r` (grayscale), `coolwarm`, `RdBu` (diverging), `jet` (**avoid**).
*   **Normalization (`norm=` with `imshow`):**
    *   `plt.Normalize(vmin=min, vmax=max)`: Linear scaling.
    *   From `astropy.visualization`: `simple_norm(data, stretch='linear', percent=99)`, `ManualInterval(vmin, vmax)`, `PercentileInterval(percent)`, `ZScaleInterval()`, `ImageNormalize(stretch=...)` using stretch objects like `LinearStretch()`, `LogStretch()`, `AsinhStretch()`, `SqrtStretch()`.

**B.4 Astropy (`astropy`)**

*Core package for astronomy.*

*   **Import:** `import astropy` (Subpackages usually imported explicitly, e.g., `from astropy.io import fits`)
*   **Units (`astropy.units`):**
    *   `import astropy.units as u`
    *   Attach units: `value * u.UnitName` (e.g., `10 * u.m`, `5 * u.km / u.s`, `1.5 * u.Msun`).
    *   Convert units: `quantity.to(u.NewUnit)` (e.g., `dist_pc.to(u.lightyear)`).
    *   Constants: `from astropy.constants import c, G, M_sun`, etc. (are `Quantity` objects).
    *   Composition: Units combine automatically (`area = length**2`, `velocity = length/time`).
    *   Equivalencies: `quantity.to(u.Hz, equivalencies=u.spectral())` for $\lambda \leftrightarrow \nu$.
*   **Coordinates (`astropy.coordinates`):**
    *   `from astropy.coordinates import SkyCoord`
    *   Create coordinate object:
        *   `SkyCoord(ra=10.5*u.deg, dec=-30.2*u.deg, frame='icrs')`
        *   `SkyCoord('5h35m17s -5d23m28s', frame='icrs')` (Parses string)
        *   `SkyCoord.from_name('M42')` (Online name resolution).
    *   Attributes: `coord.ra`, `coord.dec`, `coord.l`, `coord.b` (Galactic). Access values/units: `coord.ra.deg`, `coord.ra.hour`, `coord.ra.to_string()`.
    *   Transform frames: `coord_icrs.transform_to('galactic')`, `coord_icrs.transform_to(AltAz(obstime=time, location=loc))`.
    *   Separation: `coord1.separation(coord2)` (returns `Angle` object).
    *   Matching: `idx, d2d, d3d = coord_list1.match_to_catalog_sky(coord_list2)`.
*   **Time (`astropy.time`):**
    *   `from astropy.time import Time`
    *   Create time object: `Time('2023-10-27T12:00:00', format='isot', scale='utc')`, `Time(59880.5, format='mjd', scale='tdb')`.
    *   Attributes: `.jd`, `.mjd`, `.iso`, `.isot`, `.datetime`.
    *   Convert scales: `time_utc.tdb`.
    *   Time differences: `dt = time2 - time1` (returns `TimeDelta` object).
*   **Tables (`astropy.table`):**
    *   `from astropy.table import Table, Column, QTable` (`QTable` supports units).
    *   Create: `Table({'col1': [1, 2], 'col2': ['a', 'b']})`, `Table(rows=[...])`, `Table.read('file.csv', format='csv')`.
    *   Access columns: `table['col_name']`. Access rows: `table[index]`, `table[start:stop]`.
    *   Add/remove columns: `table['new_col'] = data`, `table.add_column(...)`, `table.remove_column(...)`.
    *   Masking: `table['col'] > 5`.
    *   Sorting: `table.sort('col_name')`.
    *   Grouping: `table.group_by('col_name')`.
    *   Joining: `join(table1, table2, keys='common_col')`.
    *   Writing: `table.write('file.ecsv', format='ascii.ecsv', overwrite=True)`.
*   **FITS I/O (`astropy.io.fits`):**
    *   `from astropy.io import fits`
    *   Open file: `with fits.open('image.fits') as hdul:`
    *   Access HDUs: `hdul[0]` (Primary), `hdul[1]` (First extension), `hdul['SCI']` (By EXTNAME). `hdul.info()`.
    *   Access header: `header = hdul[index_or_name].header`. Read keyword: `header['KEYWORD']`, `header.get('KEYWORD', default)`. Modify: `header['KEYWORD'] = value`. Add: `header['NEWKEY'] = (value, 'comment')`. History: `header.add_history('Processing step')`.
    *   Access data: `data = hdul[index_or_name].data` (returns NumPy array or `FITS_rec` for tables).
    *   Create HDUs: `fits.PrimaryHDU(data, header)`, `fits.ImageHDU(data, header)`, `fits.BinTableHDU(data, header)`, `fits.BinTableHDU.from_columns(columns)`.
    *   Create HDUList: `hdul = fits.HDUList([hdu1, hdu2, ...])`.
    *   Write file: `hdul.writeto('newfile.fits', overwrite=True)`.
*   **WCS (`astropy.wcs`):**
    *   `from astropy.wcs import WCS`
    *   Load WCS from header: `w = WCS(header)`.
    *   Pixel to World: `sky_coord = w.pixel_to_world(x_pix, y_pix)` (0-based).
    *   World to Pixel: `x_pix, y_pix = w.world_to_pixel(sky_coord)`.
    *   Access WCS keywords: `w.wcs.ctype`, `w.wcs.crval`, `w.wcs.crpix`, `w.wcs.cdelt`, `w.wcs.pc`, `w.wcs.cd`.
    *   Get celestial part: `w_celestial = w.celestial`.
    *   Create simple WCS: `w = WCS(naxis=2); w.wcs.crpix = ...; ...`
    *   Header from WCS: `wcs_header = w.to_header()`.
*   **Modeling (`astropy.modeling`):**
    *   `from astropy.modeling import models, fitting`
    *   Models: `models.Gaussian1D()`, `models.Polynomial1D(degree)`, `models.PowerLaw1D()`, `models.Const1D()`, `models.Sersic2D()`, etc.
    *   Instantiate: `g = models.Gaussian1D(amplitude=1.0, mean=0, stddev=1.0)`. Access/set params: `g.mean = 0.1`. Bounds: `g.stddev.bounds = (0, None)`. Fix: `g.mean.fixed = True`.
    *   Combine: `compound_model = models.Const1D() + models.Gaussian1D()`. Access submodels: `compound_model[0]`, `compound_model['Gaussian1D_1']`.
    *   Fitters: `fitter = fitting.LevMarLSQFitter()`, `fitter = fitting.LinearLSQFitter()`.
    *   Fit: `fitted_model = fitter(model_init, x, y, weights=1/err**2)`. Access results: `fitted_model.parameters`, `fitter.fit_info['param_cov']`.
*   **Visualization (`astropy.visualization`):**
    *   Normalization intervals: `ManualInterval()`, `PercentileInterval()`, `ZScaleInterval()`.
    *   Stretch functions: `LinearStretch()`, `LogStretch()`, `SqrtStretch()`, `AsinhStretch()`, `PowerStretch()`.
    *   Combine for imshow: `norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LogStretch())`.
    *   RGB images: `make_lupton_rgb(r, g, b, minimum, stretch, Q)`.
*   **Constants (`astropy.constants`):** `const.c`, `const.G`, `const.h`, `const.k_B`, `const.M_sun`, `const.R_sun`, `const.L_sun`, `const.au`, `const.pc`.

**B.5 Photutils (`photutils`)**

*Package for source detection and photometry.*

*   **Import:** `import photutils` (submodules often used explicitly)
*   **Background Estimation (`photutils.background`):**
    *   `from photutils.background import Background2D, MedianBackground, MADStdBackground`
    *   `from astropy.stats import SigmaClip`
    *   `sigma_clip = SigmaClip(sigma=3.0)`
    *   `bkg_estimator = MedianBackground()`
    *   `bkg = Background2D(data, box_size=(50,50), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)`
    *   Access maps: `bkg.background`, `bkg.background_rms`.
*   **Source Detection (`photutils.detection`, `photutils.segmentation`):**
    *   `from photutils.detection import DAOStarFinder, IRAFStarFinder, find_peaks`
    *   `from photutils.segmentation import detect_sources, SourceCatalog, SegmentationImage`
    *   Star finders: `daofind = DAOStarFinder(fwhm=3.0, threshold=5.*bkg_rms)`, `sources = daofind(data - bkg.background)`. Returns `Table`.
    *   Segmentation: `threshold = bkg.background + 3.0 * bkg.background_rms`, `segm = detect_sources(data, threshold, npixels=5)`. Returns `SegmentationImage`.
    *   Catalog from segmentation: `cat = SourceCatalog(data, segm, background=bkg.background)`, `props_table = cat.to_table()`.
*   **Aperture Photometry (`photutils.aperture`):**
    *   `from photutils.aperture import CircularAperture, EllipticalAperture, RectangularAperture, CircularAnnulus, aperture_photometry`
    *   Define apertures: `aperture = CircularAperture(positions, r=5.0)`. `positions` is (N, 2) array or tuple of arrays.
    *   Define sky annulus: `annulus = CircularAnnulus(positions, r_in=8.0, r_out=12.0)`.
    *   Perform photometry: `phot_table = aperture_photometry(data, aperture, error=error_map, mask=mask, local_bkg_annulus=annulus)`. Column `aperture_sum` contains background-subtracted flux if `local_bkg_annulus` used.
*   **PSF Photometry (`photutils.psf`):**
    *   `from photutils.psf import extract_stars, EPSFBuilder, IntegratedGaussianPRF, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry`
    *   `from astropy.modeling import models, fitting`
    *   Build ePSF: `stars = extract_stars(data, stars_tbl, size=25)`, `builder = EPSFBuilder(...)`, `epsf_model, fitted_stars = builder(stars)`.
    *   Define PSF model: `psf_model = IntegratedGaussianPRF(sigma=...)` or use `epsf_model`.
    *   Perform photometry: `phot_engine = BasicPSFPhotometry(psf_model, group_maker, bkg_estimator, fitter, fitshape)`, `result_table = phot_engine(image, init_guesses)`. (Similarly for `IterativelySubtractedPSFPhotometry`).

**B.6 Specutils (`specutils`)**

*Package for spectroscopic data analysis.*

*   **Import:** `import specutils` (Submodules often used explicitly)
*   **Core Object (`specutils.Spectrum1D`):**
    *   `from specutils import Spectrum1D`
    *   `spec = Spectrum1D(flux=flux_quantity, spectral_axis=wave_quantity, uncertainty=uncert, mask=mask, meta=header)` (Requires `astropy.units.Quantity` for flux/axis).
    *   Access attributes: `spec.flux`, `spec.spectral_axis`, `spec.uncertainty`, `spec.mask`, `spec.meta`.
*   **Spectral Axis Conversions:** `spec.spectral_axis.to(u.Hz, equivalencies=u.spectral())`, `spec.velocity_convention`, `spec.shift_spectrum_to()` (Doppler correction).
*   **Manipulation (`specutils.manipulation`):**
    *   Resampling: `resampler = LinearInterpolatedResampler()`, `spec_resampled = resampler(spec, new_spectral_axis)`. Also `FluxConservingResampler`, `SplineInterpolatedResampler`.
    *   Smoothing: `gaussian_smooth(spec, stddev=...)`, `median_smooth(spec, width=...)`.
*   **Fitting (`specutils.fitting`):**
    *   Continuum: `fit_generic_continuum(spec, model=Polynomial1D(2), exclude_regions=[...])`. Returns `astropy.modeling` model.
    *   Lines: `fit_lines(spec, model=Gaussian1D(), window=...)`. Returns fitted `astropy.modeling` model.
    *   Parameter estimation: `estimate_line_parameters(spec, models.Gaussian1D())`.
*   **Analysis (`specutils.analysis`):**
    *   `centroid(spec, region=...)`: Calculate flux-weighted centroid.
    *   `equivalent_width(spec, continuum=1.0, regions=...)`: Calculate EW.
    *   `fwhm(spec, regions=...)`: Calculate FWHM directly.
    *   `gaussian_sigma_width(spec, regions=...)`, `gaussian_fwhm(spec, regions=...)`: Width from Gaussian moment.
    *   `line_flux(spec, regions=...)`: Integrate flux within region (continuum subtracted assumed).
    *   `snr(spec)`: Estimate signal-to-noise ratio.
*   **Spectral Regions (`specutils.SpectralRegion`):** Define wavelength ranges: `SpectralRegion(lower_bound, upper_bound)`. Used in analysis/fitting functions.

**B.7 Lightkurve (`lightkurve`)**

*Package for Kepler/K2/TESS time series analysis.*

*   **Import:** `import lightkurve as lk`
*   **Data Search/Download:**
    *   `search_result = lk.search_lightcurve('Target Name or ID', mission='TESS', author='SPOC', sector=N)`
    *   `search_result_tpf = lk.search_targetpixelfile(...)`
    *   Download: `lc = search_result.download()`, `tpf = search_result_tpf.download()`, `lc_collection = search_result.download_all()`.
*   **LightCurve Object (`lc`):**
    *   Access data: `lc.time`, `lc.flux`, `lc.flux_err`, `lc.quality`.
    *   Basic operations: `lc.normalize()`, `lc.remove_nans()`, `lc.remove_outliers()`, `lc.flatten()`, `lc.bin()`.
    *   Plotting: `lc.plot()`, `lc.scatter()`.
    *   Folding: `lc_folded = lc.fold(period=P, epoch_time=t0)`. Plot folded: `lc_folded.plot()`.
    *   Periodograms: `pg = lc.to_periodogram(method='lombscargle')`, `bls_pg = lc.to_periodogram(method='bls', duration=...)`.
*   **TargetPixelFile Object (`tpf`):**
    *   Access data: `tpf.flux` (cube), `tpf.time`, `tpf.get_keyword(...)`, `tpf.get_header()`.
    *   Visualize: `tpf.plot(frame=index, aperture_mask=...)`.
    *   Create aperture mask: `aperture_mask = tpf.create_threshold_mask(...)`.
    *   Photometry: `lc_custom = tpf.to_lightcurve(aperture_mask=aperture_mask)`.
*   **Periodogram Objects (`pg`, `bls_pg`):**
    *   Access results: `pg.period`, `pg.power`, `pg.frequency`. `bls_pg.period`, `bls_pg.power`, `bls_pg.transit_time`, `bls_pg.depth`, `bls_pg.duration`.
    *   Find peak: `pg.period_at_max_power`, `bls.period_at_max_power`.
    *   Plotting: `pg.plot()`, `bls.plot()`.

**B.8 Astroquery (`astroquery`)**

*Programmatic access to online astronomical databases.*

*   **Import specific service:** `from astroquery.mast import Observations`, `from astroquery.gaia import Gaia`, `from astroquery.vizier import Vizier`, `from astroquery.ned import Ned`, `from astroquery.simbad import Simbad`.
*   **Common Usage Patterns:**
    *   Query by coordinates: `Observations.query_region(coords, radius=...)`, `Gaia.query_object_async(coordinate=coords, radius=...)`, `Vizier.query_region(coords, radius=..., catalog='...')`. `coords` usually an `astropy.coordinates.SkyCoord` object.
    *   Query by object name: `Observations.query_object('M31', radius=...)`, `Ned.query_object('M87')`, `Simbad.query_object('Betelgeuse')`.
    *   Get Tables: `result = Service.get_tables(...)`.
    *   Get Images/Data Products: `Observations.get_product_list(...)`, `Observations.download_products(...)`.
    *   ADQL Queries (e.g., Gaia, VizieR TAP): `Gaia.launch_job_async(adql_query).get_results()`.
*   **Results:** Typically returned as `astropy.table.Table` objects or lists of file paths.

This reference provides a starting point. The extensive capabilities of these libraries are best explored through their official online documentation and tutorials.
