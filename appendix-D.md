---
# Appendix D
# Units and Coordinates with Astropy (`astropy.units`, `astropy.coordinates`)
---

This appendix provides a more detailed exploration of two fundamental and powerful sub-packages within the Astropy library: `astropy.units` for handling physical units and `astropy.coordinates` for representing and transforming astronomical coordinate systems. Proficient use of these tools is essential for writing clear, robust, dimensionally consistent, and scientifically accurate astronomical analysis code in Python. They help prevent common errors related to unit conversions and coordinate transformations, making code more reliable and easier to interpret.

---

**D.1 Handling Physical Units (`astropy.units`)**

Scientific calculations invariably involve physical quantities that have associated units (e.g., meters, seconds, kilograms, Janskys, Angstroms, solar masses). Performing calculations manually while keeping track of units and performing necessary conversions is tedious and highly prone to errors. The `astropy.units` module provides a framework for associating physical units with numerical values, creating `Quantity` objects that automatically handle unit propagation, conversion, and dimensional consistency during arithmetic operations (Astropy Collaboration et al., 2022).

*   **D.1.1 Creating `Quantity` Objects**
    *   **Import:** `import astropy.units as u` (standard convention).
    *   **Basic Creation:** Multiply a numerical value (scalar, list, NumPy array) by a unit object from the `u` namespace:
        ```python
        import numpy as np
        import astropy.units as u

        distance = 10.0 * u.parsec
        wavelengths = np.array([4000, 5500, 7000]) * u.AA # Angstrom
        velocity = [-50.5, 10.0, 120.3] * u.km / u.s
        mass = 1.989e30 * u.kg
        flux_density = 1.5 * u.Jy
        temperature = 5778 * u.K
        ```
    *   **Parsing Unit Strings:** Create units from string representations:
        ```python
        momentum_unit = u.Unit("kg m / s")
        flux_unit = u.Unit("erg / (s cm**2 Angstrom)")
        accel = 9.8 * u.Unit("m s-2")
        ```
    *   **Checking Type:** `isinstance(distance, u.Quantity)` returns `True`.

*   **D.1.2 Arithmetic Operations**
    Arithmetic operations (+, -, *, /, **) between `Quantity` objects automatically handle units:
    ```python
    length1 = 10 * u.m
    length2 = 500 * u.cm
    total_length = length1 + length2 # Result is in meters (default base unit)
    print(total_length) # Output: 15.0 m

    speed = 100 * u.km / u.s
    time = 2 * u.hr
    distance_travelled = speed * time
    print(distance_travelled.to(u.km)) # Output: 720000.0 km
    print(distance_travelled.to(u.au)) # Output: ~0.00481 AU

    energy = 10 * u.J
    power = energy / (5 * u.s)
    print(power.to(u.W)) # Output: 2.0 W

    # Exponentiation:
    area = (5 * u.m)**2 # Output: 25.0 m2
    inv_time_sq = (1 * u.s)**(-2) # Output: 1.0 1 / s2
    ```
    Operations involving incompatible units (e.g., adding length and time) will raise a `UnitConversionError`. Operations with dimensionless quantities (e.g., multiplying by a plain number) work as expected.

*   **D.1.3 Unit Conversion (`.to()`, `.si`, `.cgs`)**
    *   **`.to()` Method:** Convert a `Quantity` to compatible units:
        ```python
        speed_kms = 100 * u.km / u.s
        speed_m_s = speed_kms.to(u.m / u.s) # Output: 100000.0 m / s
        speed_pc_Myr = speed_kms.to(u.pc / u.Myr) # Complex conversion
        print(speed_pc_Myr) # Output: ~102.2 pc / Myr
        # Attempting incompatible conversion raises error:
        # speed_kms.to(u.kg) # -> UnitConversionError
        ```
    *   **`.si` and `.cgs` Attributes:** Access the quantity's value represented in base SI or CGS units:
        ```python
        force = 10 * u.N
        print(force.si) # Output: 10.0 kg m / s2
        print(force.cgs) # Output: 1000000.0 g cm / s2
        ```

*   **D.1.4 Available Units and Constants**
    *   **Built-in Units:** `astropy.units` includes a vast library of predefined physical units (SI base and derived, CGS, astronomical units like `u.Msun`, `u.Lsun`, `u.Rsun`, `u.au`, `u.pc`, `u.lyr`, `u.Jy`, `u.mag`, `u.deg`, `u.arcmin`, `u.arcsec`, `u.AA`, `u.Hz`, etc.). Units can be accessed via `u.<UnitName>`.
    *   **Constants (`astropy.constants`):** Provides precise values of fundamental physical and astronomical constants as `Quantity` objects (e.g., `const.c`, `const.G`, `const.h`, `const.k_B`, `const.m_e`, `const.M_sun`, `const.R_earth`).
        ```python
        from astropy.constants import c, G, M_earth
        escape_vel = np.sqrt(2 * G * M_earth / (6371 * u.km))
        print(escape_vel.to(u.km/u.s)) # Output: ~11.18 km / s
        ```
    *   **Defining Custom Units:** `my_unit = u.def_unit('myunit', represents=1000 * u.m / u.s**2)`

*   **D.1.5 Equivalencies**
    Unit conversions sometimes require context or a physical relationship, handled by **equivalencies**.
    *   **Spectral Equivalencies (`u.spectral()`):** Convert between wavelength, frequency, and energy for electromagnetic radiation ($E=h\nu = hc/\lambda$).
        ```python
        wavelength = 5000 * u.AA
        frequency = wavelength.to(u.Hz, equivalencies=u.spectral())
        energy = wavelength.to(u.eV, equivalencies=u.spectral())
        print(frequency, energy)
        ```
    *   **Spectral Flux Density Equivalencies (`u.spectral_density(wavelength_or_frequency)`):** Convert between flux density units like $F_\lambda$ (e.g., erg/s/cm²/Å) and $F_\nu$ (e.g., Jy). Requires specifying the wavelength or frequency at which the conversion is performed.
        ```python
        flux_lambda = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
        lambda_pivot = 5500 * u.AA
        flux_nu = flux_lambda.to(u.Jy, equivalencies=u.spectral_density(lambda_pivot))
        print(flux_nu) # Output: ~0.00036 Jy
        ```
    *   **Doppler Equivalencies (`u.doppler_radio`, `u.doppler_optical`, `u.doppler_relativistic`):** Convert between frequency/wavelength shifts and velocities based on different Doppler formulas.
        ```python
        rest_freq = 1.420 * u.GHz # HI line
        vel = 100 * u.km / u.s
        obs_freq = vel.to(u.GHz, u.doppler_radio(rest_freq))
        print(obs_freq) # Output: ~1.419526 GHz
        ```
    *   **Other Equivalencies:** Parallax angle to distance (`u.parallax`), temperature scales (`u.temperature`), magnitude/flux (`u.mag_zero_point_equivalency`).

*   **D.1.6 Magnitudes and Logarithmic Units (`u.mag`, `u.dex`, `u.dB`)**
    `astropy.units` includes support for logarithmic units:
    *   **Magnitudes (`u.mag`):** Represents astronomical magnitudes. Specific systems (AB, ST) are handled via equivalencies or dedicated magnitude units (`u.ABmag`, `u.STmag`).
        ```python
        flux_jy = 1e-3 * u.Jy # 1 mJy
        ab_mag = flux_jy.to(u.ABmag) # Uses built-in AB zeropoint (3631 Jy)
        print(ab_mag) # Output: ~20.98 mag(AB)
        ```
    *   **Decibels (`u.dB`):** Ratio in dB.
    *   **Dex (`u.dex`):** Ratio in factors of 10 ($\log_{10}$).

Using `astropy.units` rigorously prevents common unit errors, ensures dimensional consistency, and makes code more readable and scientifically accurate by explicitly tracking the physical nature of quantities.

**D.2 Handling Astronomical Coordinates (`astropy.coordinates`)**

Representing positions and velocities on the sky, and transforming between different coordinate systems, are fundamental tasks in astronomy. The `astropy.coordinates` framework provides a powerful, object-oriented system for handling celestial and spatial coordinates, incorporating time dependence, distance information, proper motion, radial velocity, and standard reference frames (Astropy Collaboration et al., 2022; Price-Whelan et al., 2018).

*   **D.2.1 `SkyCoord`: The Core Object**
    The central class is `astropy.coordinates.SkyCoord`. It represents one or more spatial coordinates, optionally including distance and velocity information.
    *   **Creation:**
        ```python
        from astropy.coordinates import SkyCoord, Distance, Angle
        import astropy.units as u

        # From explicit RA/Dec values with units
        coord1 = SkyCoord(ra=150.1*u.deg, dec=2.2*u.deg, frame='icrs')
        # From formatted strings (parsed automatically)
        coord2 = SkyCoord('10h00m25.3s +02d12m10s', frame='icrs')
        coord3 = SkyCoord('10:00:25.3 +02:12:10', unit=(u.hourangle, u.deg)) # Explicit units
        # With distance
        coord_dist = SkyCoord(ra=270.0*u.deg, dec=-30.0*u.deg, distance=1.5*u.kpc, frame='icrs')
        # From Galactic coordinates
        coord_gal = SkyCoord(l=120.0*u.deg, b=-20.0*u.deg, frame='galactic')
        # Array of coordinates
        ras = [10.0, 11.0, 12.0] * u.deg
        decs = [5.0, 5.1, 5.2] * u.deg
        coord_array = SkyCoord(ra=ras, dec=decs, frame='icrs')
        ```
    *   **Attributes:** Access components like `.ra`, `.dec`, `.l`, `.b`, `.distance`. These return `Angle` or `Distance` objects (which are `Quantity` objects). Access numerical values with `.deg`, `.rad`, `.hour`, `.dms`, `.hms` for angles; `.pc`, `.kpc`, `.lyr`, `.au`, `.value` for distances.
    *   **Representation Types:** Coordinates can be represented in different ways internally (e.g., spherical, Cartesian). Access components via attributes like `.represent_as('spherical').lon`, `.represent_as('cartesian').x`.

*   **D.2.2 Coordinate Frames**
    Coordinates are always defined relative to a specific **frame**. `astropy.coordinates` defines numerous standard astronomical frames:
    *   **`ICRS`:** International Celestial Reference System. The standard inertial frame based on distant quasars, closely aligned with J2000 equatorial coordinates. Usually the default or preferred reference frame.
    *   **`FK5` / `FK4`:** Older equatorial coordinate systems based on catalogs of bright stars (account for precession relative to ICRS).
    *   **`Galactic`:** Coordinates based on the Milky Way plane ($l$: longitude, $b$: latitude).
    *   **`Galactocentric`:** Cartesian coordinates centered on the Galactic center, requiring assumptions about Sun's position and velocity.
    *   **`Supergalactic`:** Based on the local supercluster plane.
    *   **`AltAz`:** Horizon coordinates (Altitude, Azimuth). Requires specifying observation `obstime` (as `Time` object) and `location` (as `EarthLocation` object). Time-dependent.
    *   **`GeocentricTrueEcliptic` / `HeliocentricTrueEcliptic`:** Ecliptic coordinates centered on Earth or Sun.
    *   **Solar System Body Frames:** Frames centered on planets or the Sun (e.g., `Helioprojective`, `HeliographicStonyhurst` - often used via `sunpy.coordinates` integration).
    Frame information is crucial for transformations. `SkyCoord` objects store their frame internally.

*   **D.2.3 Coordinate Transformations (`.transform_to()`)**
    The power of the framework lies in easily transforming coordinates between different frames using the `.transform_to()` method. Astropy handles the complex calculations involving rotation matrices, precession, nutation, aberration (if relevant times are provided), and spherical trigonometry.
    ```python
    # Transform ICRS to Galactic
    coord_icrs = SkyCoord(ra=150.1*u.deg, dec=2.2*u.deg, frame='icrs')
    coord_galactic = coord_icrs.transform_to('galactic')
    print(f"ICRS: {coord_icrs.to_string('hmsdms')}")
    print(f"Galactic: l={coord_galactic.l.deg:.3f} deg, b={coord_galactic.b.deg:.3f} deg")

    # Transform to Altitude/Azimuth (requires time and location)
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    observing_time = Time('2024-03-15T22:00:00', scale='utc')
    # Example: Kitt Peak National Observatory location
    kpno_loc = EarthLocation.of_site('kpno')
    coord_altaz = coord_icrs.transform_to(AltAz(obstime=observing_time, location=kpno_loc))
    print(f"AltAz at {observing_time.iso}: Alt={coord_altaz.alt.deg:.2f} deg, Az={coord_altaz.az.deg:.2f} deg")
    ```

*   **D.2.4 Distances and Velocities**
    `SkyCoord` objects can store distance information (`distance` attribute, typically a `Distance` Quantity) and velocity components.
    *   **Distance:** Can be created using `Distance(value, unit)` or included directly in `SkyCoord` creation. Parallax angles can be converted using `Distance(parallax=angle)`.
    *   **Velocity Components:** Radial velocity (`radial_velocity`), proper motion in RA (`pm_ra_cosdec`) and Dec (`pm_dec`) can be included as `Quantity` objects during `SkyCoord` creation.
        ```python
        coord_vel = SkyCoord(ra=10*u.deg, dec=20*u.deg, frame='icrs',
                             distance=100*u.pc,
                             pm_ra_cosdec=5*u.mas/u.yr,
                             pm_dec=-10*u.mas/u.yr,
                             radial_velocity=25*u.km/u.s)
        ```
    *   **Space Motion:** When all 6D phase-space information (position + velocity) is present, transformations to other frames (like `Galactocentric`) correctly compute the 3D velocity vector in the new frame.

*   **D.2.5 Angular Separation and Catalog Matching**
    *   **Separation:** `coord1.separation(coord2)` calculates the on-sky angular distance.
    *   **Matching:** `idx, d2d, d3d = list1_coords.match_to_catalog_sky(list2_coords)` finds the nearest neighbor in `list2` for each coordinate in `list1` (Section 6.6). Essential for cross-matching catalogs.

*   **D.2.6 Working with Earth Locations (`EarthLocation`)**
    Defines locations on Earth for `AltAz` transformations or barycentric corrections.
    *   `EarthLocation.of_site('kpno')`: Get location for known observatories.
    *   `EarthLocation(lon=..., lat=..., height=...)`: Define custom location.

*   **D.2.7 Integration with WCS (`astropy.wcs`)**
    The `astropy.wcs.WCS` object's `.pixel_to_world()` method often returns `SkyCoord` objects when the WCS defines celestial coordinates, seamlessly integrating pixel positions with the coordinate framework. Similarly, `w.world_to_pixel()` accepts `SkyCoord` objects.

The `astropy.coordinates` framework provides a robust, accurate, and user-friendly system for managing the complexities of astronomical coordinate systems and transformations. Its integration with `astropy.units` and `astropy.time` ensures consistency and reduces potential errors in astrophysical calculations involving positions, distances, and velocities.

---

**References**

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* The primary paper describing the Astropy project and its community model. This appendix focuses on two core sub-packages, `astropy.units` (D.1) and `astropy.coordinates` (D.2), whose functionalities are detailed herein.

Bevington, P. R., & Robinson, D. K. (2003). *Data reduction and error analysis for the physical sciences* (3rd ed.). McGraw-Hill. *(Note: Classic textbook, pre-2020)*
*   *Summary:* Although pre-2020, this is a classic textbook covering fundamental concepts in data analysis and error propagation, relevant background for understanding the need for rigorous handling of units and uncertainties facilitated by `astropy.units` (D.1).

Feigelson, E. D., & Babu, G. J. (2012). *Modern Statistical Methods for Astronomy: With R Applications*. Cambridge University Press. https://doi.org/10.1017/CBO9781139179014 *(Note: Key statistics textbook, pre-2020)*
*   *Summary:* While pre-2020, this essential textbook provides a comprehensive overview of statistical methods used in astronomy, forming the basis for many analysis techniques where proper handling of units (`astropy.units`, D.1) and coordinates (`astropy.coordinates`, D.2) is crucial.

Gaia Collaboration, Vallenari, A., Brown, A. G. A., Prusti, T., de Bruijne, J. H. J., Arenou, F., Babusiaux, C., Biermann, M., Creevey, O. L., Ducourant, C., Evans, D. W., Eyer, L., Guerra, R., Hutton, A., Jordi, C., Klioner, S. A., Lammers, U. L., Lindegren, L., Luri, X., … Zwitter, T. (2023). Gaia Data Release 3: Summary of the contents, processing, and validation. *Astronomy & Astrophysics, 674*, A1. https://doi.org/10.1051/0004-6361/202243940
*   *Summary:* Summarizes Gaia DR3. Interacting with Gaia data heavily relies on `astropy.coordinates` (D.2) for handling positions, proper motions, and parallaxes within the ICRS frame, and `astropy.units` (D.1) for interpreting photometric and astrometric quantities.

Gregory, P. (2005). *Bayesian Logical Data Analysis for the Physical Sciences: A Comparative Approach with Mathematica® Support*. Cambridge University Press. https://doi.org/10.1017/CBO9780511535424 *(Note: Key Bayesian textbook, pre-2020)*
*   *Summary:* A standard textbook on Bayesian data analysis in physics/astronomy. Bayesian modeling often involves calculations with physical quantities where `astropy.units` (D.1) ensures correctness, and coordinate handling (`astropy.coordinates`, D.2) is needed for spatial models.

Ji, X., Frebel, A., Chiti, A., Simon, J. D., Jerkstrand, A., Lin, D., Thompson, I. B., Aguilera-Gómez, C., Casey, A. R., Gomez, F. A., Han, J., Ji, A. P., Kim, D., Marengo, M., McConnachie, A. W., Stringfellow, G. S., & Yoon, J. (2023). Chemical abundances of the stars in the Tucana II ultra-faint dwarf galaxy. *The Astronomical Journal, 165*(1), 26. https://doi.org/10.3847/1538-3881/aca4a5
*   *Summary:* This study performs detailed kinematic and chemical analysis of stars identified using Gaia data. Such analysis heavily relies on precise coordinate transformations and velocity calculations facilitated by `astropy.coordinates` (D.2) and handling physical quantities with `astropy.units` (D.1).


Price-Whelan, A. M., Sipőcz, B. M., Günther, H. M., Lim, P. L., Crawford, S. M., Conseil, S., Shupe, D. L., Craig, M. W., Dencheva, N., Ginsburg, A., VanderPlas, J. T., Bradley, L. D., Pérez-Suárez, D., de Val-Borro, M., Aldcroft, T. L., Cruz, K. L., Robitaille, T. P., Tollerud, E. J., Aranda, C., … Astropy Collaboration. (2018). The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package. *The Astronomical Journal, 156*(3), 123. https://doi.org/10.3847/1538-3881/aabc4f *(Note: Astropy v2.0 paper, pre-2020)*
*   *Summary:* This paper details Astropy v2.0, including significant developments in the `astropy.coordinates` (D.2) framework, such as improved frame transformations and velocity support. Provides background on the package's capabilities.

Sandford, N. R., Maseda, M. V., Chevallard, J., Tacchella, S., Arribas, S., Charlton, J. C., Curtis-Lake, E., Egami, E., Endsley, R., Hainline, K., Johnson, B. D., Robertson, B. E., Shivaei, I., Stacey, H., Stark, D. P., Williams, C. C., Boyett, K. N. K., Bunker, A. J., Charlot, S., … Willott, C. J. (2024). Emission line ratios from the JWST NIRSpec G395H spectrum of GN-z11. *arXiv preprint arXiv:2401.16955*. https://doi.org/10.48550/arXiv.2401.16955
*   *Summary:* Analysis of JWST spectra involves precise wavelength calibration (requiring units, D.1), flux measurements (units, D.1), and potentially coordinate information (D.2) from the instrument WCS. Highlights the use context for these tools.


Wall, J. V., & Jenkins, C. R. (2012). *Practical Statistics for Astronomers* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139168490 *(Note: Key statistics textbook, pre-2020)*
*   *Summary:* A practical guide to statistics for astronomers (pre-2020). Understanding the statistical concepts presented relies on correctly handling the physical quantities involved, highlighting the importance of tools like `astropy.units` (D.1).

Villaescusa-Navarro, F., Angles-Alcazar, D., Genel, S., Nagai, D., Nelson, D., Pillepich, A., Hernquist, L., Marinacci, F., Pakmor, R., Springel, V., Vogelsberger, M., ZuHone, J., & Weinberger, R. (2023). Splashdown: Representing cosmological simulations through neural networks. *The Astrophysical Journal Supplement Series, 266*(2), 38. https://doi.org/10.3847/1538-4365/accc3e
*   *Summary:* Working with cosmological simulation outputs requires consistent handling of cosmological units (often Mpc, Msun, velocity units), distances, and coordinate systems, tasks facilitated by `astropy.units` (D.1) and potentially `astropy.coordinates` (D.2) when mapping to sky coordinates.

