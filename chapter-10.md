---
# Chapter 10
# Large-Scale Data Analysis I: Machine Learning Methods
---

This chapter inaugurates the discussion on analyzing the immense datasets characteristic of modern astronomy, focusing specifically on the application of Machine Learning (ML) techniques. It begins by contextualizing the current era of large astronomical surveys—such as the Vera C. Rubin Observatory (LSST), Square Kilometre Array (SKA), Euclid, and Roman Space Telescope—emphasizing the unprecedented data volumes, velocities, and complexities that necessitate advanced computational approaches beyond traditional methods. Techniques for efficient data management within this "Big Data" paradigm are explored, including the role of astronomical databases and the use of structured query languages like SQL and the astronomy-specific ADQL, accessed programmatically via tools like `pyvo` and `astroquery`. The chapter then introduces fundamental concepts of Machine Learning relevant to astronomers, differentiating between supervised and unsupervised learning paradigms, discussing crucial aspects like feature engineering, model training, validation, and performance evaluation, primarily through the lens of the widely used `scikit-learn` Python library. Common astronomical applications where ML has proven transformative are surveyed, including source classification, anomaly detection, photometric redshift estimation, and pattern recognition in complex datasets. Finally, a brief introduction to Deep Learning (DL) concepts, encompassing neural networks and specialized architectures like Convolutional Neural Networks (CNNs), is provided, acknowledging their growing importance in tackling specific astronomical challenges, with references to prevalent frameworks such as TensorFlow and PyTorch. Practical examples across diverse astronomical sub-disciplines illustrate the application of these ML techniques using standard Python tools.

---

**10.1 The Era of Large Astronomical Surveys**

Contemporary astrophysics is undergoing a profound transformation driven by the advent of large-scale, systematic surveys across the electromagnetic spectrum and involving multiple cosmic messengers (Ivezić et al., 2020; Lochner & Bassett, 2021). Facilities like the Zwicky Transient Facility (ZTF), the Dark Energy Survey (DES), the Sloan Digital Sky Survey (SDSS), Gaia, eROSITA, ASKAP, MeerKAT, and upcoming endeavors such as the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), the Square Kilometre Array (SKA), the Euclid space mission, and the Nancy Grace Roman Space Telescope are generating datasets of unparalleled size, complexity, and cadence (Padovani et al., 2023; Ivezić et al., 2020; Dalton et al., 2022; Bailey et al., 2023; Spergel et al., 2015). These surveys are moving astronomy from an era of targeted observations of individual objects or small fields to one characterized by the systematic mapping of vast cosmic volumes, producing data streams measured in terabytes per night and archives accumulating petabytes or even exabytes over their lifetimes.

This data deluge presents both extraordinary opportunities for scientific discovery and formidable challenges for data processing, analysis, and interpretation (Fluke & Jacobs, 2020). The sheer **volume** necessitates highly scalable storage infrastructure, efficient data processing pipelines often leveraging High-Performance Computing (HPC) resources (Chapter 11), and algorithms optimized for handling massive datasets. The high **velocity** of data acquisition, particularly for transient surveys like LSST which aim to detect and alert on celestial changes within minutes, demands automated, real-time analysis capabilities. The inherent **variety** of astronomical data, spanning different wavelengths, resolutions, cadences, and data types (images, spectra, time series, catalogs, event lists), requires sophisticated methods for data fusion, cross-matching, and joint analysis, often relying on Virtual Observatory standards for interoperability (Chapter 2, Chapter 17). Furthermore, ensuring the **veracity** and **value** of the scientific insights extracted from these vast datasets requires rigorous calibration, meticulous uncertainty quantification, robust statistical methods, and the development of novel analysis techniques capable of identifying subtle patterns or rare phenomena hidden within the noise (Siebert et al., 2022).

Traditional methods of data analysis, often involving manual inspection or algorithms designed for smaller datasets, become computationally infeasible or statistically inadequate in this new regime. Processing petabytes of image data, searching for faint signals in billions of time series, classifying billions of detected sources, or identifying rare anomalies within massive parameter spaces requires a paradigm shift towards automated, data-driven approaches. Machine Learning (ML) and Deep Learning (DL) techniques have emerged as powerful tools uniquely suited to address many of these challenges (Lochner & Bassett, 2021; Ntampaka et al., 2019; Fluke & Jacobs, 2020). ML algorithms can learn complex patterns and relationships directly from data, enabling automated classification, regression, clustering, dimensionality reduction, and anomaly detection tasks at scales previously unimaginable. This chapter focuses on introducing the fundamental concepts of ML and exploring their application to large-scale astronomical data analysis, laying the groundwork for tackling the scientific opportunities presented by the "Big Data" era in astronomy.

**10.2 Efficient Data Storage and Querying: Astronomical Databases**

Managing the petabyte-scale datasets generated by modern astronomical surveys necessitates sophisticated data storage and querying systems that go beyond simple file-based approaches. While FITS remains the standard for data interchange and archival (Chapter 2), interacting efficiently with massive catalogs containing billions of rows (e.g., source properties, measurements) or accessing specific subsets of large image or cube collections requires database technologies and standardized query protocols (Gray et al., 2005; Plante et al., 2011; Allen et al., 2022).

**Relational Databases and SQL:** Traditional relational databases, managed by systems like PostgreSQL, MySQL, or Oracle, organize data into tables with predefined schemas (columns with specific data types). The **Structured Query Language (SQL)** is the standard language for interacting with these databases, allowing users to perform complex operations like:
*   **Selecting data:** Retrieving specific rows and columns based on various criteria (`SELECT ... FROM ... WHERE ...`).
*   **Filtering:** Applying conditions to select subsets of data (`WHERE` clause with logical operators, comparisons, range checks).
*   **Joining tables:** Combining information from multiple related tables based on common keys (e.g., matching source IDs between a detection table and a photometry table using `JOIN ... ON ...`).
*   **Aggregation:** Calculating summary statistics (e.g., counts, averages, minimums, maximums) across groups of data (`GROUP BY ...` with functions like `COUNT()`, `AVG()`, `MIN()`, `MAX()`).
*   **Sorting:** Ordering results based on specific columns (`ORDER BY ...`).
Major astronomical archives and surveys often store their primary catalogs in relational databases, optimized with indexing strategies to allow rapid querying even on tables with billions of entries. While direct SQL access might be provided, often specialized interfaces are built on top.

**Astronomical Data Query Language (ADQL):** Recognizing the specific needs of astronomical querying, particularly spatial searches on the celestial sphere, the International Virtual Observatory Alliance (IVOA) developed the **Astronomical Data Query Language (ADQL)** (Ortiz et al., 2008). ADQL is largely based on SQL but extends it with:
*   **Standard Astronomical Functions:** Includes functions for trigonometric calculations, unit conversions, and common astronomical operations.
*   **Geometric Functions:** Critically, ADQL defines functions for performing spatial queries on the sky, based on standard coordinate systems (usually ICRS). Key functions include:
    *   `POINT(coord_sys, ra, dec)`: Defines a point on the sky.
    *   `CIRCLE(coord_sys, ra_cen, dec_cen, radius)`: Defines a circular region.
    *   `BOX(coord_sys, ra_cen, dec_cen, width, height)`: Defines a rectangular region.
    *   `POLYGON(coord_sys, ra1, dec1, ra2, dec2, ...)`: Defines a polygonal region.
    *   `CONTAINS(region1, region2)`: Checks if region1 contains region2 (e.g., if a POINT is within a CIRCLE).
    *   `INTERSECTS(region1, region2)`: Checks if two regions overlap.
    *   `DISTANCE(point1, point2)`: Calculates the angular separation between two points.
These geometric functions allow users to perform efficient **cone searches** (finding objects within a certain radius of a point), box searches, or more complex spatial selections directly within the query language.

**Programmatic Access (`pyvo`, `astroquery`):** While database query languages like SQL and ADQL are powerful, performing analysis often requires integrating data retrieval directly into computational workflows, typically written in Python. Several libraries facilitate this programmatic access:
*   **`astroquery`:** This Astropy-affiliated package provides a unified interface to query a vast number of online astronomical data centers and services (e.g., SIMBAD, NED, VizieR, MAST, IRSA, CADC, ESA Archives including Gaia) (Ginsburg et al., 2019). It abstracts the underlying service protocols (often web APIs or VO protocols like TAP), allowing users to query for data based on object names, coordinates, catalog IDs, or other parameters directly from Python scripts. For services supporting ADQL (like Gaia or VizieR TAP services), `astroquery` allows submission of complex ADQL queries. Results are typically returned as `astropy.table.Table` objects or file paths to downloaded data.
*   **`pyvo`:** This package provides a lower-level interface specifically for interacting with Virtual Observatory (VO) services based on standard IVOA protocols (Demleitner et al., 2023). Key protocols include:
    *   **Table Access Protocol (TAP):** The standard for querying tabular data (catalogs) using ADQL. `pyvo.dal.TAPService` allows connecting to TAP endpoints, submitting synchronous or asynchronous ADQL queries, and retrieving results as `VOTable` objects (which can be converted to Astropy Tables).
    *   **Simple Image Access (SIA), Simple Spectral Access (SSA), Simple Cone Search (SCS):** Older protocols for discovering and retrieving images, spectra, or catalog data within a specified region of the sky. `pyvo` provides interfaces to query these services as well.
    *   **DataLink:** A protocol for discovering related data products associated with a primary dataset (e.g., finding calibration files, previews, or spectra associated with an image).
`pyvo` is often used internally by `astroquery` but can also be used directly for more fine-grained control over VO interactions.

These database technologies and programmatic interfaces are essential tools for efficiently navigating and accessing the massive datasets required for ML-driven analysis in the era of large surveys, enabling researchers to extract relevant subsets of data for training models or applying learned algorithms. The example below demonstrates a simple cone search using `astroquery` to retrieve Gaia data via its TAP service using an ADQL query, illustrating programmatic access to large survey catalogs.

This Python code utilizes the `astroquery` library to programmatically query the Gaia DR3 catalog, a quintessential example of accessing large astronomical databases essential for ML training and analysis. It defines search parameters – the central sky coordinates (`SkyCoord`) and a search radius (`Angle`). An ADQL query string is constructed to select specific columns (source ID, position, parallax, proper motions, G-band magnitude) from the `gaiadr3.gaia_source` table for all sources falling within a circular region defined by the `CONTAINS`, `POINT`, and `CIRCLE` ADQL functions. The query is submitted asynchronously to the Gaia archive TAP service using `Gaia.launch_job_async`, and the results are retrieved as an `astropy.table.Table` object, ready for further processing or use as input for ML tasks.

```python
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
# Requires astroquery: pip install astroquery
try:
    from astroquery.gaia import Gaia
    astroquery_available = True
except ImportError:
    print("astroquery not found, skipping database query example.")
    astroquery_available = False

# Define search parameters for querying Gaia DR3
search_center_ra = 185.0 # degrees (e.g., Coma Cluster region)
search_center_dec = 27.5 # degrees
search_radius_arcmin = 10.0 # arcminutes

if astroquery_available:
    # Create SkyCoord object for the search center
    search_coord = SkyCoord(ra=search_center_ra*u.deg, dec=search_center_dec*u.deg, frame='icrs')
    search_radius = search_radius_arcmin * u.arcmin
    print(f"Preparing to query Gaia DR3 around {search_coord.to_string('hmsdms')} within {search_radius:.1f}...")

    # Construct the ADQL query string
    # Select relevant columns: source_id, position, parallax, proper motions, G magnitude
    # Use WHERE clause with CONTAINS(POINT(...), CIRCLE(...)) for cone search
    adql_query = f"""
    SELECT
      source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM
      gaiadr3.gaia_source
    WHERE
      1=CONTAINS(POINT('ICRS', ra, dec),
                 CIRCLE('ICRS', {search_coord.ra.deg}, {search_coord.dec.deg}, {search_radius.to(u.deg).value}))
      AND parallax IS NOT NULL AND parallax > 0  -- Example: Select stars with positive parallax
    ORDER BY phot_g_mean_mag ASC -- Order by brightness
    """
    # Can add more constraints, e.g., on magnitude limits, parallax error, etc.

    print("Submitting ADQL query to Gaia Archive...")
    try:
        # Launch the query asynchronously (recommended for potentially large queries)
        job = Gaia.launch_job_async(adql_query)
        # Get the results table when the job completes
        results_table = job.get_results()
        # Results table is an Astropy Table object

        print(f"\nQuery successful! Retrieved {len(results_table)} sources.")
        print("First 5 rows of the results table:")
        print(results_table[:5])

        # This results_table can now be used for further analysis, feature
        # engineering, or as input to machine learning algorithms.
        # For example, calculate absolute magnitude using parallax:
        # distance_pc = 1000.0 / results_table['parallax'] # Parallax in mas
        # abs_mag_g = results_table['phot_g_mean_mag'] - 5 * np.log10(distance_pc) + 5

    except Exception as e:
        print(f"An error occurred during the Gaia query: {e}")
else:
    print("Skipping database query example: astroquery unavailable.")

```

The preceding Python script effectively demonstrates accessing large astronomical survey data stored in online databases through programmatic queries, a vital step for many Machine Learning applications. It utilizes the `astroquery.gaia` module to interact with the Gaia DR3 archive. After defining the desired sky region (center coordinates and search radius), it constructs a query using the Astronomical Data Query Language (ADQL). This query specifically selects sources within a circular region on the sky (`CONTAINS`, `CIRCLE`) and retrieves key parameters like position, parallax, proper motion, and magnitude. The query is submitted to the Gaia archive's Table Access Protocol (TAP) service, and the results are returned as an `astropy.table.Table`. This table, containing potentially thousands or millions of entries depending on the search region, provides structured data readily usable for subsequent feature extraction and ML model training or application.

**10.3 Machine Learning Concepts for Astronomers (`scikit-learn`)**

Machine Learning (ML) provides a powerful set of computational tools for finding patterns, making predictions, and extracting insights from data without being explicitly programmed for the specific task (Hastie et al., 2009; Murphy, 2012; Bishop, 2006). Instead of relying on predefined rules, ML algorithms learn from examples provided in a training dataset. Given the large, complex, and high-dimensional datasets prevalent in modern astronomy, ML techniques have become indispensable for automating tasks like classification, regression, clustering, and anomaly detection (Lochner & Bassett, 2021; Fluke & Jacobs, 2020). The **`scikit-learn`** library is the cornerstone of general-purpose ML in Python, providing efficient implementations of a vast array of algorithms along with tools for data preprocessing, model selection, and evaluation (Pedregosa et al., 2011).

ML paradigms are broadly categorized based on the type of learning task:
1.  **Supervised Learning:** The algorithm learns a mapping from input features ($X$) to known output labels or values ($y$) based on a labeled training dataset $\{(X_i, y_i)\}$.
    *   **Classification:** The goal is to assign input data points to predefined discrete categories or classes. The output label $y$ is categorical. Examples in astronomy include: classifying objects as stars, galaxies, or quasars based on photometric colors or morphology; identifying transient types (e.g., SN Ia vs. SN II) based on light curve features; classifying galaxy morphologies. Common algorithms include: Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM), K-Nearest Neighbors (KNN), and Neural Networks (Section 10.5).
    *   **Regression:** The goal is to predict a continuous output value $y$ based on input features $X$. Examples include: estimating photometric redshifts (photo-z) for galaxies based on their magnitudes/colors; predicting stellar parameters (temperature, gravity, metallicity) from spectra; estimating distances based on light curve properties. Common algorithms include: Linear Regression, Ridge Regression, Lasso, Polynomial Regression, Support Vector Regression (SVR), Decision Trees/Random Forests for regression, Gradient Boosting Regressors, and Neural Networks.
2.  **Unsupervised Learning:** The algorithm explores the inherent structure within an unlabeled dataset ($X$) without predefined outputs.
    *   **Clustering:** The goal is to group similar data points together into clusters based on their features, without prior knowledge of group assignments. Examples include: discovering new classes of variable stars based on light curve shapes; grouping galaxies based on their properties to identify distinct populations; identifying structures in high-dimensional parameter spaces from simulations. Common algorithms include: K-Means Clustering, DBSCAN, Hierarchical Clustering (Agglomerative Clustering), Gaussian Mixture Models (GMM).
    *   **Dimensionality Reduction:** The goal is to reduce the number of input features (dimensionality) while preserving the most important information or structure in the data. This is useful for visualization, noise reduction, feature extraction, and improving the performance of subsequent supervised learning algorithms. Examples include: reducing the dimensionality of spectral data or multi-band photometry before classification; visualizing high-dimensional catalog data in 2D or 3D. Common algorithms include: Principal Component Analysis (PCA), Independent Component Analysis (ICA), t-Distributed Stochastic Neighbor Embedding (t-SNE), Uniform Manifold Approximation and Projection (UMAP).
    *   **Anomaly/Outlier Detection:** The goal is to identify data points that deviate significantly from the typical patterns or distributions within the dataset. Examples include: finding rare or unusual astronomical objects or events in large surveys; identifying faulty data points or instrumental artifacts. Algorithms often involve density estimation (e.g., using GMMs, Kernel Density Estimation) or distance-based methods (e.g., Isolation Forest, Local Outlier Factor - LOF).

Key concepts in the ML workflow, facilitated by `scikit-learn`, include:
*   **Features ($X$):** The measurable input variables or attributes used to describe each data point (e.g., magnitudes, colors, spectral line widths, image moments, light curve statistics). Selecting and engineering informative features (**feature engineering**) is often critical for ML model performance. **Feature scaling** (e.g., standardization to zero mean and unit variance, or normalization to [0, 1]) is usually necessary as many algorithms are sensitive to the relative scales of different features.
*   **Training Data:** The dataset used to train the ML model. For supervised learning, this includes both features ($X_{train}$) and corresponding known labels/values ($y_{train}$).
*   **Model Training/Fitting:** The process where the algorithm learns the patterns or mapping from the training data by adjusting its internal parameters to optimize a specific objective function (e.g., minimizing prediction error for regression, maximizing classification accuracy, maximizing likelihood for probabilistic models). `scikit-learn` provides a consistent `.fit(X_train, y_train)` interface for training most models.
*   **Model Evaluation:** Assessing the performance of the trained model on *unseen* data is crucial to ensure it generalizes well and avoids **overfitting** (where the model learns the training data too well, including its noise, but performs poorly on new data). This typically involves splitting the available data into separate **training** and **test** sets (`sklearn.model_selection.train_test_split`). The model is trained only on the training set and evaluated on the independent test set using appropriate performance **metrics** (e.g., accuracy, precision, recall, F1-score, AUC for classification; Mean Squared Error (MSE), R-squared ($R^2$) for regression; Silhouette score for clustering).
*   **Cross-Validation:** A more robust evaluation technique where the training data is further split into multiple "folds." The model is trained on all but one fold and evaluated on the held-out fold, rotating which fold is held out. This provides a more stable estimate of the model's generalization performance (`sklearn.model_selection.cross_val_score`, `KFold`).
*   **Hyperparameter Tuning:** ML algorithms often have "hyperparameters" (e.g., number of trees in a Random Forest, regularization strength in SVM/Ridge, number of neighbors in KNN, number of clusters in K-Means) that are not learned from data directly but must be set beforehand. Finding the optimal hyperparameter values often involves searching over a grid of possibilities (`sklearn.model_selection.GridSearchCV`) or using randomized search (`RandomizedSearchCV`), evaluating performance for each combination using cross-validation.

`scikit-learn` provides a unified, well-documented API for accessing a wide range of ML algorithms, preprocessing tools (scaling, imputation), model selection utilities (train/test split, cross-validation, grid search), and performance metrics, making it an indispensable tool for applying ML techniques to astronomical data analysis.

**10.4 Astrocomputing Applications of Machine Learning**

Machine Learning algorithms have found diverse and impactful applications across virtually all subfields of astronomy, driven by the need to efficiently analyze large, complex datasets and automate tasks that were previously laborious or intractable (Lochner & Bassett, 2021; Ntampaka et al., 2019; Asgari et al., 2023; Fluke & Jacobs, 2020). ML excels at identifying patterns, classifying objects, predicting properties, and detecting anomalies within high-dimensional parameter spaces. Some prominent applications include:

1.  **Source Classification:** Distinguishing between different types of astronomical objects based on their observed properties.
    *   *Star/Galaxy/Quasar Separation:* Classifying objects in imaging surveys based on morphological features (e.g., concentration, asymmetry) and photometric properties (colors). Early applications used decision trees or SVMs; deep learning (CNNs on images) is now common (e.g., Kim & Brunner, 2017).
    *   *Transient Classification:* Automatically classifying detected transient events (supernovae, stellar flares, variable stars, etc.) based on features extracted from their light curves (shape, duration, color evolution) or contextual information. Random Forests, Gradient Boosting, and Recurrent Neural Networks (RNNs) are often employed by alert brokers (Section 8.4) (Förster et al., 2021; Möller et al., 2021; Pichara et al., 2021).
    *   *Variable Star Classification:* Grouping stars into known variability classes (Cepheids, RR Lyrae, eclipsing binaries, etc.) based on detailed light curve characteristics derived from surveys like OGLE, Kepler, TESS, and ZTF. Methods range from feature-based classification with Random Forests to direct light curve classification with CNNs or RNNs.

2.  **Regression and Parameter Estimation:** Predicting continuous physical parameters from observational data.
    *   *Photometric Redshifts (Photo-z):* Estimating the redshifts of galaxies using only their observed magnitudes or fluxes in multiple filter bands, without requiring expensive spectroscopic observations. This is crucial for analyzing large imaging surveys (LSST, Euclid). Techniques range from empirical methods like template fitting and polynomial fitting to ML methods like K-Nearest Neighbors, Random Forests, Artificial Neural Networks (ANNs), and Gaussian Processes (Salvato et al., 2019; Euclid Collaboration et al., 2024). ML methods often achieve higher accuracy by learning complex, non-linear relationships between colors and redshift.
    *   *Stellar Parameter Estimation:* Predicting stellar properties like effective temperature ($T_{eff}$), surface gravity ($\log g$), and metallicity ([Fe/H]) from low- or high-resolution spectra or multi-band photometry. Regression algorithms (e.g., ANNs, Random Forests, SVMs) trained on stars with known parameters from reference catalogs or detailed modeling are widely used (e.g., The Cannon - Ness et al., 2015).
    *   *Cosmological Parameter Inference:* ML, particularly simulation-based inference (SBI) or likelihood-free inference (LFI) using neural networks, is increasingly used to estimate cosmological parameters directly from complex summary statistics derived from large-scale structure data (simulations or observations) or CMB maps, potentially bypassing the need for explicit likelihood calculations (Alsing et al., 2019; Cranmer et al., 2020).

3.  **Clustering and Discovery:** Finding groups or structure in unlabeled data.
    *   *Discovering New Object Classes:* Applying clustering algorithms (e.g., K-Means, DBSCAN, hierarchical clustering) to large datasets based on photometric colors, spectral features, or light curve shapes can reveal previously unknown populations or subclasses of objects exhibiting distinct properties.
    *   *Identifying Structures:* Clustering stars in multi-dimensional spaces defined by position, kinematics (from Gaia), and chemistry (from spectroscopic surveys) helps identify co-moving groups, stellar streams, or chemically distinct populations within the Milky Way (e.g., Hunt et al., 2023).

4.  **Anomaly Detection:** Identifying rare, unusual, or unexpected objects or events that deviate significantly from the bulk of the data.
    *   *Finding Rare Transients:* Algorithms like Isolation Forests or autoencoders can be trained on large sets of "normal" light curves or difference image alerts to flag events with unusual shapes, durations, or colors that might represent novel astrophysical phenomena (Pruzhinskaya et al., 2019; Ishida et al., 2021).
    *   *Identifying Peculiar Objects:* Searching for outliers in large multi-dimensional parameter spaces derived from catalogs (e.g., unusual colors, extreme kinematic properties) can lead to the discovery of rare types of stars, galaxies, or quasars.
    *   *Data Quality Control:* Identifying instrumental artifacts or problematic data points that appear as outliers in feature space.

5.  **Data Processing and Calibration Enhancement:**
    *   *Cosmic Ray Rejection:* ML, particularly CNNs, can be trained to identify and mask cosmic ray hits in images, potentially outperforming traditional algorithms in complex scenarios (Section 3.8) (Zhang & Bloom, 2020; Jia et al., 2023).
    *   *PSF Modeling:* ML techniques can potentially be used to model complex, spatially varying PSFs (Section 6.4.1).
    *   *Improving Calibration:* ML might assist in refining photometric or wavelength calibration by learning subtle systematic effects from large datasets.

The successful application of ML in astronomy often requires careful **feature engineering** (selecting or creating meaningful inputs for the algorithms), rigorous **validation** using appropriate metrics and independent test sets, and critically, **interpretability** – understanding *why* an ML model makes a particular prediction or classification, which remains an active area of research, especially for complex Deep Learning models.

**10.5 Introduction to Deep Learning in Astronomy**

**Deep Learning (DL)** is a subfield of Machine Learning based on **Artificial Neural Networks (ANNs)** with multiple layers (hence "deep"). These networks are inspired by the structure and function of the human brain, consisting of interconnected nodes or "neurons" organized in layers. Each connection has a weight, and each neuron applies an activation function to the weighted sum of its inputs. By adjusting these weights during training (typically using backpropagation and gradient descent optimization), deep neural networks can learn extremely complex, hierarchical representations and non-linear relationships directly from raw data, often bypassing the need for extensive manual feature engineering required by traditional ML algorithms (LeCun et al., 2015; Goodfellow et al., 2016). The ability to learn intricate patterns from high-dimensional data like images and sequences has led to transformative successes for DL in fields like computer vision and natural language processing, and its adoption in astronomy is rapidly growing (Fluke & Jacobs, 2020; Ntampaka et al., 2019; Villaescusa-Navarro et al., 2023).

Key DL architectures relevant to astronomy include:
*   **Multi-Layer Perceptrons (MLPs) / Fully Connected Networks (FCNs):** The simplest form of deep network, where neurons in one layer are connected to all neurons in the next layer. Effective for tasks involving tabular data or pre-extracted features (e.g., photo-z estimation from magnitudes, stellar parameter regression from spectral indices).
*   **Convolutional Neural Networks (CNNs):** Specifically designed for processing grid-like data, such as images. CNNs use layers containing learnable filters (kernels) that slide (convolve) across the input image, detecting spatial hierarchies of features – edges and textures in early layers, more complex motifs and object parts in deeper layers. Key components include convolutional layers, pooling layers (for down-sampling and spatial invariance), and activation functions (like ReLU). CNNs excel at image-based tasks:
    *   Galaxy morphology classification directly from images (e.g., Cheng et al., 2021).
    *   Identifying strong gravitational lenses in survey images (e.g., Jacobs et al., 2019).
    *   Detecting cosmic rays or image artifacts (Section 10.4).
    *   Classifying transients based on difference image cutouts.
    *   Potentially estimating parameters directly from images (e.g., photo-z from multi-band image cutouts).
*   **Recurrent Neural Networks (RNNs):** Designed to process sequential data, where the order matters, such as time series or sequences of spectra. RNNs have internal memory loops that allow information from previous steps in the sequence to influence the processing of current steps. Variants like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) use gating mechanisms to better handle long-range dependencies. Applications include:
    *   Time series classification (e.g., variable star types, transient events based on light curves) (e.g., Naul et al., 2018).
    *   Time series forecasting or imputation (predicting future points or filling gaps).
    *   Analysis of sequential spectral data.
*   **Autoencoders (AEs):** Unsupervised networks trained to reconstruct their input data. They consist of an encoder (compressing the input into a lower-dimensional latent representation) and a decoder (reconstructing the original data from the latent representation). Useful for:
    *   Dimensionality reduction and feature extraction (the latent space captures salient features).
    *   Anomaly detection (inputs that are poorly reconstructed often correspond to outliers).
    *   Data compression.
    *   Generative modeling (Variational Autoencoders - VAEs).
*   **Generative Adversarial Networks (GANs):** Consist of two networks, a generator and a discriminator, trained adversarially. The generator tries to create realistic synthetic data (e.g., images of galaxies), while the discriminator tries to distinguish between real and generated data. Used for generating mock observations, augmenting training sets, or potentially image translation/enhancement tasks.

**Frameworks (`TensorFlow`, `PyTorch`):** Implementing and training deep learning models typically relies on specialized software frameworks that provide automatic differentiation (for gradient calculation), GPU acceleration (crucial for performance, see Chapter 11), pre-built layers, optimizers, and model-building APIs. The two dominant frameworks are:
*   **TensorFlow:** Developed by Google, often used with the high-level Keras API (`tf.keras`) for ease of use (Abadi et al., 2016).
*   **PyTorch:** Developed by Facebook's AI Research lab, known for its Pythonic interface and dynamic computation graphs (Paszke et al., 2019).
Both frameworks offer extensive ecosystems and are widely used in astronomical DL applications.

**Challenges:** While powerful, DL models often require very large labeled training datasets, significant computational resources (especially GPUs) for training, and careful hyperparameter tuning. Perhaps the biggest challenge is **interpretability** – understanding *how* a deep network arrives at its decision, which is crucial for building trust and verifying scientific validity. Techniques like attention maps, saliency maps, and layer-wise relevance propagation aim to provide insights into DL model behavior but remain an active area of research. Despite these challenges, the ability of DL to learn directly from complex, high-dimensional raw data positions it as a key technology for future discoveries in the era of large astronomical surveys.

**10.6 Examples in Practice (Python): Big Data & ML Applications**

The following examples illustrate the application of various Machine Learning techniques to tackle specific analysis tasks across different astronomical domains, reflecting the types of problems encountered when dealing with large datasets. These examples utilize common Python libraries like `scikit-learn` for traditional ML, `astroquery` for data retrieval, and conceptually `tensorflow` or `pytorch` for deep learning illustrations. They aim to provide practical starting points for applying ML to real astronomical data analysis workflows.

**10.6.1 Solar: Flare Prediction using Random Forest**
Predicting solar flares is a key challenge in space weather forecasting, with significant implications for satellite operations and communication systems. Machine Learning, particularly supervised classification, can be applied to learn patterns in pre-flare solar active region properties (e.g., derived from SDO/HMI magnetograms or AIA images) that are predictive of subsequent flare occurrence (e.g., Bobra & Couvidat, 2015; Nishizuka et al., 2021). This example conceptually demonstrates training a Random Forest classifier using `scikit-learn`. It simulates loading a dataset containing features extracted from active regions (e.g., magnetic complexity parameters, size, previous flare history) and corresponding labels indicating whether a significant flare occurred within a subsequent time window (e.g., 24 hours). The Random Forest model is trained on this labeled data to predict the likelihood of future flares based on input features.

```python
import numpy as np
import pandas as pd
# Requires scikit-learn: pip install scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    sklearn_available = True
except ImportError:
    print("scikit-learn not found, skipping Solar flare prediction example.")
    sklearn_available = False
import matplotlib.pyplot as plt
import seaborn as sns # For confusion matrix plotting

# --- Simulate Flare Prediction Dataset ---
# In reality, this data would come from processing sequences of SDO images/magnetograms
# and correlating extracted features with GOES flare catalogs.
if sklearn_available:
    n_samples = 1000
    # Simulate Features (X): magnetic complexity, size, previous flares etc. (highly simplified)
    # Create more features for a realistic scenario
    features = pd.DataFrame({
        'magnetic_complexity': np.random.rand(n_samples) * 10, # Higher value -> more complex
        'active_region_area': np.random.lognormal(mean=5, sigma=1, size=n_samples),
        'previous_flare_intensity': np.random.gamma(shape=1, scale=1e-6, size=n_samples) # Most are low intensity
    })
    # Simulate Target (y): Flare occurrence (binary: 0 = No Flare, 1 = Flare)
    # Make flare occurrence more likely for complex/large regions (simplified logic)
    flare_probability = 0.05 + 0.5 * (features['magnetic_complexity'] / 10.0) * (np.log10(features['active_region_area']) / 7.0)
    flare_probability = np.clip(flare_probability, 0.01, 0.95) # Limit probability
    labels = np.random.binomial(n=1, p=flare_probability, size=n_samples)
    target = pd.Series(labels, name='Flare_Occurred')
    print("Simulated flare feature dataset created.")
    print(f"  Number of non-flare samples: {np.sum(target == 0)}")
    print(f"  Number of flare samples: {np.sum(target == 1)}")

    # --- Prepare Data for Training ---
    X = features # Input features DataFrame
    y = target   # Target labels Series

    # Split data into training and testing sets
    # stratify=y ensures similar class proportions in train/test sets (important for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nData split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")

    # --- Train Random Forest Classifier ---
    print("Training Random Forest Classifier...")
    # Initialize the classifier
    # n_estimators: number of trees in the forest
    # class_weight='balanced': Adjusts weights inversely proportional to class frequencies (good for imbalanced data)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the model using the training data
    rf_classifier.fit(X_train, y_train)
    print("Training complete.")

    # --- Evaluate Model Performance ---
    print("\nEvaluating model performance on the test set...")
    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1] # Probabilities for class 1 (Flare)

    # Print classification report (precision, recall, F1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Plot confusion matrix heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flare', 'Flare'], yticklabels=['No Flare', 'Flare'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # --- Feature Importance ---
    print("\nFeature Importances:")
    importances = rf_classifier.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)


else:
    print("Skipping Solar flare prediction example: scikit-learn unavailable.")

```

This Python script conceptually demonstrates the application of a supervised machine learning classifier, specifically a Random Forest, to the problem of solar flare prediction using `scikit-learn`. It begins by simulating a dataset where each row represents an observation of a solar active region, characterized by numerical features (e.g., measures of magnetic complexity, area – these would be derived from SDO/HMI data in a real application). A binary target label indicates whether a significant flare subsequently occurred. The data is split into training and testing sets, ensuring the proportion of flaring vs. non-flaring examples is maintained (stratification). A `RandomForestClassifier` model is initialized, using parameters like `n_estimators` (number of decision trees) and `class_weight='balanced'` (important for handling the typically imbalanced nature of flare data where non-flaring instances vastly outnumber flaring ones). The model is trained using the `.fit()` method on the training features and labels. Finally, the trained model's predictive performance is evaluated on the unseen test set using standard classification metrics (`classification_report`, `confusion_matrix`, `roc_auc_score`), providing insights into its ability to distinguish between flaring and non-flaring conditions. The script also extracts feature importances, indicating which input parameters the model found most predictive.

**10.6.2 Planetary: Asteroid Taxonomic Classification (Clustering)**
Asteroids exhibit diverse surface compositions, reflected in their photometric colors and spectral properties. Grouping asteroids into taxonomic classes based on these observed properties helps understand their origins, evolution, and relationship to meteorite samples. Unsupervised clustering algorithms can be used to identify natural groupings within asteroid datasets without predefined labels. This example simulates applying the K-Means clustering algorithm from `scikit-learn` to a dataset of asteroid colors (e.g., derived from multi-band photometry like SDSS u-g, g-r, r-i) to partition them into a predefined number of clusters, potentially corresponding to major taxonomic groups (e.g., S-type, C-type, X-type). Feature scaling is applied beforehand, as K-Means is sensitive to feature ranges.

```python
import numpy as np
import pandas as pd
# Requires scikit-learn: pip install scikit-learn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler # For feature scaling
    from sklearn.metrics import silhouette_score # To evaluate clustering quality
    sklearn_available = True
except ImportError:
    print("scikit-learn not found, skipping Asteroid clustering example.")
    sklearn_available = False
import matplotlib.pyplot as plt
import seaborn as sns # For plotting

# --- Simulate Asteroid Color Dataset ---
if sklearn_available:
    n_asteroids = 500
    # Simulate features (e.g., SDSS colors u-g, g-r, r-i) for different groups
    # Group 1: S-type like (redder slope)
    g1_ug = np.random.normal(1.4, 0.1, int(n_asteroids * 0.4))
    g1_gr = np.random.normal(0.6, 0.05, int(n_asteroids * 0.4))
    g1_ri = np.random.normal(0.2, 0.05, int(n_asteroids * 0.4))
    # Group 2: C-type like (flatter slope)
    g2_ug = np.random.normal(1.2, 0.1, int(n_asteroids * 0.4))
    g2_gr = np.random.normal(0.4, 0.05, int(n_asteroids * 0.4))
    g2_ri = np.random.normal(0.1, 0.05, int(n_asteroids * 0.4))
    # Group 3: X-type like (intermediate/varied)
    g3_ug = np.random.normal(1.3, 0.15, int(n_asteroids * 0.2))
    g3_gr = np.random.normal(0.5, 0.1, int(n_asteroids * 0.2))
    g3_ri = np.random.normal(0.15, 0.08, int(n_asteroids * 0.2))
    # Combine into a DataFrame
    features = pd.DataFrame({
        'u-g': np.concatenate([g1_ug, g2_ug, g3_ug]),
        'g-r': np.concatenate([g1_gr, g2_gr, g3_gr]),
        'r-i': np.concatenate([g1_ri, g2_ri, g3_ri])
    })
    # Shuffle the data
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Simulated asteroid color dataset created.")

    # --- Prepare Data for Clustering ---
    # K-Means is sensitive to feature scales, so standardize the data
    print("Standardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # --- Apply K-Means Clustering ---
    # Choose the number of clusters (k)
    # This often requires prior knowledge or experimentation (e.g., using elbow method or silhouette score)
    n_clusters = 3 # Assume we expect 3 main taxonomic groups
    print(f"Applying K-Means clustering with k={n_clusters}...")

    # Initialize and fit the K-Means model
    # n_init='auto' runs the algorithm multiple times with different centroids seeds
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)

    # Get the cluster labels assigned to each asteroid
    cluster_labels = kmeans.labels_
    # Get the coordinates of the cluster centers (in scaled space)
    cluster_centers_scaled = kmeans.cluster_centers_
    # Inverse transform centers to original feature space for interpretation
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

    print("K-Means clustering complete.")
    print(f"Cluster center coordinates (in original color space):")
    print(pd.DataFrame(cluster_centers, columns=features.columns))

    # Add cluster labels to the original DataFrame
    features['Cluster'] = cluster_labels

    # --- Evaluate Clustering Quality (Optional) ---
    # Silhouette score measures how similar an object is to its own cluster
    # compared to other clusters (ranges from -1 to 1, higher is better).
    try:
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print(f"\nClustering Silhouette Score: {silhouette_avg:.4f}")
    except Exception as sil_err:
         print(f"\nCould not calculate silhouette score: {sil_err}") # May fail if only 1 cluster found etc.

    # --- Visualize the Clusters ---
    # Plot pairs of colors, color-coded by cluster label
    print("Visualizing clusters...")
    plt.figure(figsize=(12, 5))
    # Plot u-g vs g-r
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=features, x='g-r', y='u-g', hue='Cluster', palette='viridis', s=20, alpha=0.7)
    # Plot cluster centers
    plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='X', s=100, c='red', label='Cluster Centers')
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.xlabel('g - r color')
    plt.ylabel('u - g color')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    # Plot g-r vs r-i
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=features, x='r-i', y='g-r', hue='Cluster', palette='viridis', s=20, alpha=0.7, legend=False)
    plt.scatter(cluster_centers[:, 2], cluster_centers[:, 1], marker='X', s=100, c='red')
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.xlabel('r - i color')
    plt.ylabel('g - r color')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

else:
    print("Skipping Asteroid clustering example: scikit-learn unavailable.")

```

This Python script demonstrates the use of unsupervised clustering, specifically the K-Means algorithm from `scikit-learn`, for asteroid taxonomic classification based on photometric colors. It simulates a dataset containing typical asteroid colors (e.g., u-g, g-r, r-i) designed to loosely represent different taxonomic groups (S-type, C-type, X-type). Because K-Means relies on distance calculations, the script first applies feature scaling using `StandardScaler` to give each color index equal importance. The `KMeans` algorithm is then initialized with a predefined number of clusters ($k=3$ in this example) and fitted to the scaled color data. The algorithm assigns each asteroid to one of the $k$ clusters based on proximity to the cluster centroids in the multi-dimensional color space. The script extracts the resulting cluster labels for each asteroid and calculates the cluster centroids (transformed back to the original color space for easier interpretation). Finally, it visualizes the results by plotting pairs of asteroid colors, color-coding the points by their assigned cluster label, demonstrating how K-Means can automatically group objects with similar spectral/color properties, potentially revealing underlying taxonomic classes within the dataset. The silhouette score provides a quantitative measure of how well-separated the resulting clusters are.

**10.6.3 Stellar: Dimensionality Reduction of Gaia Photometry (PCA)**
The Gaia mission provides multi-band photometry (G, BP, RP) and low-resolution spectra for over a billion stars, creating a high-dimensional dataset. Dimensionality reduction techniques can be valuable for visualizing the overall structure of this dataset, identifying distinct stellar populations or sequences, or extracting compact feature representations for subsequent analysis. Principal Component Analysis (PCA) is a common linear technique that finds orthogonal axes (principal components) capturing the maximum variance in the data. This example simulates applying PCA using `scikit-learn` to a dataset of Gaia G, BP, and RP magnitudes (and derived colors) to reduce its dimensionality, perhaps for visualizing stellar sequences in a 2D plane defined by the first two principal components.

```python
import numpy as np
import pandas as pd
# Requires scikit-learn: pip install scikit-learn
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    print("scikit-learn not found, skipping Stellar PCA example.")
    sklearn_available = False
import matplotlib.pyplot as plt
# Requires astroquery for real data, simulated here
# from astroquery.gaia import Gaia

# --- Simulate Gaia Photometry Dataset ---
if sklearn_available:
    n_stars = 2000
    # Simulate features: G mag, BP-RP color, G-RP color (example features)
    # Create a mix representing main sequence + some red giants
    # Main Sequence (bluer, follows track)
    n_ms = int(n_stars * 0.8)
    ms_bp_rp = np.random.uniform(-0.2, 2.0, n_ms) # BP-RP color range
    # Approximate main sequence G vs BP-RP relation (simplified)
    ms_g_mag = 5.0 + 2.5 * ms_bp_rp + np.random.normal(0, 0.3, n_ms)
    ms_g_rp = 0.1 + 0.8 * ms_bp_rp + np.random.normal(0, 0.1, n_ms) # G-RP color
    # Red Giants (redder, brighter for given color)
    n_rg = n_stars - n_ms
    rg_bp_rp = np.random.uniform(0.8, 2.5, n_rg)
    rg_g_mag = 3.0 + 1.5 * rg_bp_rp + np.random.normal(0, 0.5, n_rg)
    rg_g_rp = 0.5 + 0.6 * rg_bp_rp + np.random.normal(0, 0.15, n_rg)
    # Combine
    features = pd.DataFrame({
        'Gmag': np.concatenate([ms_g_mag, rg_g_mag]),
        'BP_RP': np.concatenate([ms_bp_rp, rg_bp_rp]),
        'G_RP': np.concatenate([ms_g_rp, rg_g_rp])
    })
    # Add simulated noise/scatter to colors
    features['BP_RP'] += np.random.normal(0, 0.05, n_stars)
    features['G_RP'] += np.random.normal(0, 0.05, n_stars)
    # Shuffle
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Simulated Gaia-like photometry dataset created.")

    # --- Prepare Data for PCA ---
    # PCA is sensitive to scales, standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X = features[['Gmag', 'BP_RP', 'G_RP']] # Select features for PCA
    X_scaled = scaler.fit_transform(X)

    # --- Apply Principal Component Analysis (PCA) ---
    # Choose number of components to keep (e.g., 2 for visualization)
    n_components = 2
    print(f"Applying PCA to reduce to {n_components} components...")

    # Initialize and fit PCA model
    pca = PCA(n_components=n_components)
    # Fit PCA on scaled data and transform data to principal component space
    X_pca = pca.fit_transform(X_scaled)

    print("PCA transformation complete.")
    # Explained variance ratio shows how much variance each component captures
    print(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")

    # The result X_pca is an array with shape (n_stars, n_components)
    # Add PCA components back to the DataFrame for plotting/analysis
    features['PC1'] = X_pca[:, 0]
    features['PC2'] = X_pca[:, 1]

    # --- Visualize in PCA Space ---
    print("Visualizing data in Principal Component space...")
    plt.figure(figsize=(8, 6))
    # Scatter plot using the first two principal components
    # Color points by G magnitude (example) or BP-RP color
    plt.scatter(features['PC1'], features['PC2'], c=features['BP_RP'], cmap='viridis', s=5, alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Gaia Photometry PCA (PC1 vs PC2)")
    plt.colorbar(label='BP - RP Color')
    plt.grid(True, alpha=0.3)
    # Add arrows showing original feature axes projection onto PCA space (optional)
    # ... code to plot feature vectors pca.components_ ...
    plt.show()

else:
    print("Skipping Stellar PCA example: scikit-learn unavailable.")

```

This Python script demonstrates the application of Principal Component Analysis (PCA), a dimensionality reduction technique, to simulated Gaia multi-band photometric data using `scikit-learn`. It creates a dataset mimicking Gaia observations (G magnitude, BP-RP color, G-RP color) for a mix of stellar populations (main sequence and red giants). As PCA is sensitive to the relative scales of input features, the data is first standardized using `StandardScaler`. The `PCA` algorithm is then initialized, specifying the desired number of output dimensions (principal components, here $n=2$ for visualization). The `.fit_transform()` method computes the principal components that capture the maximum variance in the standardized data and transforms the data points into this new lower-dimensional space (`X_pca`). The script prints the fraction of the total variance explained by each principal component, indicating how much information is retained. Finally, it visualizes the dataset by plotting the first principal component (PC1) against the second (PC2), color-coding the points by an original feature (like BP-RP color). This 2D representation can reveal underlying structures, like the main sequence and red giant branch, that were present in the original higher-dimensional photometric space.

**10.6.4 Exoplanetary: TESS Light Curve Classification (Conceptual NN)**
Classifying the nature of signals detected in TESS light curves – distinguishing genuine exoplanet transits from astrophysical false positives (like eclipsing binaries) or instrumental systematics – is a critical task. Deep Learning models, particularly Convolutional Neural Networks (CNNs), have shown promise for this task by learning spatial-temporal patterns directly from pixel-level data (Target Pixel Files - TPFs) or morphological features from processed light curves (Shallue & Vanderburg, 2018; Osborn et al., 2020; Yu et al., 2019). This example provides a highly conceptual outline using `tensorflow.keras` to define a simple CNN architecture that could be trained to classify TESS light curve "cutouts" (segments centered on potential events) as either 'Transit' or 'Not Transit'. It focuses on the model definition aspect, assuming suitably preprocessed input data (e.g., normalized light curve segments or local/global views from TPFs) would be provided for training.

```python
import numpy as np
# Requires tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tensorflow_available = True
except ImportError:
    print("TensorFlow/Keras not found, skipping Exoplanet NN classification example.")
    tensorflow_available = False
import matplotlib.pyplot as plt

# --- Conceptual Example: CNN for TESS Light Curve Classification ---
# This example focuses ONLY on defining a simple CNN model structure.
# It does NOT include data loading, preprocessing, training, or evaluation,
# which are substantial tasks requiring real TESS data and careful setup.

if tensorflow_available:
    print("Defining a conceptual Convolutional Neural Network (CNN) model for classification...")

    # Assume input data would be preprocessed light curve segments (e.g., fixed length)
    # Example input shape: (length_of_segment, 1) for 1D light curve data
    # Or potentially a 2D representation (e.g., local/global views combined)
    input_shape_example = (201, 1) # Example: segment length 201 points, 1 channel

    # --- Define the CNN Model Architecture using Keras Sequential API ---
    # Simple CNN architecture for demonstration
    model = keras.Sequential(
        [
            # Input Layer (implicitly defined by first layer's input_shape)
            keras.Input(shape=input_shape_example),

            # Convolutional Layer 1: Learns local patterns
            # filters: number of output filters (feature maps)
            # kernel_size: length of the 1D convolution window
            # activation: activation function (e.g., 'relu')
            layers.Conv1D(filters=16, kernel_size=5, activation="relu", padding='same'),
            # Pooling Layer 1: Downsamples, provides robustness to small shifts
            layers.MaxPooling1D(pool_size=2),

            # Convolutional Layer 2: Learns higher-level patterns from Layer 1 features
            layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding='same'),
            # Pooling Layer 2
            layers.MaxPooling1D(pool_size=2),

            # Flatten Layer: Converts 2D feature maps to a 1D vector for Dense layers
            layers.Flatten(),

            # Dense (Fully Connected) Layer: Standard neural network layer
            layers.Dense(units=64, activation="relu"),
            # Dropout Layer: Regularization technique to prevent overfitting
            layers.Dropout(0.5),

            # Output Layer: Single neuron with sigmoid activation for binary classification
            # Sigmoid outputs a probability (0 to 1) for the positive class ('Transit')
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    # --- Print Model Summary ---
    print("\nConceptual CNN Model Summary:")
    model.summary()

    # --- Compile the Model (Specify optimizer, loss function, metrics) ---
    # This step is needed before training.
    # Optimizer: Algorithm to update network weights (e.g., 'adam')
    # Loss function: Measures difference between prediction and true label (e.g., 'binary_crossentropy')
    # Metrics: Performance measure monitored during training (e.g., 'accuracy')
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("\n(Model compilation step would follow)")

    # --- Training (Conceptual) ---
    # This step would require loading preprocessed training data (X_train, y_train)
    # and validation data (X_val, y_val).
    # history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))
    print("(Model training step with actual data would follow)")

    # --- Prediction (Conceptual) ---
    # Once trained, the model can predict probabilities for new light curve segments:
    # probabilities = model.predict(new_light_curve_segments)
    # predicted_classes = (probabilities > 0.5).astype("int32") # Threshold probability
    print("(Model prediction step on new data would follow)")

else:
    print("Skipping Exoplanet NN classification example: TensorFlow/Keras unavailable.")

```

This Python script provides a conceptual blueprint for constructing a simple Convolutional Neural Network (CNN) using the `tensorflow.keras` API, aimed at classifying segments of TESS light curves as either containing an exoplanet transit or not. It focuses solely on defining the model architecture, assuming that appropriate input data (e.g., fixed-length, normalized light curve segments centered on potential events) would be prepared separately. The defined `keras.Sequential` model consists of typical CNN layers: `Conv1D` layers with ReLU activation learn hierarchical temporal features from the 1D light curve data, `MaxPooling1D` layers provide downsampling and some invariance to the exact position of features, `Flatten` converts the learned features into a 1D vector, and standard `Dense` (fully connected) layers perform the final classification, with a `Dropout` layer added for regularization to prevent overfitting. The output layer uses a sigmoid activation function to produce a probability score between 0 and 1, indicating the likelihood that the input segment represents a transit. While the script defines the architecture and prints its summary, it explicitly notes that the crucial steps of compiling the model (defining loss function and optimizer) and training it on actual, preprocessed TESS data are complex prerequisite tasks not executed here.

**10.6.5 Galactic: Large ADQL Query for Stellar Populations**
Studying the structure and formation history of the Milky Way often requires analyzing the properties (positions, kinematics, photometry, abundances) of vast numbers of stars drawn from large survey catalogs like Gaia, APOGEE, GALAH, etc. Accessing and filtering these large catalogs efficiently requires programmatic queries using ADQL against database services like the ESA Gaia Archive or NOIRLab's Astro Data Lab. This example demonstrates constructing and executing a moderately complex ADQL query using `astroquery.gaia` to select a specific stellar population – for instance, potential Red Giant Branch (RGB) stars within a certain volume around the Sun – based on cuts in parallax, proper motion, magnitude, and color space. This showcases how large survey databases are queried to assemble datasets for Galactic studies.

```python
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
# Requires astroquery: pip install astroquery
try:
    from astroquery.gaia import Gaia
    astroquery_available = True
except ImportError:
    print("astroquery not found, skipping Galactic ADQL query example.")
    astroquery_available = False

# --- Define Query Parameters for Stellar Population Selection ---
# Example: Select potential Red Giant Branch stars within ~2 kpc
# Criteria:
# - Parallax > 0.5 mas (distance < 2 kpc) with good S/N
# - Specific color-magnitude range typical for RGB stars
# - Low tangential velocity component (optional, e.g., for disk stars)

parallax_min = 0.5 # mas
parallax_snr_min = 5.0
g_mag_max = 17.0
bp_rp_min = 0.8 # Select redder stars
# Define a region in Color-Magnitude Diagram (CMD) for RGB (simplified polygon)
# Points defining a box/polygon in (BP_RP, Abs_G_Mag) space
# Absolute G Mag = Gmag + 5 * log10(parallax_mas) - 10
# Example CMD box: BP_RP > 0.8, Abs_G between -2 and +4
abs_g_mag_min = -2.0
abs_g_mag_max = 4.0

if astroquery_available:
    print("Constructing ADQL query for potential RGB stars near the Sun...")
    # Construct the ADQL query string with multiple constraints
    # Use Gaia DR3 table: gaiadr3.gaia_source
    # Apply cuts on parallax, parallax S/N, apparent G magnitude, and CMD position
    adql_query = f"""
    SELECT TOP 10000 -- Limit query size for demonstration
      source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
      (phot_bp_mean_mag - phot_rp_mean_mag) AS bp_rp,
      (phot_g_mean_mag + 5 * LOG10(parallax) - 10) AS abs_g_mag
    FROM
      gaiadr3.gaia_source
    WHERE
      parallax IS NOT NULL AND parallax > {parallax_min}
      AND parallax_over_error IS NOT NULL AND parallax_over_error > {parallax_snr_min}
      AND phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag < {g_mag_max}
      AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL
      AND (phot_bp_mean_mag - phot_rp_mean_mag) > {bp_rp_min}
      AND (phot_g_mean_mag + 5 * LOG10(parallax) - 10) > {abs_g_mag_min}
      AND (phot_g_mean_mag + 5 * LOG10(parallax) - 10) < {abs_g_mag_max}
      -- Optional: Add constraints on proper motion if desired
      -- AND SQRT(pmra*pmra + pmdec*pmdec) < some_value
    """
    # Note: Calculating absolute magnitude directly in WHERE clause can be slow.
    # Sometimes better to retrieve apparent mags/parallax and calculate later.

    print("Submitting ADQL query to Gaia Archive...")
    try:
        # Launch the query (use synchronous for smaller TOP N queries for simplicity)
        # job = Gaia.launch_job_async(adql_query)
        # results_table = job.get_results()
        results_table = Gaia.launch_job(adql_query).get_results()

        print(f"\nQuery successful! Retrieved {len(results_table)} potential RGB candidates (limited by TOP 10000).")
        print("First 5 rows of the results table:")
        # Format columns for display
        results_table['abs_g_mag'].info.format = '.3f'
        results_table['bp_rp'].info.format = '.3f'
        results_table['parallax'].info.format = '.3f'
        print(results_table['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag', 'bp_rp', 'abs_g_mag'][:5])

        # This results_table now contains a filtered sample of stars likely belonging
        # to the Red Giant Branch population within approx 2 kpc, suitable for
        # further kinematic or chemical analysis, or as input for ML models studying
        # Galactic structure or stellar evolution.

    except Exception as e:
        print(f"An error occurred during the Gaia ADQL query: {e}")
else:
    print("Skipping Galactic ADQL query example: astroquery unavailable.")

```

This Python script demonstrates how to construct and execute a relatively complex Astronomical Data Query Language (ADQL) query against the massive Gaia DR3 catalog using `astroquery.gaia` to select a specific stellar population for Galactic studies. The goal is to isolate potential Red Giant Branch (RGB) stars located relatively nearby (within ~2 kpc). The ADQL query incorporates multiple constraints within the `WHERE` clause: it selects stars with significant positive parallax measurements (indicating proximity and reliable distance estimates), applies limits on apparent G magnitude, and uses calculated color (BP-RP) and absolute G magnitude (derived from apparent G and parallax) to select stars falling within a specific region of the Color-Magnitude Diagram characteristic of RGB stars. The query retrieves key parameters (ID, position, parallax, proper motions, magnitudes, calculated color/absolute magnitude) for the selected stars, limited here by `TOP 10000` for demonstration purposes. The resulting `results_table` provides a targeted dataset extracted from the vast Gaia catalog, ready for detailed analysis of this specific Galactic stellar population, potentially serving as input for Machine Learning models.

**10.6.6 Extragalactic: Photometric Redshift Estimation (Regression)**
Estimating the redshifts of large numbers of galaxies from multi-band photometric data alone (photometric redshifts or photo-z's) is essential for large imaging surveys where obtaining spectra for every object is infeasible. Machine Learning regression algorithms are widely used for this task, learning the complex mapping between galaxy colors (and potentially other features like morphology or brightness) and spectroscopic redshift ($z_{spec}$) using a smaller training set of galaxies for which both photometry and $z_{spec}$ are available. This example conceptually demonstrates training a Random Forest Regressor using `scikit-learn` to predict photo-z based on simulated galaxy magnitudes in several bands. It covers data preparation (feature selection, splitting), model training, and evaluation using standard regression metrics.

```python
import numpy as np
import pandas as pd
# Requires scikit-learn: pip install scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler # Scaling often helpful for regression too
    sklearn_available = True
except ImportError:
    print("scikit-learn not found, skipping Photo-z regression example.")
    sklearn_available = False
import matplotlib.pyplot as plt

# --- Simulate Galaxy Photometry + Spectroscopic Redshift Dataset ---
# In reality, this comes from matching a photometric catalog to a spectroscopic survey
if sklearn_available:
    n_galaxies = 2000
    # Simulate true spectroscopic redshifts
    true_zspec = np.random.uniform(0.05, 1.5, n_galaxies)
    # Simulate magnitudes in several bands (e.g., u, g, r, i, z) based on redshift
    # Simplified model: Colors get redder with redshift (passive evolution proxy)
    # Add scatter and noise
    mag_u = 18.0 + 3.0 * true_zspec + 2.5 * np.log10(true_zspec / 0.5 + 1) + np.random.normal(0, 0.5, n_galaxies)
    mag_g = mag_u - (1.0 + 0.5*true_zspec + np.random.normal(0, 0.2, n_galaxies))
    mag_r = mag_g - (0.5 + 0.3*true_zspec + np.random.normal(0, 0.15, n_galaxies))
    mag_i = mag_r - (0.3 + 0.1*true_zspec + np.random.normal(0, 0.1, n_galaxies))
    mag_z = mag_i - (0.2 + 0.05*true_zspec + np.random.normal(0, 0.1, n_galaxies))
    # Create DataFrame
    galaxy_data = pd.DataFrame({
        'u_mag': mag_u, 'g_mag': mag_g, 'r_mag': mag_r, 'i_mag': mag_i, 'z_mag': mag_z,
        'z_spec': true_zspec # Target variable
    })
    # Add measurement errors (uncorrelated here for simplicity)
    for band in ['u', 'g', 'r', 'i', 'z']:
        galaxy_data[f'{band}_err'] = np.random.uniform(0.02, 0.15, n_galaxies) # Simplified errors
    print("Simulated galaxy photometry and redshift dataset created.")

    # --- Prepare Data for Regression ---
    # Define features (magnitudes or colors) and target (z_spec)
    # Using magnitudes directly here, colors (e.g., u-g, g-r) often work better
    feature_names = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']
    X = galaxy_data[feature_names]
    y = galaxy_data['z_spec'] # Target redshift

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nData split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")

    # Scale features (optional but often good practice for regressors)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use scaler fitted on training data

    # --- Train Random Forest Regressor ---
    print("Training Random Forest Regressor for Photo-z...")
    # Initialize the regressor
    # n_estimators: number of trees
    # max_depth: controls complexity of individual trees
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1) # Use all CPU cores

    # Train the model
    rf_regressor.fit(X_train_scaled, y_train)
    print("Training complete.")

    # --- Evaluate Model Performance ---
    print("\nEvaluating photo-z performance on the test set...")
    # Make predictions on the test set
    y_pred_photoz = rf_regressor.predict(X_test_scaled)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred_photoz)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_photoz)
    # Also common: normalized median absolute deviation (NMAD) of dz/(1+z)
    dz = y_pred_photoz - y_test
    nmad = 1.4826 * np.median(np.abs(dz / (1 + y_test)))
    # Outlier fraction (|dz|/(1+z) > 0.15)
    outlier_fraction = np.mean(np.abs(dz / (1 + y_test)) > 0.15)

    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2) Score: {r2:.4f}")
    print(f"  Normalized MAD (NMAD): {nmad:.4f}")
    print(f"  Outlier Fraction (|dz|/(1+z) > 0.15): {outlier_fraction:.4f}")

    # --- Plot Results ---
    print("Plotting true redshift vs predicted photo-z...")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_photoz, s=5, alpha=0.5, label=f'RMSE={rmse:.3f}\nNMAD={nmad:.3f}')
    plt.plot([0, 1.6], [0, 1.6], 'r--', label='Ideal (y=x)') # Diagonal line
    plt.xlabel("True Spectroscopic Redshift (z_spec)")
    plt.ylabel("Predicted Photometric Redshift (photo-z)")
    plt.title("Photometric Redshift Performance (Random Forest)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)
    plt.show()

else:
    print("Skipping Photo-z regression example: scikit-learn unavailable.")

```

This Python script illustrates the process of estimating photometric redshifts (photo-z's) for galaxies using a supervised machine learning regression technique, specifically a Random Forest Regressor from `scikit-learn`. It begins by simulating a dataset containing multi-band magnitudes (u, g, r, i, z) for a sample of galaxies, along with their corresponding "true" spectroscopic redshifts ($z_{spec}$), which serve as the target variable for training. The script selects the magnitudes as input features ($X$) and $z_{spec}$ as the target ($y$). The data is split into training and testing sets, and the features are standardized using `StandardScaler`. A `RandomForestRegressor` model is then initialized and trained on the scaled training data using the `.fit()` method, learning the relationship between galaxy magnitudes/colors and redshift. After training, the model's performance is evaluated on the held-out test set by predicting photo-z's (`y_pred_photoz`) and comparing them to the true $z_{spec}$ values using standard regression metrics like Root Mean Squared Error (RMSE), R-squared ($R^2$), and metrics commonly used in photo-z studies like the Normalized Median Absolute Deviation (NMAD) and outlier fraction. A scatter plot comparing $z_{spec}$ and predicted photo-z visually assesses the accuracy and scatter of the predictions.

**10.6.7 Cosmology: CNN Application for Strong Lens Identification**
Identifying strong gravitational lenses – where the gravity of a massive foreground galaxy or cluster bends the light from a background source, creating multiple images or arcs – is crucial for cosmology (measuring Hubble constant, probing dark matter substructure) and galaxy evolution studies. Finding these rare systems in vast imaging surveys is challenging. Deep Learning, particularly Convolutional Neural Networks (CNNs), has proven highly effective at classifying image cutouts as either containing a strong lens or not, learning the complex morphological features characteristic of lensing systems directly from image pixels (e.g., Jacobs et al., 2019; Petrillo et al., 2019; Davies et al., 2021). This example provides a conceptual outline of defining and potentially training a CNN model using `tensorflow.keras` for strong lens identification, similar to the TESS classification example but applied to 2D image cutouts.

```python
import numpy as np
# Requires tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tensorflow_available = True
except ImportError:
    print("TensorFlow/Keras not found, skipping Strong Lens CNN example.")
    tensorflow_available = False
import matplotlib.pyplot as plt

# --- Conceptual Example: CNN for Strong Lens Identification ---
# Focuses on model definition. Assumes input image cutouts are preprocessed.

if tensorflow_available:
    print("Defining a conceptual CNN model for Strong Lens classification...")

    # Assume input data: 2D image cutouts (e.g., 64x64 pixels, potentially multiple color channels)
    input_shape_example = (64, 64, 3) # Example: 64x64 pixels, 3 color channels (e.g., g, r, i)

    # --- Define the CNN Model Architecture (Example) ---
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape_example),

            # Block 1
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Block 2
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Block 3
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(units=128, activation="relu"),
            layers.Dropout(0.5),
            # Output Layer (Binary: Lens vs Non-Lens)
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    # --- Print Model Summary ---
    print("\nConceptual Strong Lens CNN Model Summary:")
    model.summary()

    # --- Compile the Model ---
    # (Required before training)
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])
    print("\n(Model compilation step would follow)")

    # --- Training (Conceptual) ---
    # Requires a large labeled dataset of image cutouts: (X_train_images, y_train_labels)
    # y_train_labels: 0 for non-lens, 1 for lens
    # Data augmentation (rotations, flips, shifts) is crucial for training robust image classifiers.
    # history = model.fit(X_train_images, y_train_labels, batch_size=32, epochs=20, validation_split=0.2)
    print("(Model training step with labeled image cutouts would follow)")

    # --- Prediction (Conceptual) ---
    # Apply the trained model to new survey image cutouts:
    # lens_probabilities = model.predict(new_image_cutouts)
    # predicted_classes = (lens_probabilities > threshold).astype("int32") # Use appropriate threshold
    print("(Model prediction on new images would follow)")

else:
    print("Skipping Strong Lens CNN example: TensorFlow/Keras unavailable.")

```

This final Python script outlines the structure of a Convolutional Neural Network (CNN) designed for the automated identification of strong gravitational lenses in large astronomical imaging surveys, using the `tensorflow.keras` framework. It focuses on the model definition, assuming that input data consists of preprocessed 2D image cutouts (e.g., 64x64 pixels with multiple color channels) centered on potential candidates. The example defines a sequential CNN model comprising several convolutional blocks (`Conv2D` layers with ReLU activation followed by `MaxPooling2D` layers). These convolutional layers are designed to automatically learn relevant spatial features indicative of lensing (arcs, multiple images, Einstein rings) at different scales directly from the pixel data. The features extracted by the convolutional blocks are then flattened into a vector and passed through fully connected (`Dense`) layers, including `Dropout` for regularization, before reaching a final output neuron with a sigmoid activation function. This output neuron provides a probability score indicating the likelihood that the input image cutout contains a strong gravitational lens (a binary classification task: lens vs. non-lens). While the script defines the architecture, it emphasizes that the critical steps of compiling the model and training it on a large, labeled dataset of real and simulated lens/non-lens images (often requiring significant computational resources and data augmentation techniques) are necessary subsequent steps for creating a functional lens finder.

---

**References**

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., … Zheng, X. (2016). TensorFlow: Large-scale machine learning on heterogeneous distributed systems. *arXiv preprint arXiv:1603.04467*. https://doi.org/10.48550/arXiv.1603.04467 *(Note: Foundational software paper, pre-2020)*
*   *Summary:* The foundational paper describing the TensorFlow framework. Although pre-2020, TensorFlow (often via Keras) remains a dominant library for implementing Deep Learning models (Section 10.5) in astronomy, including CNNs and ANNs mentioned in examples.

Allen, A., Teuben, P., Paddy, K., Greenfield, P., Droettboom, M., Conseil, S., Ninan, J. P., Tollerud, E., Norman, H., Deil, C., Bray, E., Sipőcz, B., Robitaille, T., Kulumani, S., Barentsen, G., Craig, M., Pascual, S., Perren, G., Lian Lim, P., … Streicher, O. (2022). Astropy: A community Python package for astronomy. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.6514771
*   *Summary:* This Zenodo record archives a version of the Astropy package. Its affiliated packages `astroquery` and `pyvo` provide essential tools for accessing the large astronomical databases (Section 10.2) often required as input for ML algorithms.

Alsing, J., Charnock, T., Feeney, S., & Wandelt, B. (2019). Fast likelihood-free cosmology with density estimation and active learning. *Monthly Notices of the Royal Astronomical Society, 488*(3), 4440–4456. https://doi.org/10.1093/mnras/stz1961 *(Note: Pre-2020, but introduces key SBI/LFI concepts)*
*   *Summary:* This paper (pre-2020) explores likelihood-free inference using density estimation neural networks for cosmological parameter estimation. It represents an advanced ML application relevant to regression/parameter estimation discussed conceptually (Section 10.4).

Asgari, M., Dvornik, A., Heymans, C., Hoekstra, H., Schmidt, S. J., Wright, A. H., Bilicki, M., Cabayol, L., Conselice, C. J., Erwin, P., Hildebrandt, H., Joachimi, B., Kannawadi, A., Kuijken, K., Lin, C.-A., Nakajima, R., Shan, H., Tudorica, A., & Valentijn, E. (2023). KiDS-1000 cosmology: machine learning classification and redshift calibration for weak lensing sources. *Astronomy & Astrophysics, 676*, A31. https://doi.org/10.1051/0004-6361/202245694
*   *Summary:* Details the use of machine learning (Self-Organizing Maps and Gaussian Mixture Models) for star-galaxy classification and photometric redshift calibration within the KiDS weak lensing survey. Illustrates ML applications in large extragalactic surveys (Sections 10.4, 10.6.6).

Bailey, S., Abareshi, B., Abidi, A., Abolfathi, B., Aerts, J., Aguilera-Gomez, C., Ahlen, S., Alam, S., Alexander, D. M., Alfarsy, R., Allen, L., Prieto, C. A., Alves-Oliveira, N., Anand, A., Armengaud, E., Ata, M., Avilés, A., Avon, M., Brooks, D., … Zou, H. (2023). The Data Release 1 of the Dark Energy Spectroscopic Instrument. *The Astrophysical Journal, 960*(1), 75. https://doi.org/10.3847/1538-4357/acff2f
*   *Summary:* Describes the first DESI data release and its processing pipeline, including the `redrock` code for redshift determination, which heavily utilizes template fitting and cross-correlation informed by ML concepts (Section 10.4/7.5). Highlights data scale challenges (Section 10.1).

Cheng, T.-Y., Conselice, C. J., Aragón-Salamanca, A., Aguena, M., Allam, S., Andrade-Oliveira, F., Annis, J., Bacon, D., Bertin, E., Bhargava, S., Brooks, D., Burke, D. L., Carnero Rosell, A., Carrasco Kind, M., Carretero, J., Costanzi, M., da Costa, L. N., De Vicente, J., Desai, S., … Wilkinson, R. D. (2021). Galaxy morphology classification in the Dark Energy Survey Year 3 data with convolutional neural networks. *Monthly Notices of the Royal Astronomical Society, 503*(3), 4376–4391. https://doi.org/10.1093/mnras/stab701
*   *Summary:* Applies Convolutional Neural Networks (CNNs, Section 10.5) to classify galaxy morphologies directly from DES image data. Provides a concrete example of using deep learning for source classification in large surveys (Section 10.4).

Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. *Proceedings of the National Academy of Sciences, 117*(48), 30055–30062. https://doi.org/10.1073/pnas.1912789117
*   *Summary:* This review discusses simulation-based inference (SBI) or likelihood-free inference (LFI), often employing ML/DL techniques. Relevant to advanced cosmological parameter estimation applications mentioned in Section 10.4.

Demleitner, M., Taylor, M., Dowler, P., Major, B., Normand, J., Benson, K., & pylibs Development Team. (2023). pyvo 1.4.1: Fix error in datalink parsing. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.7858974
*   *Summary:* This Zenodo record archives a version of `pyvo`, the Python library providing access to Virtual Observatory (VO) services using standard protocols like TAP (for ADQL queries) and SIA/SSA. Essential for programmatic database access (Section 10.2).

Euclid Collaboration, Euclid Collaboration et al. (2024). Euclid preparation. XXXI. The effect of data processing on photometric redshift estimation for the Euclid survey. *Astronomy & Astrophysics*, *681*, A93. https://doi.org/10.1051/0004-6361/202347891
*   *Summary:* Examines photometric redshift (photo-z) estimation methods and data processing effects specifically for the Euclid survey. Highly relevant to the photo-z application of ML regression discussed in Section 10.4 and Example 10.6.6.

Fluke, C. J., & Jacobs, C. (2020). Surveying the approaches and challenges for deep learning applications in astrophysics. *WIREs Data Mining and Knowledge Discovery, 10*(3), e1357. https://doi.org/10.1002/widm.1357
*   *Summary:* Provides a broad overview of Deep Learning applications in astronomy, covering methods (CNNs, RNNs, etc., Section 10.5), diverse applications (classification, parameter estimation, etc., Section 10.4), and the challenges associated with large datasets (Section 10.1).

Ginsburg, A., Sipőcz, B. M., Brasseur, C. E., Cowperthwaite, P. S., Casey, A. R., Gagliano, A., Hedges, C., Voss, N., Parviainen, H., Ioannidis, P., & Barclay, T. (2019). Astroquery: An Astronomical Web-Querying Package in Python. *The Astronomical Journal, 157*(3), 98. https://doi.org/10.3847/1538-3881/aafc33 *(Note: Foundational software paper, pre-2020)*
*   *Summary:* The main paper describing the `astroquery` package. Although pre-2020, `astroquery` is the primary tool demonstrated in Section 10.2 and used conceptually in examples for programmatically accessing astronomical catalogs and databases from Python.

Hunt, J. A. S., Mistry, M. N., & Price-Whelan, A. M. (2023). GALAH DR4 and Gaia DR3 reveal structure and substructure in the Milky Way’s last massive merger. *Monthly Notices of the Royal Astronomical Society, 525*(3), 4697–4718. https://doi.org/10.1093/mnras/stad2447
*   *Summary:* Uses clustering techniques (related to Section 10.3) on combined Gaia kinematic data and GALAH chemical abundances to identify substructure in the Milky Way halo, illustrating unsupervised learning applications in Galactic archaeology.

Ishida, E. E. O., Vilalta, R., & TARGET Collaboration. (2021). Active learning for discovery in synoptic surveys: A study case on the Zwicky Transient Facility alert stream. *Monthly Notices of the Royal Astronomical Society, 508*(3), 4065–4074. https://doi.org/10.1093/mnras/stab2834
*   *Summary:* Explores active learning strategies for identifying interesting anomalies (Section 10.4) within large transient alert streams like ZTF, aiming to optimize follow-up resources. Demonstrates advanced ML techniques for large data streams.

Jia, S., Zhang, Z., Wang, J., & Bloom, J. S. (2023). Self-Supervised Learning for Astronomical Image Cleaning. *arXiv preprint arXiv:2310.14929*. https://doi.org/10.48550/arXiv.2310.14929
*   *Summary:* Presents a self-supervised deep learning method for detecting various image artifacts, including cosmic rays. Illustrates the application of modern DL techniques to data processing challenges (Section 10.4).

Lochner, M., & Bassett, B. A. (2021). SELF-CARE: Supervised Evaluation of Large Features for Classification and Regression Ensemble. *Astronomy and Computing, 36*, 100482. https://doi.org/10.1016/j.ascom.2021.100482
*   *Summary:* Introduces an ensemble ML method and discusses challenges in applying ML to large astronomical datasets (Section 10.1), particularly regarding feature engineering and model evaluation (Section 10.3).

Nishizuka, N., Sugiura, K., Kubo, Y., Den, M., Watari, S., & Ishii, M. (2021). Deep flare net (DeFN) for multi-wavelength solar flare prediction. *Earth and Space Science, 8*(2), e2020EA001244. https://doi.org/10.1029/2020EA001244
*   *Summary:* Applies Deep Learning (CNNs) to multi-wavelength solar data for flare prediction, providing a recent example of DL tackling the classification task outlined conceptually in Example 10.6.1.

Ntampaka, M., ZuHone, J., Eisenstein, D., Nagai, D., Vikhlinin, A., Hernquist, L., Marinacci, F., Nelson, D., Pillepich, A., Pakmor, R., Springel, V., & Weinberger, R. (2019). Machine Learning in Galaxy Cluster Mergers: Training Sets and Application. *The Astrophysical Journal, 873*(2), 131. https://doi.org/10.3847/1538-4357/ab0761 *(Note: Pre-2020, relevant ML application)*
*   *Summary:* Although pre-2020, this paper demonstrates using ML (Random Forests) to classify merging states of galaxy clusters in simulations, an example of ML applied to simulation data analysis related to Sections 10.3 and 10.4.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32*. https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html *(Note: Foundational software paper, pre-2020)*
*   *Summary:* The main paper introducing PyTorch. While pre-2020, PyTorch is one of the two dominant frameworks (along with TensorFlow) for Deep Learning (Section 10.5) used extensively in astronomical research.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. *(Note: Foundational software paper, pre-2020)*
*   *Summary:* The foundational paper for the `scikit-learn` library. Although pre-2020, `scikit-learn` remains the standard Python library for general-purpose Machine Learning (Section 10.3) and is used in several examples (10.6.1, 10.6.2, 10.6.3, 10.6.6).

Pichara, K., Protopapas, P., Huijse, P., & Zegers, P. (2021). An Unsupervised Active Learning Method for Astronomical Time Series Classification. *arXiv preprint arXiv:2110.03892*. https://doi.org/10.48550/arXiv.2110.03892
*   *Summary:* Explores unsupervised and active learning methods for classifying time series data (relevant to Section 10.3/10.4). Illustrates advanced ML techniques for handling large unlabeled time-domain datasets.

Villaescusa-Navarro, F., Angles-Alcazar, D., Genel, S., Nagai, D., Nelson, D., Pillepich, A., Hernquist, L., Marinacci, F., Pakmor, R., Springel, V., Vogelsberger, M., ZuHone, J., & Weinberger, R. (2023). Splashdown: Representing cosmological simulations through neural networks. *The Astrophysical Journal Supplement Series, 266*(2), 38. https://doi.org/10.3847/1538-4365/accc3e
*   *Summary:* Focuses on using Deep Learning (specifically Autoencoders/neural representations, Section 10.5) to compress and represent large cosmological simulation datasets, addressing data volume challenges (Section 10.1).

Zhang, Z., & Bloom, J. S. (2020). DeepCR: Cosmic Ray Removal with Deep Learning. *The Astrophysical Journal, 889*(1), 49. https://doi.org/10.3847/1538-4357/ab6195
*   *Summary:* Presents a deep learning (CNN) approach for cosmic ray removal in astronomical images, demonstrating an application of DL to data processing challenges (Section 10.4).

