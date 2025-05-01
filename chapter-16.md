---
# Chapter 16
# Reproducibility in Computational Astrophysics
---

This chapter addresses the critical principles and practices underpinning reproducibility and transparency in computational astrophysics. As astronomical research becomes increasingly reliant on complex software pipelines, sophisticated data analysis algorithms, and large-scale numerical simulations, ensuring that computational results can be independently verified, understood, and built upon by the wider scientific community is paramount for the credibility and progress of the field. The discussion begins by outlining the core tenets of reproducibility and the broader context of the Open Science movement, emphasizing the scientific, ethical, and practical motivations for making computational research transparent and repeatable. It then delves into specific best practices for writing reproducible code, focusing on clarity, modularity, and comprehensive documentation. The indispensable role of version control systems, particularly Git and platforms like GitHub, for tracking changes, managing code development, and facilitating collaboration is detailed. Strategies for robust computational environment management using tools like Conda or virtual environments are presented as essential for ensuring that software dependencies can be precisely replicated. The chapter contrasts different workflow paradigms, such as executable scripts versus interactive Jupyter Notebooks, discussing their respective strengths and weaknesses regarding reproducibility. The concept of data provenance—meticulously tracking the origin and processing history of data—is highlighted as crucial for understanding and validating results. Finally, best practices for sharing code and data through archives, assigning persistent identifiers (DOIs), and applying appropriate licenses are discussed, including leveraging the infrastructure of the Virtual Observatory (VO) for standardized and reproducible data access via tools like `pyvo` and `astroquery`. Practical examples illustrate setting up reproducible project structures and utilizing relevant tools.

---

**16.1 Reproducibility and Open Science Principles**

In any scientific discipline, the ability for independent researchers to reproduce experimental or analytical results is a cornerstone of the scientific method, underpinning the verification, validation, and cumulative advancement of knowledge (Peng, 2011; National Academies of Sciences, Engineering, and Medicine, 2019). In fields like astrophysics, where computation now plays a central role alongside theory and observation (Chapter 1), this principle extends fundamentally to computational workflows. **Reproducibility** in this context generally refers to the ability of an independent researcher to obtain qualitatively similar results using the original author's data and analysis code/software (often termed "computational reproducibility"). A related, sometimes stricter, concept is **replicability**, which refers to obtaining consistent results when performing a new, independent study (potentially with new data) intended to answer the same scientific question (Plesser, 2018; ACM, 2020). Ensuring computational reproducibility is a critical first step towards broader scientific replicability.

The increasing complexity of astronomical data analysis pipelines, the reliance on sophisticated software libraries with intricate dependencies, and the sheer volume of data involved make achieving computational reproducibility a non-trivial challenge (Siebert et al., 2022; Shamir et al., 2020). However, striving for reproducibility offers significant benefits:
*   **Verification and Trust:** Allows independent researchers (including reviewers, collaborators, and future researchers) to verify the correctness of the analysis and build confidence in the published results.
*   **Debugging and Understanding:** Facilitates debugging of code and deeper understanding of the analysis methodology by enabling others (and the original authors later) to step through the workflow.
*   **Building Upon Previous Work:** Enables researchers to easily reuse, adapt, and extend existing analysis code for new datasets or related scientific questions, accelerating future research.
*   **Avoiding Duplication of Effort:** Sharing reproducible workflows prevents others from having to "reinvent the wheel" for standard analysis tasks.
*   **Training and Education:** Provides valuable, executable examples for training students and newcomers in computational methods.
*   **Preventing Errors:** The discipline required to make analysis reproducible often helps authors identify and correct errors in their own code or logic before publication.

Reproducibility is a core component of the broader **Open Science** movement, which advocates for making scientific research processes and outputs (including data, code, methods, publications) transparent, accessible, and reusable for everyone (Wilkinson et al., 2016; Nosek et al., 2015; Allen et al., 2022). Key principles of Open Science relevant to computational astrophysics include:
*   **Open Data:** Making observational data and simulation outputs publicly available in accessible archives, following FAIR principles (Findable, Accessible, Interoperable, Reusable).
*   **Open Source Software:** Developing and sharing analysis software and computational tools under open source licenses, allowing inspection, reuse, modification, and contribution by the community (e.g., the Astropy ecosystem - Astropy Collaboration et al., 2022).
*   **Open Methods:** Clearly and completely documenting the computational methods, algorithms, parameters, and software versions used in the analysis, often by sharing the analysis code itself.
*   **Open Access Publishing:** Making research publications freely available to read without subscription barriers.

Journals, funding agencies, and institutions are increasingly recognizing the importance of reproducibility and Open Science, implementing policies that encourage or mandate the sharing of data and code associated with publications (e.g., Stodden et al., 2016). Embracing reproducibility is not just about compliance; it is about fostering a more robust, efficient, collaborative, and trustworthy scientific enterprise. The subsequent sections detail practical techniques and tools that enable astronomers to make their computational research more reproducible.

**16.2 Practices for Reproducible Code**

The foundation of computational reproducibility lies in the analysis code itself. Code that is difficult to understand, run, or modify poses significant barriers to verification and reuse. Adopting good software development practices, even for relatively small analysis scripts, is crucial (Wilson et al., 2014; Turk, 2013; Siebert et al., 2022).

Key practices include:
1.  **Clarity and Readability:**
    *   **Meaningful Variable/Function Names:** Use descriptive names that clearly indicate the purpose of variables and functions (e.g., `spectral_resolution` instead of `sr`, `calculate_sky_background` instead of `proc1`).
    *   **Consistent Style:** Follow established style guides (e.g., PEP 8 for Python) for formatting, indentation, and naming conventions. Tools like `flake8` or `black` can help enforce consistency.
    *   **Comments:** Use comments judiciously to explain *why* something is done or clarify complex logic, not just *what* the code does (which should be clear from the code itself if well-written). Explain assumptions or non-obvious choices.
    *   **Avoid "Magic Numbers":** Replace hard-coded numerical constants with named variables defined at the beginning of the script or function, improving readability and making parameters easier to change.

2.  **Modularity:**
    *   **Functions:** Break down complex analysis workflows into smaller, well-defined functions, each responsible for a specific task (e.g., `load_data`, `subtract_bias`, `detect_sources`, `perform_photometry`, `plot_results`). This improves organization, makes code easier to test and debug, and promotes reusability.
    *   **Pure Functions:** Where possible, write functions that are "pure" – their output depends only on their input arguments, and they have no side effects (e.g., modifying global variables). Pure functions are easier to reason about and test.
    *   **Code Structure:** Organize code logically within scripts or modules. Group related functions together. Use clear separation between data loading, processing, analysis, and plotting steps.

3.  **Documentation:**
    *   **Docstrings:** Write informative docstrings for functions, classes, and modules explaining their purpose, arguments (including types and units), return values, and any important usage notes or assumptions. Follow standard docstring formats (e.g., NumPy or Google style). Tools like Sphinx can automatically generate documentation from docstrings.
    *   **README Files:** Include a `README` file at the top level of a project directory explaining what the project/code does, how to install dependencies, how to run the analysis, and providing contact information.
    *   **Inline Comments:** As mentioned, use comments to clarify non-obvious parts of the code.

4.  **Automation:**
    *   **Scripting:** Encapsulate the entire analysis workflow within executable scripts (e.g., Python scripts, shell scripts) rather than relying solely on manual steps performed interactively (e.g., within a basic Python interpreter or GUI). Scripts document the exact sequence of operations and make the analysis easily repeatable.
    *   **Configuration Files:** Store parameters (e.g., file paths, thresholds, model settings) in separate configuration files (e.g., YAML, JSON, INI formats) rather than hard-coding them directly in scripts. This makes it easier to change parameters and run the analysis with different settings without modifying the code. Libraries like `configparser` or `PyYAML` can parse these files.
    *   **Workflow Management Tools (Advanced):** For highly complex, multi-stage pipelines with dependencies between steps, consider using workflow management systems like Snakemake or Nextflow (Köster & Rahmann, 2012; Di Tommaso et al., 2017). These tools define workflows using specialized languages, manage dependencies, automate execution (potentially across clusters), and enhance reproducibility.

5.  **Testing:**
    *   **Assertions:** Include assertions (`assert` statements) within the code to check for expected conditions or intermediate results, helping to catch errors early during execution.
    *   **Unit Tests (Recommended):** For reusable functions or critical analysis steps, write unit tests using frameworks like `pytest`. Unit tests are small pieces of code that automatically verify the correctness of individual functions by providing known inputs and checking for expected outputs. Running tests regularly helps ensure code modifications don't break existing functionality.

Adopting these practices requires an initial investment of time but pays significant dividends in terms of code reliability, maintainability, reusability, and, crucially, the reproducibility of the scientific results derived from it.

**16.3 Version Control with Git and GitHub**

Scientific software and analysis scripts are rarely static; they evolve as bugs are fixed, features are added, parameters are tuned, or analysis strategies are refined. Tracking these changes systematically is essential for reproducibility, allowing researchers (including future selves) to access specific historical versions of the code corresponding to published results or intermediate analyses. **Version Control Systems (VCS)** are software tools designed explicitly for this purpose. **Git** is the dominant, distributed VCS used worldwide in software development and increasingly in scientific research (Blischak et al., 2016; Perez-Riverol et al., 2016). **GitHub**, GitLab, and Bitbucket are popular web-based platforms that provide hosting for Git repositories, along with powerful tools for collaboration, issue tracking, and project management.

**Why Use Version Control (Git)?**
*   **Tracking History:** Git records snapshots (called "commits") of your entire project directory (code, scripts, configuration files, documentation, potentially small data files) every time you choose to save changes. Each commit stores the full state, the changes made relative to the previous commit, a unique identifier (hash), author information, timestamp, and a descriptive commit message explaining the changes. This creates a complete, navigable history of the project's evolution.
*   **Reverting Changes:** If errors are introduced or an analysis direction proves unfruitful, Git makes it easy to revert files or the entire project back to any previous committed state, preventing loss of work and facilitating exploration.
*   **Branching and Merging:** Git allows creating separate lines of development called "branches." Researchers can create branches to experiment with new features, analysis techniques, or bug fixes without affecting the main, stable version of the code (often the `main` or `master` branch). Once changes on a branch are complete and tested, they can be **merged** back into the main branch. This workflow is essential for managing parallel development and experimentation without disrupting the core analysis.
*   **Understanding Changes:** Tools like `git diff` allow comparing different versions of files or commits, highlighting exactly what changed, which is invaluable for debugging or understanding how results may have evolved.
*   **Backup (Partial):** While not a replacement for dedicated backups, storing a Git repository on a remote platform like GitHub provides an offsite copy of the codebase.
*   **Collaboration:** Git is designed for distributed workflows. Multiple researchers can work on the same codebase independently, track each other's changes, and merge their contributions efficiently (see Chapter 17).

**Basic Git Workflow:**
1.  **Initialize Repository:** In the main project directory, run `git init` to create a new Git repository. This creates a hidden `.git` subdirectory that stores the entire history and metadata.
2.  **Stage Changes:** After modifying files, use `git add <filename>` or `git add .` (to add all changes in the current directory) to stage the changes you want to include in the next commit. Staging allows selective committing.
3.  **Commit Changes:** Use `git commit -m "Descriptive commit message"` to save the staged changes as a snapshot in the repository history. The message should concisely explain the purpose of the changes made.
4.  **View Status/History:** Use `git status` to see current modifications and staged changes. Use `git log` to view the commit history.
5.  **Branching:** Create a new branch with `git branch <branch_name>`. Switch to it with `git checkout <branch_name>` (or `git switch <branch_name>` in newer Git versions). Create and switch in one step with `git checkout -b <branch_name>`.
6.  **Merging:** After completing work on a branch, switch back to the main branch (`git checkout main`) and merge the changes from the feature branch (`git merge <branch_name>`). Git attempts to automatically combine the changes; conflicts may arise if the same lines were modified differently on both branches, requiring manual resolution.

**Using GitHub (or similar platforms):**
1.  **Create Remote Repository:** Create a new repository on GitHub.com.
2.  **Connect Local to Remote:** Link your local repository to the remote one using `git remote add origin <repository_url>`.
3.  **Push Changes:** Upload your local commits (e.g., from the `main` branch) to the remote repository using `git push origin main`.
4.  **Pull Changes:** Download changes made by collaborators (or yourself from another machine) from the remote repository to your local one using `git pull origin main`.
5.  **Cloning:** Create a local copy of an existing remote repository using `git clone <repository_url>`.

**Reproducibility Benefits:**
*   **Linking Code to Results:** When publishing results, researchers can cite the specific Git commit hash corresponding to the exact version of the code used for the analysis, allowing others to check out precisely that version.
*   **Transparency:** Hosting code publicly on platforms like GitHub allows others to inspect the analysis methodology directly.
*   **Tracking Parameter Changes:** Committing configuration files alongside code ensures that changes in parameters are also version controlled.

Using Git and platforms like GitHub should be considered standard practice for any computational research project, regardless of size. It provides an essential safety net, facilitates collaboration, and is a cornerstone of reproducible computational science.

**16.4 Computational Environment Management**

Reproducing computational results requires not only the original code and data but also the exact **computational environment** in which the code was executed (Turk, 2013; Siebert et al., 2022). This includes the specific version of the programming language (e.g., Python 3.9 vs. 3.10), the versions of all required libraries and packages (e.g., NumPy, SciPy, Astropy, `specutils`, `photutils`, `scikit-learn`), and potentially even operating system dependencies. Software libraries evolve rapidly; functions may be added, removed, or change behavior between versions. Running the same analysis code with different library versions can lead to subtle discrepancies or even outright errors, hindering reproducibility. Therefore, precisely documenting and recreating the software environment is critical.

Tools like **Conda** (Section 1.6) and Python's built-in **`venv`** module combined with the **`pip`** package installer are essential for managing computational environments effectively.

**Using Conda for Environment Management:**
*   **Environment Creation:** As discussed in Section 1.6, Conda allows creating isolated environments with specific Python versions and packages (`conda create --name myenv python=3.10 astropy numpy scipy pandas ...`).
*   **Activation:** Environments must be activated before use (`conda activate myenv`).
*   **Package Installation:** Packages are installed into the active environment (`conda install <package>` or `pip install <package>`). Conda excels at managing complex dependencies, including non-Python libraries.
*   **Exporting Environments (`environment.yml`):** The key reproducibility feature is exporting the environment's specification:
    `conda env export > environment.yml`
    This command generates a YAML file (`environment.yml`) listing all packages (including transitive dependencies) installed in the environment, along with their precise version numbers and the channel they were installed from (e.g., `conda-forge`).
*   **Recreating Environments:** This `environment.yml` file should be shared alongside the analysis code (e.g., committed to the Git repository). Another researcher (or the original author on a different machine or at a later time) can then precisely recreate the identical environment using:
    `conda env create -f environment.yml`
    This command reads the file, resolves the dependencies, and installs the exact specified package versions, ensuring the software environment matches the one used for the original analysis.

**Using `venv` and `pip`:**
For projects relying solely on Python packages installable via `pip`, Python's built-in `venv` module provides an alternative.
*   **Environment Creation:** Create an environment in a specified directory (e.g., `myenv_dir`): `python -m venv myenv_dir`
*   **Activation:** Activate the environment (syntax varies by OS, e.g., `source myenv_dir/bin/activate` on Linux/macOS).
*   **Package Installation:** Install packages using `pip install <package>`.
*   **Exporting Dependencies (`requirements.txt`):** Generate a list of installed Python packages and their versions:
    `pip freeze > requirements.txt`
    This creates a `requirements.txt` file listing packages and versions (e.g., `numpy==1.24.3`, `astropy==5.3.4`).
*   **Recreating Environments:** Share the `requirements.txt` file. Others can create a new virtual environment, activate it, and install the exact dependencies using:
    `pip install -r requirements.txt`
*   **Limitations:** `pip` and `requirements.txt` primarily manage Python packages. They are less effective at handling complex non-Python dependencies (e.g., C libraries, compilers) that some scientific packages require, which is where Conda often provides a more robust solution.

**Best Practices:**
*   **Create Environments per Project:** Avoid installing packages into the base system Python. Create a separate, isolated environment for each distinct research project or analysis.
*   **Document Dependencies:** Always generate and commit the appropriate dependency file (`environment.yml` for Conda, `requirements.txt` for pip/venv) along with your code in version control.
*   **Specify Versions:** Ensure the dependency file lists specific versions for all critical packages to guarantee reproducibility. Avoid relying on installing the "latest" version, as this can change over time. Conda's `environment.yml` is generally better at capturing the full dependency graph, including non-Python components.
*   **Consider Containers (Advanced):** For even greater reproducibility, especially across different operating systems or when complex system libraries are involved, containerization technologies like Docker or Apptainer (formerly Singularity) can encapsulate the entire software environment, including the OS libraries (see Chapter 17).

Managing computational environments meticulously using tools like Conda or venv/pip and documenting dependencies through `environment.yml` or `requirements.txt` files is a non-negotiable step for ensuring that computational analyses can be reliably reproduced by others or by oneself in the future.

**16.5 Reproducible Analysis Workflows: Scripts vs. Jupyter Notebooks**

The way an analysis workflow is implemented significantly impacts its reproducibility. Two common paradigms in Python-based scientific computing are executable scripts and interactive Jupyter Notebooks. Each has strengths and weaknesses regarding reproducibility.

**Executable Scripts (.py files):**
*   **Structure:** Contain Python code intended to be run sequentially from top to bottom, typically executed from the command line (e.g., `python analysis_script.py`).
*   **Pros for Reproducibility:**
    *   **Linear Execution:** The execution order is explicit and unambiguous, making it clear how results are generated step-by-step.
    *   **Automation:** Easily automated and incorporated into larger pipelines or batch processing systems.
    *   **Version Control Friendliness:** Plain text files are well-suited for version control with Git; diffs clearly show code changes.
    *   **Environment Integration:** Cleanly separates code from the execution environment; environment management (Section 16.4) ensures dependencies are met.
    *   **Parameterization:** Easily parameterized using command-line arguments (`argparse`) or configuration files (Section 16.2).
*   **Cons:**
    *   **Less Interactive Exploration:** Less convenient for iterative data exploration and visualization during development compared to notebooks. Output (plots, tables) is typically saved to files rather than displayed inline.
    *   **Documentation Separation:** Explanatory text and visualizations are separate from the code execution logic (though docstrings and comments help).

**Jupyter Notebooks (.ipynb files):**
*   **Structure:** Combine executable code cells, narrative text (Markdown), mathematical equations (LaTeX), and inline visualizations (plots, tables) within a single document, typically run interactively in a web browser via a Jupyter server (Kluyver et al., 2016).
*   **Pros:**
    *   **Interactive Exploration & Visualization:** Excellent for developing analysis ideas, exploring data interactively, and visualizing intermediate and final results directly alongside the code and explanations.
    *   **Narrative Integration:** Allows weaving explanatory text, equations, and code together, creating a literate programming environment that can serve as a computational narrative or report.
    *   **Educational Value:** Very effective for tutorials, demonstrations, and sharing step-by-step analyses.
*   **Cons for Reproducibility:**
    *   **Out-of-Order Execution:** Code cells can be executed in any order, and variables/states persist across cells. This makes it difficult to guarantee that re-running the notebook from top to bottom will produce the same result as the interactively generated version, as the final state might depend on an execution history that isn't explicitly captured. **This is a major reproducibility hazard.**
    *   **Hidden State:** Variables defined or modified in earlier cells affect later cells, potentially leading to unexpected behavior if cells are re-run selectively or in a different order. Kernels might need restarting to ensure a clean state.
    *   **Version Control Challenges:** Notebook files (`.ipynb`) are JSON files containing code, output, and metadata. Raw diffs in Git can be difficult to interpret, although tools like `nbdime` exist to help compare notebooks more effectively. Outputs stored within the notebook increase file size and can cause merge conflicts.
    *   **Environment Coupling:** While notebooks run within specific kernels/environments, ensuring the *correct* environment is used when reopening or rerunning a notebook requires careful management by the user (or integration with environment specification tools).

**Best Practices for Reproducible Notebooks:**
*   **Always Run Sequentially:** Before saving or sharing, restart the kernel and run all cells sequentially from top to bottom (`Kernel -> Restart & Run All`) to ensure the notebook produces the intended results in a linear fashion.
*   **Clear Cell Dependencies:** Structure the notebook logically so that dependencies flow clearly from earlier cells to later ones. Avoid jumping back and forth or redefining variables unnecessarily.
*   **Parameterize:** Define parameters (file paths, thresholds) in early cells or load them from configuration files, rather than scattering them throughout the code.
*   **Version Control Strategy:** Commit notebooks to Git, potentially after clearing outputs (`jupyter nbconvert --clear-output`) to reduce noise in diffs. Use tools like `nbdime` for comparing notebook versions. Consider exporting the notebook to an executable script (`jupyter nbconvert --to script`) for archival or automated execution, although this loses the narrative context.
*   **Environment Documentation:** Include cells that check for or document the required package versions (e.g., using `pip freeze` or checking versions explicitly) or provide clear instructions for setting up the correct Conda/venv environment.

**Conclusion:** While Jupyter Notebooks are excellent for exploration, development, and documentation, executable scripts generally offer a more robust and inherently reproducible workflow for finalized analyses, especially those intended for automated execution or as part of larger pipelines. A common effective strategy is to develop and explore interactively in notebooks, but then refactor the finalized analysis into well-documented, modular Python scripts or packages for production runs and archival, potentially keeping the notebook as supplementary documentation of the development process (but clearly indicating the script as the definitive analysis workflow).

**16.6 Data Provenance**

**Data provenance** refers to the documented history or lineage of a piece of data, tracing its origin, processing steps, and transformations from raw acquisition to its final state (Simmhan et al., 2005; Moreau et al., 2008; Chirigati & Freire, 2023). Maintaining detailed provenance information is crucial for understanding how a specific result was derived, assessing its reliability, debugging potential errors in the processing pipeline, and ensuring reproducibility (Miles et al., 2007; Siebert et al., 2022). Without provenance, interpreting data products, especially complex ones derived from automated pipelines, becomes opaque and difficult to verify.

Key components of data provenance information include:
*   **Origin:** Where did the raw data come from (telescope, instrument, simulation code, specific observation ID, proposal ID)? Who acquired it? When?
*   **Processing Steps:** What specific algorithms and software tools were applied to the data at each stage (e.g., bias subtraction, flat-fielding, cosmic ray rejection, source detection, calibration, stacking, spectral extraction, model fitting)?
*   **Software Versions:** The exact versions of all software libraries, packages, and potentially the operating system and compilers used for each processing step. This is critical because changes in software versions can alter results (Section 16.4).
*   **Parameters and Configuration:** The specific parameter values, thresholds, configuration files, or command-line options used for each algorithm or processing step.
*   **Input Data:** References to the specific input data files (e.g., raw science frames, calibration files, reference catalogs) used at each step, including their versions or unique identifiers if available.
*   **Intermediate Products:** References to any intermediate data products generated during the workflow.
*   **Execution Environment:** Details about the hardware and software environment where the processing occurred (e.g., specific cluster, operating system version).
*   **Timestamping:** Recording when each processing step was performed.
*   **Agent:** Information about the person or automated system responsible for executing the processing step.

**Capturing Provenance:**
*   **Manual Documentation:** Maintaining detailed logbooks, README files, or comments within scripts documenting the processing steps, software versions, and parameters used. Prone to errors and omissions, difficult to automate.
*   **FITS Header Keywords:** Utilizing standard (`HISTORY`, `COMMENT`) and custom FITS keywords to record processing steps, software names/versions, parameter values, and input file names directly within the headers of processed FITS files. Common practice in astronomical pipelines, but can become verbose and lacks structured queryability.
*   **Logging:** Implementing detailed logging within analysis scripts to record operations, parameters, software versions, and timestamps to separate log files.
*   **Version Control (Git):** Tracking changes to code, scripts, and configuration files using Git (Section 16.3) provides provenance for the *analysis workflow itself*. Linking specific data products to specific Git commit hashes is crucial.
*   **Workflow Management Systems:** Tools like Snakemake or Nextflow automatically track dependencies and execution details, providing some level of workflow provenance.
*   **Specialized Provenance Systems/Standards:** More formal approaches involve using dedicated provenance tracking systems or adhering to standards like the W3C PROV model (Moreau et al., 2008). These systems aim to capture provenance information in a structured, machine-readable format, allowing for automated querying and analysis of data lineage. Implementations include libraries that automatically capture provenance during script execution or database systems designed to store provenance graphs. The IVOA is also developing standards for representing provenance within the Virtual Observatory context (Servillat et al., 2015; IVOA Provenance Data Model - Proposed Recommendation).

**Benefits:** Recording detailed provenance facilitates debugging (by tracing errors back through the processing chain), ensures transparency, enables precise reproducibility (by allowing others to replicate the exact processing steps with the same inputs, software, and parameters), supports validation of results, and increases the long-term value and trustworthiness of derived data products. While capturing fully comprehensive provenance automatically can be challenging, adopting practices like using version control, documenting environments, logging key steps, and utilizing FITS HISTORY keywords significantly enhances the traceability and reproducibility of computational results.

**16.7 Data and Code Sharing Practices**

Making the data and code underlying published scientific results publicly available is a cornerstone of Open Science and essential for enabling verification, reproducibility, and future research built upon that work (Stodden et al., 2016; Wilkinson et al., 2016; Allen et al., 2022). Sharing practices involve depositing data and code in appropriate repositories, ensuring they are well-documented, assigning persistent identifiers, and applying clear usage licenses.

**Data Sharing:**
*   **Repositories:** Deposit research data (raw, processed, derived catalogs, simulation outputs) in established, trustworthy digital repositories that ensure long-term preservation and accessibility.
    *   *Astronomy Archives:* Major observatories (e.g., MAST for HST/JWST, ESO Science Archive, Keck Observatory Archive, NOIRLab Astro Data Archive), survey data centers (e.g., SDSS Science Archive Server, IRSA for Spitzer/WISE), and international data centers (e.g., CDS - VizieR/SIMBAD, CADC) host vast amounts of observational data. Data associated with publications should ideally link back to these primary archives.
    *   *General Purpose Repositories:* Platforms like **Zenodo**, **Figshare**, Harvard Dataverse, or institutional repositories can host derived data products, supplementary materials, or simulation data not suitable for primary archives. These platforms often provide Digital Object Identifiers (DOIs) for datasets.
*   **FAIR Principles:** Data should be shared following the FAIR principles:
    *   **Findable:** Assign globally unique and persistent identifiers (e.g., DOIs). Ensure rich metadata describes the data. Index the data in searchable resources (e.g., VizieR, VO Registry).
    *   **Accessible:** Data should be retrievable via standard protocols (e.g., HTTP, VO protocols like TAP/SIA/SSA). Authentication/authorization should be clear if needed. Metadata should remain accessible even if data are removed.
    *   **Interoperable:** Use standard formats (e.g., FITS, VOTable), vocabularies (e.g., IVOA UCDs), and coordinate systems (WCS). Ensure metadata uses community standards.
    *   **Reusable:** Provide clear documentation, data provenance information (Section 16.6), and a clear data usage license (see below).
*   **Data Management Plans (DMPs):** Many funding agencies now require DMPs outlining how data generated during a project will be managed, documented, preserved, and shared.

**Code Sharing:**
*   **Repositories:** Share analysis code, scripts, simulation codes, or software packages publicly, typically via version control hosting platforms.
    *   **GitHub / GitLab / Bitbucket:** The standard platforms for hosting Git repositories (Section 16.3). Provide version control, issue tracking, collaboration tools, and visibility.
    *   **Zenodo:** Can archive specific software releases from GitHub repositories and assign DOIs to the software itself, making it citable.
    *   **Astrophysics Source Code Library (ASCL):** A curated registry for astronomy software, assigning unique ASCL IDs for citation and discovery (Allen et al., 2013). Registering significant software packages in ASCL is recommended.
*   **Documentation:** Code must be accompanied by clear documentation (README files, installation instructions, usage examples, API documentation if applicable) to enable others to understand and use it (Section 16.2).
*   **Dependencies:** Clearly document all software dependencies and their specific versions, ideally via environment files (`environment.yml`, `requirements.txt` - Section 16.4). Consider containerization (Docker/Apptainer) for complex environments (Chapter 17).
*   **Licensing:** Apply a clear open source license to the code to define how others can use, modify, and distribute it. Common licenses in scientific software include MIT, BSD (2-clause or 3-clause), Apache 2.0, or GPLv3. Choosing an OSI-approved license is recommended. The license should be included as a `LICENSE` file in the repository.

**Persistent Identifiers (DOIs):** Assigning Digital Object Identifiers (DOIs) to datasets, software releases, and publications provides stable, persistent links that ensure long-term citability and accessibility, even if URLs change. Services like Zenodo can mint DOIs for data and software uploads/archives. Citing data and software using DOIs is becoming standard practice.

Adopting open sharing practices for data and code requires effort but significantly enhances the impact and reliability of research, fostering a more collaborative and reproducible scientific ecosystem.

**16.8 The Virtual Observatory (VO) for Reproducible Data Access**

The **International Virtual Observatory Alliance (IVOA)** develops standards and protocols aimed at making astronomical data from diverse archives worldwide seamlessly discoverable, accessible, and interoperable (Allen et al., 2022; Dowler et al., 2022; Plante et al., 2011). By providing standardized ways to query and retrieve data programmatically, the Virtual Observatory (VO) infrastructure plays a crucial role in enabling reproducible data access within computational workflows. Instead of relying on manual downloads via web interfaces (which are difficult to script and reproduce), researchers can use VO protocols and tools like `pyvo` and `astroquery` to fetch the exact datasets needed directly within their analysis scripts.

Key VO protocols facilitating reproducible data access include:
*   **Table Access Protocol (TAP):** The standard for querying tabular data (catalogs) using ADQL (Section 10.2). Allows precise selection of data based on complex criteria (spatial, temporal, parameter ranges). Queries submitted via TAP are inherently scriptable and reproducible. `pyvo.dal.TAPService` and `astroquery` modules (e.g., `astroquery.gaia`) provide interfaces.
*   **Simple Image Access (SIA) / Simple Spectral Access (SSA):** Protocols for querying image and spectral datasets, respectively, based primarily on sky position and optionally other parameters (e.g., wavelength range, time). They return lists of available datasets matching the criteria, along with access URLs. While less flexible than TAP for complex queries, they provide standardized ways to discover and access image/spectral data archives programmatically. `pyvo.dal.SIAv2Service`, `pyvo.dal.SSAService`, and various `astroquery` modules utilize these.
*   **DataLink:** A protocol designed to link related datasets. Given an identifier for a primary dataset (e.g., an observation ID), DataLink allows discovering associated files like calibration data, previews, pipeline logs, or data products derived from it (Dowler et al., 2015; IVOA DataLink Recommendation). This is crucial for accessing the full context needed to reproduce processing or understand data quality. `pyvo.datalink` provides an interface.
*   **VO Registry:** A distributed database containing descriptions of VO-compliant data collections, services (TAP, SIA, SSA, etc.), and organizations. Allows programmatic discovery of relevant data resources based on keywords, wavelength coverage, data type, etc. (`pyvo.registry`).

**Benefits for Reproducibility:**
*   **Scriptable Access:** VO protocols are designed for machine-to-machine communication, making data retrieval easily scriptable within Python using `pyvo` or `astroquery`.
*   **Precise Queries:** ADQL queries via TAP allow for precise, unambiguous specification of the desired data subset, ensuring the same data can be retrieved later.
*   **Standardization:** Relying on IVOA standards promotes interoperability and reduces dependence on archive-specific interfaces, making scripts more portable and less likely to break due to website changes.
*   **Provenance:** Queries themselves (e.g., the ADQL string used) can be saved as part of the analysis provenance, documenting exactly how the input data was obtained.
*   **Access to Calibration Data:** Protocols like DataLink facilitate programmatic access to necessary calibration files, crucial for reproducing reduction pipelines.

By leveraging VO standards and tools, researchers can incorporate data retrieval directly into their reproducible workflows, ensuring that the specific data used for an analysis can be clearly documented and potentially re-acquired by others attempting to verify the results. This moves away from manual downloads towards more transparent and automated data acquisition practices.

**16.9 Examples in Practice (Python & Workflow): Reproducible Project Setups**

The following examples illustrate practical aspects of setting up reproducible computational projects in different astronomical contexts, incorporating concepts like Git repository structure, environment management files, documented scripts or notebooks, and programmatic data access.

**16.9.1 Solar: Git Repo Structure for SDO Analysis**
Analyzing SDO data might involve fetching specific data products (e.g., AIA images, HMI magnetograms for a certain time range and feature), processing them (e.g., alignment, feature tracking), and generating plots or derived data. A reproducible project structure using Git is essential.

```
sdo_flare_analysis/
├── README.md            # Project overview, setup instructions, run commands
├── environment.yml      # Conda environment specification (sunpy, astropy, etc.)
├── data/                  # Directory for storing downloaded/raw data (potentially excluded via .gitignore)
│   └── external/          # Data from external sources (e.g., GOES)
│   └── processed/         # Intermediate or processed data products
├── notebooks/             # Jupyter notebooks for exploration, visualization, reporting
│   └── 01_data_exploration.ipynb
│   └── 02_feature_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── scripts/               # Core analysis scripts (executable Python files)
│   ├── fetch_sdo_data.py  # Script using sunpy.net.Fido/VSO to download data
│   ├── process_images.py  # Script for alignment, processing steps
│   ├── analyze_features.py # Script for measurements/analysis
│   └── plot_results.py    # Script to generate final plots/figures
├── src/                   # Optional: Python package containing reusable functions
│   ├── __init__.py
│   └── analysis_utils.py
├── results/               # Output results (figures, tables, derived values)
│   ├── figures/
│   └── tables/
├── .gitignore           # Specifies files/directories Git should ignore (e.g., large data files, secrets)
└── LICENSE                # Software license file (e.g., MIT, BSD)

```
**Explanation:** This structure separates concerns: `README.md` provides entry point documentation. `environment.yml` ensures environment reproducibility (Section 16.4). `scripts/` contains the core, potentially automated workflow using Python scripts (Section 16.5). `notebooks/` holds exploratory work or final reports (run sequentially!). `src/` contains reusable Python functions (good practice for larger projects). `data/` and `results/` hold data and outputs (large data often excluded from Git via `.gitignore`, potentially stored elsewhere with pointers/download scripts). The entire directory is tracked using Git (Section 16.3), with a clear `LICENSE`. The `fetch_sdo_data.py` script would use libraries like `sunpy` (interfacing with VO-like services like VSO) for reproducible data acquisition (Section 16.8).

**16.9.2 Planetary: Documented Jupyter Notebook for Simulation Analysis**
Analyzing outputs from planetary simulations (e.g., orbital dynamics, atmospheric models) is often done interactively. A well-documented Jupyter Notebook can be reproducible if structured carefully.

```python
# --- Top Cell: Environment Setup ---
# Ensure necessary libraries are installed and document versions
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rebound # Example simulation library
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Rebound: {rebound.__version__}")
# Provide instructions: "Run this notebook using the environment defined in environment.yml"

# --- Cell 2: Load Simulation Parameters ---
# Load parameters from a config file or define clearly here
config_file = 'simulation_params.yaml'
# params = load_config(config_file) # Placeholder
params = {'integration_time': 1e4, 'n_particles': 100}
print("Loaded parameters:", params)

# --- Cell 3: Load Simulation Output Data ---
# Load data file (e.g., generated by a separate simulation script)
data_file = 'simulation_output.csv' # Or other format
# simulation_data = pd.read_csv(data_file) # Placeholder
# Create dummy data for example
simulation_data = pd.DataFrame({
    'time': np.linspace(0, params['integration_time'], 50),
    'particle_0_x': np.random.rand(50), 'particle_0_y': np.random.rand(50),
    'particle_1_x': np.random.rand(50), 'particle_1_y': np.random.rand(50),
    # ... more particle data ...
})
print(f"Loaded simulation data: {len(simulation_data)} timesteps")

# --- Cell 4: Data Processing / Analysis Step 1 ---
# Markdown explaining the step
# Code performing a specific calculation
# E.g., Calculate distance between particle 0 and 1
simulation_data['distance_01'] = np.sqrt(
    (simulation_data['particle_0_x'] - simulation_data['particle_1_x'])**2 +
    (simulation_data['particle_0_y'] - simulation_data['particle_1_y'])**2
)
print("Calculated distance between particles 0 and 1.")

# --- Cell 5: Visualization Step 1 ---
# Markdown explaining the plot
# Code generating the plot
plt.figure()
plt.plot(simulation_data['time'], simulation_data['distance_01'])
plt.xlabel("Time (years)")
plt.ylabel("Distance (AU)")
plt.title("Distance between Particle 0 and 1 vs Time")
plt.grid(True)
plt.show()

# --- Subsequent Cells: More Analysis and Plots ---
# Continue with clear separation of steps and explanations

# --- Final Cell: Summary / Save Derived Data ---
# Summarize key results
# Save any derived tables or figures
# results_summary = {'mean_distance': simulation_data['distance_01'].mean()}
# save_results(results_summary, 'results_summary.json') # Placeholder
# plt.savefig('final_plot.pdf') # Save figures
print("Analysis complete. Summary saved.")

# Note at end: "This notebook should be run sequentially from top to bottom."
```
**Explanation:** This Jupyter notebook example emphasizes reproducibility practices (Section 16.5). The first cell documents the required environment and library versions. Parameters and data loading are clearly defined early on. Each subsequent step (analysis calculation, visualization) is contained within separate cells, preceded by Markdown text explaining the purpose. Code is commented where necessary. By ensuring a logical flow and documenting dependencies, this notebook becomes more understandable and reproducible, especially if accompanied by an `environment.yml` file and if always run sequentially (`Restart & Run All`).

**16.9.3 Stellar: Documented Python Function with Unit Tests**
For reusable analysis components, such as a function to calculate stellar physical parameters from observed properties using specific empirical relations, writing well-documented functions with unit tests is best practice.

```python
# In file: src/stellar_utils.py
import numpy as np
import astropy.units as u

def estimate_stellar_radius(log_g, mass):
    """
    Estimates stellar radius from surface gravity and mass.

    Uses the relation R = sqrt(G * M / g), where g = 10**log_g (in cgs).

    Parameters
    ----------
    log_g : float or np.ndarray
        Logarithm (base 10) of the surface gravity in cgs units (cm/s^2).
    mass : float or np.ndarray or astropy.units.Quantity
        Stellar mass. If not a Quantity, assumed to be in Solar masses.

    Returns
    -------
    astropy.units.Quantity
        Estimated stellar radius in Solar radii (R_sun). Returns NaN if
        inputs are invalid (e.g., negative mass, non-positive g).
    """
    from astropy.constants import G, M_sun, R_sun

    # Input validation and unit handling
    if not isinstance(mass, u.Quantity):
        mass_kg = np.asarray(mass) * M_sun.value # Assume Solar mass if no units
    else:
        mass_kg = mass.to(u.kg).value

    log_g_val = np.asarray(log_g)
    # Calculate g in m/s^2 (convert from log10(cgs))
    g_cgs = 10**log_g_val
    g_mks = g_cgs / 100.0 # cm/s^2 to m/s^2

    # Check for invalid inputs
    if np.any(mass_kg <= 0) or np.any(g_mks <= 0):
        # Return NaN where inputs are invalid, preserving array shape if input was array
        result_shape = np.broadcast(log_g_val, mass_kg).shape
        radius_m = np.full(result_shape, np.nan) * u.m
    else:
        # Calculate radius in meters: R = sqrt(G * M / g)
        radius_m = np.sqrt(G.value * mass_kg / g_mks) * u.m

    # Convert radius to Solar radii and return
    return radius_m.to(u.R_sun)


# In file: tests/test_stellar_utils.py
import pytest # Requires pytest: pip install pytest
import numpy as np
import astropy.units as u
from astropy.constants import R_sun, M_sun, G
from src.stellar_utils import estimate_stellar_radius # Import function to test

def test_estimate_stellar_radius_sun():
    """Test the radius estimate for Sun-like parameters."""
    log_g_sun = np.log10( (G * M_sun / R_sun**2).to(u.cm/u.s**2).value )
    mass_sun = 1.0 * u.M_sun
    estimated_radius = estimate_stellar_radius(log_g_sun, mass_sun)
    # Check if the result is close to 1 R_sun (allow some tolerance)
    np.testing.assert_allclose(estimated_radius.to(u.R_sun).value, 1.0, rtol=1e-6)

def test_estimate_stellar_radius_array():
    """Test with array inputs."""
    log_g_vals = np.array([4.44, 4.0]) # Sun-like, slightly lower g
    mass_vals = np.array([1.0, 1.2]) * u.M_sun
    expected_radii = np.array([1.0, np.sqrt(1.2 / (10**4.0 / 10**4.44))]) # Approx expected R_sun
    estimated_radii = estimate_stellar_radius(log_g_vals, mass_vals)
    np.testing.assert_allclose(estimated_radii.to(u.R_sun).value, expected_radii, rtol=1e-6)

def test_estimate_stellar_radius_invalid_input():
    """Test handling of invalid inputs."""
    assert np.isnan(estimate_stellar_radius(-1.0, 1.0 * u.M_sun)) # Invalid log_g -> g<=0
    assert np.isnan(estimate_stellar_radius(4.0, -1.0 * u.M_sun)) # Invalid mass
    # Test array with some invalid values
    log_g_vals = np.array([4.4, -1.0, 4.0])
    mass_vals = np.array([1.0, 1.0, -1.0]) * u.M_sun
    estimated_radii = estimate_stellar_radius(log_g_vals, mass_vals)
    assert np.isfinite(estimated_radii[0])
    assert np.isnan(estimated_radii[1])
    assert np.isnan(estimated_radii[2])

# To run tests: execute `pytest` in the terminal from the project root directory.
```
**Explanation:** This example shows a Python function (`estimate_stellar_radius`) designed for reuse, placed within a source directory (`src/`). The function includes a detailed docstring explaining its purpose, parameters (with type hints and units), and return value, adhering to good documentation practices (Section 16.2). It incorporates unit handling using `astropy.units` and basic input validation. Crucially, a separate test file (`tests/test_stellar_utils.py`) contains unit tests written using the `pytest` framework. These tests (`test_...` functions) check the function's output against known values (e.g., for the Sun) and verify its behavior with array inputs and invalid inputs. Running `pytest` automatically discovers and executes these tests, providing assurance that the function behaves correctly and that future modifications do not introduce regressions. This combination of documented, modular code and automated testing significantly enhances reliability and reproducibility.

**16.9.4 Exoplanetary: Reproducible TESS Data Fetching (`astroquery`)**
Accessing data for exoplanet studies, such as TESS light curves, should be done programmatically to ensure reproducibility. Relying on manual downloads from web interfaces makes it hard to track exactly which data version or product was used. `astroquery` provides interfaces to archives like MAST.

```python
import numpy as np
# Requires astroquery: pip install astroquery
try:
    from astroquery.mast import Observations
    astroquery_available = True
except ImportError:
    print("astroquery not found, skipping TESS data fetching example.")
    astroquery_available = False
import os
from astropy.time import Time

# --- Define Target and Observation Criteria ---
target_tic_id = "261136679" # Pi Mensae
sector = 1
pipeline = "SPOC" # Science Processing Operations Center pipeline
product_type = "LC" # Light curve files
file_extension = "lc.fits" # Standard TESS light curve file extension

# Define output directory
output_dir = f"tess_data_tic{target_tic_id}_s{sector}"
os.makedirs(output_dir, exist_ok=True)
print(f"Will download TESS data for TIC {target_tic_id}, Sector {sector} to '{output_dir}'")

if astroquery_available:
    try:
        # --- Query MAST for TESS Observations ---
        print("Querying MAST...")
        # Use Observations.query_criteria to find relevant observation records
        obs_table = Observations.query_criteria(
            obs_collection="TESS",
            target_name=target_tic_id,
            sequence_number=sector # Use sequence_number for sector
        )
        if len(obs_table) == 0:
            raise ValueError("No observations found for specified target/sector.")

        print(f"Found {len(obs_table)} observation entries.")
        # Further filter if needed (e.g., based on specific proposal ID, filters etc.)

        # --- Query for Specific Data Products (Light Curves) ---
        print(f"Querying for {pipeline} {product_type} products...")
        # Use Observations.get_product_list to find available data products for the observation(s)
        # Need observation ID(s) from obs_table (e.g., 'obsid' or 'observation_id' column)
        # Use first observation found for simplicity
        obs_id = obs_table['obsid'][0]
        product_list = Observations.get_product_list(obs_id)

        # Filter product list for the desired type and pipeline
        filtered_products = Observations.filter_products(
            product_list,
            productType="SCIENCE",
            productSubGroupDescription=product_type, # LC for light curve
            description=f"*{file_extension}" # Ensure it's the FITS LC file
            # Could add provenance_name=pipeline filter if available/reliable
        )
        # Further filter for SPOC pipeline if needed, checking 'provenance_name' or other columns
        spoc_products = filtered_products[[(pipeline.upper() in name.upper()) if name else False for name in filtered_products['provenance_name']]]


        if len(spoc_products) == 0:
             # Try filtering based on description if provenance name fails
             spoc_products = filtered_products[[('SPOC' in desc) if desc else False for desc in filtered_products['description']]]
             if len(spoc_products) == 0:
                   print("Filtered products before SPOC check:")
                   print(filtered_products['obsid', 'description', 'provenance_name'])
                   raise ValueError(f"No {pipeline} {product_type} products found for obs_id {obs_id}.")


        print(f"Found {len(spoc_products)} {pipeline} {product_type} product(s) to download.")
        print(spoc_products['obsid', 'description', 'provenance_name'])

        # --- Download Data Products ---
        print("Downloading data products...")
        # Use Observations.download_products to download the filtered list
        manifest = Observations.download_products(
            spoc_products['obsid'], # Provide unique observation identifiers
            download_dir=output_dir,
            mrp_only=False # Download the science files
        )
        print(f"Download complete. Manifest:\n{manifest}")

        # Record the query parameters and downloaded file list for provenance
        provenance_info = {
            'query_time': Time.now().iso,
            'target': target_tic_id, 'sector': sector,
            'pipeline': pipeline, 'product_type': product_type,
            'downloaded_files': [row['Local Path'] for row in manifest]
        }
        # Save provenance_info to a file (e.g., JSON) in the output directory
        # import json
        # with open(os.path.join(output_dir, 'download_provenance.json'), 'w') as f:
        #     json.dump(provenance_info, f, indent=2)
        print("\nProvenance information recorded (conceptually).")
        print("Downloaded data is ready for analysis.")

    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
else:
    print("Skipping TESS data fetching example: astroquery unavailable.")

```
**Explanation:** This script demonstrates reproducible data acquisition for TESS light curves using `astroquery.mast`. Instead of manual downloads, it defines the target (TIC ID), desired sector, and data product type (`LC`) programmatically. It uses `Observations.query_criteria` to find the relevant TESS observation records in the MAST archive and `Observations.get_product_list` and `Observations.filter_products` to identify the specific SPOC pipeline light curve files (`lc.fits`). The `Observations.download_products` function then downloads these specified files to a designated directory. By scripting the exact query parameters and download commands, this process ensures that the *same* data products can be reliably retrieved by anyone running the script (assuming data availability in the archive), fulfilling a key requirement for reproducibility (Section 16.8). Saving the query details and downloaded file list provides crucial provenance (Section 16.6).

**16.9.5 Galactic: Script using `pyvo` for VO Data Query**
Accessing data from Galactic plane surveys (e.g., VPHAS+, VVV, Spitzer GLIMPSE) stored in Virtual Observatory (VO) compliant archives can be done reproducibly using `pyvo`. This example shows a script using `pyvo.dal.TAPService` to execute an ADQL query against a hypothetical VO TAP service to retrieve sources from a Galactic plane survey catalog within a specific region and magnitude range.

```python
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
# Requires pyvo: pip install pyvo
try:
    import pyvo as vo
    pyvo_available = True
except ImportError:
    print("pyvo not found, skipping Galactic VO query example.")
    pyvo_available = False

# --- Define VO Service and Query Parameters ---
# URL of the TAP service providing the catalog data
# Replace with a real TAP service URL, e.g., from GAVO, CADC, ESA archives
# Example using a known VizieR TAP service (for a different catalog for demo)
# tap_service_url = "http://tapvizier.u-strasbg.fr/TAPVizieR/tap"
# Use a placeholder URL for demonstration
tap_service_url = "http://example.vo/tap" # <<<< REPLACE WITH REAL TAP SERVICE URL
# Target table name within the TAP service
# Replace with the actual table name for the desired survey catalog
target_table_name = "schema.galactic_survey_dr2" # <<<< REPLACE WITH REAL TABLE NAME

# Define search region (e.g., small area in Galactic plane)
search_coord = SkyCoord(l=30.5*u.deg, b=0.1*u.deg, frame='galactic')
search_radius = 5.0 * u.arcmin

# Define query constraints (e.g., magnitude limit)
mag_limit = 18.0
mag_band = 'r_mag' # Assume column name for r-band magnitude

# --- Construct ADQL Query ---
adql_query = f"""
SELECT TOP 500 -- Limit results for demonstration
  source_id, l, b, {mag_band}, {mag_band}_error -- Adjust column names as needed
FROM
  {target_table_name}
WHERE
  1=CONTAINS(POINT('GALACTIC', l, b),
             CIRCLE('GALACTIC', {search_coord.l.deg}, {search_coord.b.deg}, {search_radius.to(u.deg).value}))
  AND {mag_band} IS NOT NULL AND {mag_band} < {mag_limit}
ORDER BY {mag_band} ASC
"""
print("Constructed ADQL Query:")
print(adql_query)

# --- Execute Query using pyvo ---
if pyvo_available:
    print(f"\nQuerying TAP service at {tap_service_url}...")
    try:
        # Create TAPService instance
        tap_service = vo.dal.TAPService(tap_service_url)

        # Check if service is available (optional but good practice)
        # tap_service.check_availability() # This might raise an error if URL is bad

        # Execute the query synchronously (for smaller result sets)
        # For large queries, use run_async
        # Handle potential connection errors or invalid URLs gracefully
        try:
            results = tap_service.search(adql_query) # Returns a pyvo.dal.TAPResults object
        except Exception as conn_err:
             if "Name or service not known" in str(conn_err) or "Could not resolve host" in str(conn_err):
                   print(f"Error: Could not connect to TAP service at '{tap_service_url}'.")
                   print("Please replace with a valid TAP service URL.")
                   results = None # Set results to None
             else:
                   raise conn_err # Re-raise other connection errors

        if results:
            # Convert results to an Astropy Table
            results_table = results.to_table()
            print(f"\nQuery successful! Retrieved {len(results_table)} sources.")
            print("First 5 rows of the results table:")
            print(results_table[:5])

            # Save results (optional)
            # output_file = 'galactic_vo_query_results.ecsv'
            # results_table.write(output_file, format='ascii.ecsv', overwrite=True)
            # print(f"Results saved to {output_file}")
        else:
             print("Query execution failed or returned no results (potentially due to invalid URL).")


    except ImportError:
        print("Error: pyvo library is required but not found.")
    except Exception as e:
        # Catch VO query errors or other issues
        print(f"An unexpected error occurred during the VO query: {e}")
else:
    print("Skipping Galactic VO query example: pyvo unavailable.")

```
**Explanation:** This script demonstrates reproducible data access from Virtual Observatory (VO) services using `pyvo`. It defines the URL of a target TAP service and the name of the table containing the desired Galactic survey data. It constructs a precise ADQL query to select sources within a specific Galactic coordinate range and magnitude limit. The `pyvo.dal.TAPService` class is used to establish a connection to the service, and the `tap_service.search()` method executes the ADQL query. The results are retrieved as a `pyvo.dal.TAPResults` object, which is easily converted to an `astropy.table.Table` for further analysis. By embedding the exact service URL and ADQL query within the script, the data acquisition step becomes transparent and precisely reproducible (Section 16.8), ensuring that the same dataset can be retrieved consistently. Error handling for connection issues is included.

**16.9.6 Extragalactic: Creating `requirements.txt` for Pipeline**
A complex analysis pipeline, perhaps for measuring galaxy morphologies or fitting SEDs, will depend on numerous Python packages. Ensuring others can run the pipeline requires documenting these dependencies precisely.

```bash
# Example steps in the terminal (within the activated project environment)

# 1. Activate the correct Conda or venv environment for the project
#    conda activate galaxy_morphology_env
#    source venv_morphology/bin/activate

# 2. Ensure all necessary packages are installed using conda or pip
#    pip install numpy astropy scipy matplotlib photutils scikit-image statmorph ...

# 3. Generate the requirements.txt file using pip freeze
pip freeze > requirements.txt

# 4. Inspect the requirements.txt file (optional)
#    cat requirements.txt
#    (It will list all packages in the environment, e.g.:)
#    astropy==5.3.4
#    numpy==1.24.3
#    photutils==1.7.0
#    scikit-image==0.21.0
#    scipy==1.10.1
#    statmorph==0.5.0
#    ... and all dependencies ...

# 5. Add requirements.txt to Git version control
#    git add requirements.txt
#    git commit -m "Add requirements file for dependency management"

# --- How others use it ---
# 1. Clone the Git repository
#    git clone <repository_url>
#    cd <repository_directory>
# 2. Create a new virtual environment (recommended)
#    python -m venv venv_new
#    source venv_new/bin/activate
# 3. Install the exact dependencies from the file
#    pip install -r requirements.txt
```
**Explanation:** This example outlines the crucial workflow for documenting Python package dependencies using `pip freeze`. After activating the specific virtual environment (`conda` or `venv`) where the galaxy morphology pipeline was developed and tested (ensuring all required libraries like `astropy`, `photutils`, `statmorph` are installed), the command `pip freeze > requirements.txt` is executed. This command lists all Python packages currently installed in the active environment, along with their exact version numbers, and saves this list to the `requirements.txt` file. This file serves as a precise blueprint of the software dependencies (Section 16.4). Committing `requirements.txt` to the project's Git repository allows collaborators or future users to easily recreate the identical Python environment by creating a new virtual environment and running `pip install -r requirements.txt`, significantly enhancing the reproducibility of the pipeline across different machines and times.

**16.9.7 Cosmology: Including Version Info in Script Logs**
For long-running analyses or complex pipelines, especially those used for publication results, embedding software version information directly into output log files provides invaluable provenance.

```python
import sys
import logging
import datetime
import platform
# Import libraries used in the analysis
import numpy as np
import scipy
import astropy
# import specific_cosmology_code

# --- Setup Logging ---
log_file = f'cosmology_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler(sys.stdout)]) # Log to file and console

# --- Log System and Version Information ---
logging.info("Starting Cosmology Analysis Pipeline")
logging.info(f"Timestamp: {datetime.datetime.now()}")
logging.info(f"Python Version: {sys.version}")
logging.info(f"Platform: {platform.platform()}")
# Log versions of key libraries
logging.info(f"NumPy Version: {np.__version__}")
logging.info(f"SciPy Version: {scipy.__version__}")
logging.info(f"Astropy Version: {astropy.__version__}")
# Add other critical libraries used in the analysis
# try:
#     import specific_cosmology_code
#     logging.info(f"Specific Code Version: {specific_cosmology_code.__version__}")
# except (ImportError, AttributeError):
#     logging.warning("Could not determine version for specific_cosmology_code.")

# --- Analysis Parameters ---
param1 = 10.5
param2 = 'value_a'
logging.info(f"Input Parameters: param1={param1}, param2='{param2}'")
input_data_file = 'input_cosmo_data.fits'
logging.info(f"Input Data File: {input_data_file}")

# --- Main Analysis Code ---
logging.info("Starting main analysis section...")
try:
    # Placeholder for actual analysis steps
    # data = load_data(input_data_file)
    # results = perform_calculation(data, param1)
    # logging.info(f"Intermediate result: {np.mean(results):.4f}")
    # save_results(results, 'output_cosmo.fits')
    logging.info("Placeholder analysis step completed.")
    result_value = 123.456 # Dummy result

    logging.info(f"Final Result Value: {result_value:.3f}")
    logging.info("Analysis finished successfully.")

except Exception as e:
    logging.error("An error occurred during analysis:", exc_info=True) # Log traceback

# --- End Log ---
logging.info("Closing log file.")

# Example content of the generated .log file:
# 2024-03-15 11:20:30,123 - INFO - Starting Cosmology Analysis Pipeline
# 2024-03-15 11:20:30,123 - INFO - Timestamp: 2024-03-15 11:20:30.123456
# 2024-03-15 11:20:30,123 - INFO - Python Version: 3.10.4 (...)
# 2024-03-15 11:20:30,123 - INFO - Platform: Linux-5.15.0-...-x86_64-with-...
# 2024-03-15 11:20:30,123 - INFO - NumPy Version: 1.24.3
# 2024-03-15 11:20:30,123 - INFO - SciPy Version: 1.10.1
# 2024-03-15 11:20:30,123 - INFO - Astropy Version: 5.3.4
# 2024-03-15 11:20:30,123 - INFO - Input Parameters: param1=10.5, param2='value_a'
# 2024-03-15 11:20:30,123 - INFO - Input Data File: input_cosmo_data.fits
# 2024-03-15 11:20:30,123 - INFO - Starting main analysis section...
# 2024-03-15 11:20:30,123 - INFO - Placeholder analysis step completed.
# 2024-03-15 11:20:30,123 - INFO - Final Result Value: 123.456
# 2024-03-15 11:20:30,123 - INFO - Analysis finished successfully.
# 2024-03-15 11:20:30,123 - INFO - Closing log file.

```
**Explanation:** This Python script demonstrates how to incorporate detailed logging, including crucial software version information, into a computational workflow, enhancing provenance and reproducibility. It uses Python's built-in `logging` module to set up logging to both the console and a timestamped file. At the beginning of the script execution, it logs essential environment information: the current timestamp, Python version, platform details, and critically, the versions of key scientific libraries (`numpy`, `scipy`, `astropy`, and potentially specific cosmology codes) imported using their `__version__` attributes. It also logs important input parameters and filenames used in the analysis. During the (placeholder) analysis steps, intermediate results or progress messages can be logged. If errors occur, they are logged with tracebacks. This practice ensures that the log file associated with a specific analysis run contains a snapshot of the exact software environment (Section 16.4) and parameters used, providing invaluable information for debugging, verifying results, and precise reproduction (Section 16.6).

---

**References**

Allen, A., Momcheva, I., Tollerud, E., et al. (2022). Technical Roadmap for Libraries of Astronomical Python Packages. *Zenodo*. https://doi.org/10.5281/zenodo.7337728
*   *Summary:* This roadmap document for astronomical Python libraries emphasizes the importance of interoperability, sustainability, and implicitly, reproducibility through shared standards and best practices, relevant to Sections 16.1 and 16.7.

Allen, M. G., Schmidt, T., & Kooper, R. (2013). Astrophysics Source Code Library. *Astronomical Data Analysis Software and Systems XXII*, 475, 277. *(Note: ASCL description, pre-2020)*
*   *Summary:* Although pre-2020, this conference proceeding describes the Astrophysics Source Code Library (ASCL), a key registry for finding and citing astronomical software, relevant to code sharing practices (Section 16.7).

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project's community and development model, which inherently promotes Open Science principles (Section 16.1). Its tools are fundamental to many reproducible workflows demonstrated.

Blischak, J. D., Davenport, E. R., & Wilson, G. (2016). A Quick Introduction to Version Control with Git and GitHub. *PLoS Computational Biology, 12*(1), e1004668. https://doi.org/10.1371/journal.pcbi.1004668 *(Note: Foundational Git tutorial, pre-2020)*
*   *Summary:* While pre-2020, this paper provides an excellent introduction to Git and GitHub specifically aimed at scientists. It covers the core concepts and benefits directly relevant to version control for reproducibility (Section 16.3).

Chirigati, F., & Freire, J. (2023). Provenance for Interactive Visualizations. *Synthesis Lectures on Visualization, 9*(1), 1–104. https://doi.org/10.1007/978-3-031-04802-8
*   *Summary:* This recent lecture synthesis, while focused on interactive visualization, covers modern concepts and challenges in capturing and utilizing data provenance (Section 16.6) in computational workflows.

Demleitner, M., Taylor, M., Dowler, P., Major, B., Normand, J., Benson, K., & pylibs Development Team. (2023). pyvo 1.4.1: Fix error in datalink parsing. *Zenodo*. [Data set]. https://doi.org/10.5281/zenodo.7858974
*   *Summary:* This Zenodo record archives a version of `pyvo`. `pyvo` provides the Python interface to Virtual Observatory protocols (TAP, SIA, SSA, DataLink), enabling the reproducible, programmatic data access discussed in Section 16.8 and demonstrated in Example 16.9.5.

Di Tommaso, P., Chatzou, M., Floden, E. W., Barja, P. P., Palumbo, E., & Notredame, C. (2017). Nextflow enables reproducible computational workflows. *Nature Biotechnology, 35*(4), 316–319. https://doi.org/10.1038/nbt.3820 *(Note: Workflow manager paper, pre-2020)*
*   *Summary:* Introduces Nextflow, a popular workflow management system mentioned in Section 16.2. While pre-2020, it represents tools that enhance reproducibility by automating complex pipelines and managing dependencies.

Dowler, P., Demleitner, M., Taylor, M., & Benson, K. (2022). IVOA Recommendation: VOTable Format Definition Version 1.5. *International Virtual Observatory Alliance*. https://www.ivoa.net/documents/VOTable/20221020/REC-VOTable-1.5-20221020.pdf
*   *Summary:* Defines the VOTable standard. VOTable is often used in conjunction with VO protocols like TAP (Section 16.8) for exchanging tabular data in a standardized, interoperable format, aiding reproducibility.

Kluyver, T., Ragan-Kelley, B., Pérez, F., Granger, B., Bussonnier, M., Frederic, J., Kelley, K., Hamrick, J., Grout, J., Corlay, S., Ivanov, P., Avila, D., Abdalla, S., & Willing, C. (2016). Jupyter Notebooks – a publishing format for reproducible computational workflows. *Positioning and Power in Academic Publishing: Players, Agents and Agendas*, 87–90. *(Note: Foundational Jupyter paper, pre-2020)*
*   *Summary:* Although pre-2020, this paper positions Jupyter Notebooks as a format for reproducible workflows. It provides context for the discussion in Section 16.5 comparing notebooks and scripts regarding reproducibility.

Köster, J., & Rahmann, S. (2012). Snakemake—a scalable bioinformatics workflow engine. *Bioinformatics, 28*(19), 2520–2522. https://doi.org/10.1093/bioinformatics/bts480 *(Note: Workflow manager paper, pre-2020)*
*   *Summary:* Introduces Snakemake, another popular workflow management system mentioned in Section 16.2. While pre-2020 and originating in bioinformatics, Snakemake is used in astronomy to automate complex pipelines reproducibly.

Muna, D., Blanco-Cuaresma, S., Ponder, K., Pasham, D., Teuben, P., Williams, P., A.-P., Lim, P. L., Shupe, D., Tollerud, E., Hedges, C., Robitaille, T., D'Eugenio, F., & Astropy Collaboration. (2023). Software Citation in Astronomy: Current Practices and Recommendations from the Astropy Project. *arXiv preprint arXiv:2306.06699*. https://doi.org/10.48550/arXiv.2306.06699
*   *Summary:* This paper discusses the importance and current practices of software citation in astronomy, directly relevant to ensuring reproducibility and giving credit for computational tools used in research, aligning with Sections 16.1 and 16.7.

National Academies of Sciences, Engineering, and Medicine. (2019). *Reproducibility and Replicability in Science*. The National Academies Press. https://doi.org/10.17226/25303
*   *Summary:* A comprehensive report defining and discussing the concepts of reproducibility and replicability across scientific disciplines. Provides essential background and motivation for the principles discussed in Section 16.1.

Perez-Riverol, Y., Gatto, L., Wang, R., Sachsenberg, T., Uszkoreit, J., Leprevost, F. d. V., Fufezan, C., Ternent, T., Eglen, S. J., Brazma, A., Hubbard, S. J., & Vizcaíno, J. A. (2016). Ten Simple Rules for Taking Advantage of Git and GitHub. *PLoS Computational Biology, 12*(7), e1004947. https://doi.org/10.1371/journal.pcbi.1004947 *(Note: Git/GitHub tutorial, pre-2020)*
*   *Summary:* While pre-2020, this provides practical rules for effectively using Git and GitHub in a scientific context, directly relevant to the version control practices discussed in Section 16.3.

Shamir, L., Jones, K. M., Toderici, G., Penner, H., Grier, J. A., Baugh, C., Dickinson, M. E., Estrada, J., Ferres, L., Greene, J. E., Gwyn, S., Hancock, B. A., Johnson, M., Marshall, P. J., Mendes de Oliveira, C., Miller, D. R., Miller, J., Nord, B., Pinto, M., … Simpson, R. J. (2020). Reproducibility in Astrophysical Research Computing. *arXiv preprint arXiv:2006.03033*. https://doi.org/10.48550/arXiv.2006.03033
*   *Summary:* This paper specifically addresses reproducibility challenges within astrophysical research computing, discussing issues like software dependencies, workflow documentation, and data handling. Highly relevant to the entire chapter, especially Sections 16.1, 16.4, 16.5.

Siebert, S., Dyer, B. D., & Hogg, D. W. (2022). Practical challenges in reproducibility for computational astrophysics. *arXiv preprint arXiv:2212.05459*. https://doi.org/10.48550/arXiv.2212.05459
*   *Summary:* Focuses on the practical difficulties encountered when trying to achieve reproducibility in computational astrophysics. Directly informs the discussion on code practices (Section 16.2), environment management (Section 16.4), and provenance (Section 16.6).

Wilkinson, M. D., Dumontier, M., Aalbersberg, Ij. J., Appleton, G., Axton, M., Baak, A., Blomberg, N., Boiten, J.-W., da Silva Santos, L. B., Bourne, P. E., Bouwman, J., Brookes, A. J., Clark, T., Crosas, M., Dillo, I., Dumon, O., Edmunds, S., Evelo, C. T., Finkers, R., … Mons, B. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data, 3*(1), 160018. https://doi.org/10.1038/sdata.2016.18 *(Note: Foundational FAIR principles paper, pre-2020)*
*   *Summary:* The seminal paper defining the FAIR Guiding Principles (Findable, Accessible, Interoperable, Reusable). While pre-2020, these principles are central to Open Science and data sharing best practices discussed in Sections 16.1 and 16.7.
