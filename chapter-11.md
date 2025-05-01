---
# Chapter 11
# Large-Scale Data Analysis II: High-Performance Computing Techniques
---
![imagem](imagem.png)

*This chapter continues the exploration of advanced computational methods required for large-scale astronomical data analysis, shifting focus from Machine Learning algorithms (discussed in Chapter 10) to techniques designed to accelerate computationally intensive tasks and handle problems that exceed the capabilities of single-core processing. It begins by identifying common computational bottlenecks encountered in astronomical research, ranging from complex simulations and large-N body problems to demanding data processing steps like radio interferometric imaging or iterative model fitting in high-dimensional parameter spaces. The core of the chapter delves into strategies for leveraging parallel processing architectures, starting with multi-core Central Processing Units (CPUs). Concepts such as parallelism, concurrency, and shared memory models are introduced, followed by practical discussions of Python libraries like `multiprocessing` and `joblib` for process-based parallelism on a single machine, and an introduction to the Message Passing Interface (MPI) via `mpi4py` for distributed memory parallelism across multiple nodes in High-Performance Computing (HPC) clusters. Distributed computing frameworks like `Dask` and `Ray`, which facilitate scaling standard Python data analysis workflows across cores and clusters, are then examined. The chapter subsequently explores the significant acceleration achievable using massively parallel processors, focusing primarily on Graphics Processing Units (GPUs). Fundamental GPU architecture concepts (CUDA, parallel threads) are discussed, alongside Python libraries like `CuPy` (providing a NumPy-like interface for GPU arrays) and `Numba` (offering just-in-time compilation for CPU and GPU code acceleration). The utilization of GPUs within established Deep Learning frameworks is also highlighted. A brief overview of specialized hardware like Tensor Processing Units (TPUs) is included. Finally, essential practices for identifying performance bottlenecks through code profiling and applying optimization strategies, such as vectorization and efficient memory access patterns, are presented. Throughout the chapter, illustrative examples demonstrate the application of these HPC techniques to accelerate common computational tasks across diverse astronomical domains.*

---

**11.1 Computational Bottlenecks in Astronomy**

While modern computing hardware offers remarkable processing power, many cutting-edge problems in astrophysics push the boundaries of computational feasibility, encountering significant **bottlenecks** that limit the scale, resolution, complexity, or speed of analysis and simulation (Di Matteo et al., 2023; Turk et al., 2019). These bottlenecks can arise from various factors, including the sheer volume of data to be processed, the inherent computational complexity of the algorithms employed, limitations in data movement speed (I/O or network bandwidth), or constraints imposed by memory capacity. Identifying the nature of these bottlenecks is the first step towards selecting appropriate High-Performance Computing (HPC) techniques to overcome them.

Common types of computational bottlenecks in astronomy include:

1.  **CPU-Bound Tasks (Compute-Intensive):** These tasks are limited primarily by the processing speed of the CPU. The algorithm involves a large number of arithmetic operations (floating-point or integer) relative to the amount of data being processed or moved.
    *   **Large-N Simulations:** Simulations involving gravitational interactions between many particles (N-body simulations for cosmology, galaxy dynamics, star cluster evolution) scale computationally as $O(N^2)$ (direct summation) or $O(N \log N)$ (using tree codes or Particle-Mesh methods), becoming extremely compute-intensive for large $N$ (millions to billions of particles).
    *   **Hydrodynamic/Magnetohydrodynamic (MHD) Simulations:** Solving the equations of fluid dynamics or MHD on large grids or with many particles (e.g., using Smoothed Particle Hydrodynamics - SPH, or Adaptive Mesh Refinement - AMR codes) involves complex calculations at each time step for each grid cell or particle, often requiring significant computational time (Ntormousi & Teyssier, 2022).
    *   **Radiative Transfer Calculations:** Modeling the transport of photons through complex media (e.g., stellar atmospheres, nebulae, accretion disks, intergalactic medium) often involves solving the integro-differential radiative transfer equation, which can be computationally demanding, especially in multiple dimensions or with detailed microphysics (e.g., non-LTE effects).
    *   **Complex Model Fitting (e.g., MCMC):** Parameter estimation using algorithms like Markov Chain Monte Carlo (MCMC) or Nested Sampling (Chapter 12) often requires evaluating a computationally expensive theoretical model or likelihood function hundreds of thousands or millions of times to adequately explore high-dimensional parameter spaces (Buchner, 2021).
    *   **Signal Processing:** Computationally intensive algorithms like Fast Fourier Transforms (FFTs) applied to very large datasets (e.g., high-resolution radio data cubes, time series with billions of points) or complex correlation/deconvolution procedures.
    *   **Iterative Algorithms:** Many scientific algorithms involve iterative refinement (e.g., iterative PSF subtraction, image deconvolution, solving implicit equations in simulations), where each iteration can be computationally costly.

2.  **Memory-Bound Tasks:** These tasks are limited by the amount of Random Access Memory (RAM) available on the system or the speed at which data can be loaded from RAM into the CPU caches and registers. The computation itself might be fast, but performance stalls while waiting for data.
    *   **Processing Large Single Images/Cubes:** Loading and performing operations on extremely large images or data cubes (e.g., from wide-field imagers, IFUs, or simulations) that exceed the available RAM requires out-of-core processing techniques or distributed memory approaches.
    *   **Large Matrix Operations:** Linear algebra operations (e.g., solving large systems of equations, matrix decompositions) involving matrices that do not fit comfortably within CPU caches suffer from memory latency.
    *   **Irregular Memory Access:** Algorithms involving frequent, non-sequential access to large data structures (e.g., graph traversal, sparse matrix operations, some types of neighbor searches) can be limited by memory bandwidth and latency as data needs to be fetched from different memory locations.

3.  **I/O-Bound Tasks:** Performance is limited by the speed of reading data from or writing data to persistent storage (hard disk drives - HDDs, solid-state drives - SSDs, network file systems - NFS).
    *   **Reading/Writing Large Files:** Processing pipelines that sequentially read large raw data files, perform relatively quick operations, and write large intermediate or final product files can be bottlenecked by disk I/O speed.
    *   **Database Queries:** Complex queries against very large databases (Section 10.2) can be limited by the disk speed of the database server, although indexing aims to mitigate this.
    *   **Checkpointing:** Large simulations often periodically save their state (checkpointing) to disk for fault tolerance or restarts. Writing massive checkpoint files can consume significant time and I/O bandwidth.

4.  **Network-Bound Tasks:** In distributed computing environments (clusters, grids, cloud), performance can be limited by the speed and latency of the network connecting different processing nodes or connecting nodes to storage.
    *   **Data Transfer:** Moving large datasets between compute nodes or between compute nodes and remote storage systems.
    *   **Communication Overhead (MPI):** Parallel algorithms using the Message Passing Interface (MPI) require frequent communication between processes running on different nodes. High latency or low bandwidth networks can severely degrade the performance of communication-intensive MPI applications (Section 11.2.4).
    *   **Distributed Filesystems:** Accessing data stored on distributed or parallel file systems (like Lustre or GPFS) involves network communication.

Identifying the primary bottleneck for a given task is crucial for choosing the right acceleration strategy. **Profiling tools** (Section 11.6) are essential for measuring where computation time is spent or where delays occur. CPU-bound tasks benefit most directly from parallelization across multiple CPU cores or offloading to accelerators like GPUs. Memory-bound tasks may require code restructuring for better data locality, using larger memory nodes, or adopting distributed memory approaches. I/O-bound tasks might be improved by using faster storage (SSDs, parallel file systems), optimizing file access patterns, or using data compression. Network-bound tasks necessitate faster network infrastructure or algorithms designed to minimize communication. The techniques discussed in this chapter primarily focus on addressing CPU-bound and, to some extent, memory-bound problems through parallel and accelerated computing.

**11.2 Multi-Core CPU Parallelization Strategies**

Modern Central Processing Units (CPUs) typically contain multiple independent processing units, known as **cores**, on a single chip. A typical laptop or desktop CPU might have 4 to 16 cores, while server-grade CPUs can have dozens of cores. **Parallel computing** on multi-core CPUs aims to speed up computationally intensive tasks by dividing the work among these available cores and executing different parts of the computation simultaneously (Asanović et al., 2006; Reinders, 2007). This is particularly effective for CPU-bound problems (Section 11.1) that can be broken down into independent or semi-independent sub-tasks. Python offers several built-in and third-party libraries to facilitate multi-core parallelization.

*   **11.2.1 Concepts: Parallelism, Concurrency, Shared Memory**
    It's important to distinguish between parallelism and concurrency:
    *   **Parallelism:** Executing multiple computations *simultaneously* on different hardware units (e.g., multiple CPU cores, multiple GPUs, multiple nodes in a cluster). The goal is primarily speedup – reducing the total wall-clock time to complete a task.
    *   **Concurrency:** Managing multiple tasks that may *overlap* in time, allowing them to make progress independently, even if not executing strictly simultaneously (e.g., handling multiple user requests on a web server, managing I/O operations while computation proceeds). The goal is often responsiveness or efficient resource utilization. Concurrency can be achieved on a single core through time-slicing or event loops (e.g., using Python's `asyncio`). This chapter focuses primarily on *parallelism* for computational speedup.

    Multi-core CPUs typically operate within a **shared memory** architecture. All cores on the same CPU chip (and often within the same physical compute node) have access to the same main system RAM. This simplifies programming compared to distributed memory systems (like clusters), as processes or threads running on different cores can potentially access and modify the same data structures in memory directly. However, shared memory access also introduces challenges:
    *   **Race Conditions:** If multiple processes/threads attempt to read and write to the same memory location concurrently without proper synchronization, the final result can depend unpredictably on the exact timing of operations, leading to incorrect results.
    *   **Synchronization Overhead:** Mechanisms like locks, mutexes, or semaphores are needed to coordinate access to shared resources and prevent race conditions. However, these synchronization primitives introduce overhead and can become bottlenecks themselves if contention is high.
    *   **Cache Coherency:** Each core typically has its own local cache memory. Ensuring that all cores have a consistent view of data that might reside in multiple caches (cache coherency) is handled by hardware but adds complexity and potential performance implications.
    *   **Global Interpreter Lock (GIL) in CPython:** The standard CPython interpreter uses a Global Interpreter Lock (GIL), which allows only *one thread* to execute Python bytecode at any given time within a single process, even on multi-core systems. This means that using Python's built-in `threading` module for CPU-bound tasks typically does *not* achieve true parallelism and offers little speedup (it's primarily useful for concurrency involving I/O-bound tasks where threads can release the GIL while waiting). To achieve true CPU parallelism in Python, one must typically use multiple *processes* instead of threads, or utilize libraries that bypass the GIL (like `NumPy`/`SciPy` operations that release the GIL during C-level computations, or tools like `Numba`).

*   **11.2.2 Python's `multiprocessing` Module**
    The `multiprocessing` module in Python's standard library provides a way to achieve true parallelism for CPU-bound tasks by creating and managing separate *processes* instead of threads. Each process runs in its own memory space and has its own Python interpreter, thus bypassing the GIL limitation.
    *   **Process Creation:** Processes can be created using the `multiprocessing.Process` class.
    *   **Pool of Workers:** A common and convenient pattern uses `multiprocessing.Pool`. A `Pool` object manages a fixed number of worker processes. Tasks can be submitted to the pool using methods like:
        *   `pool.map(func, iterable)`: Applies the function `func` to each item in the `iterable` in parallel, distributing the work among the pool workers, and returns a list of results in order. Simple but requires the entire iterable to be available upfront.
        *   `pool.imap(func, iterable)`: Similar to `map` but returns an iterator, potentially more memory-efficient for large iterables as results are yielded as they complete.
        *   `pool.apply_async(func, args)`: Submits a single task asynchronously. Returns an `AsyncResult` object used to retrieve the result later. Useful for heterogeneous tasks.
        *   `pool.starmap(func, iterable_of_tuples)`: Like `map`, but function arguments are provided as tuples in the iterable.
    *   **Data Transfer:** Since processes have separate memory spaces, data (arguments passed to functions, results returned) must be transferred between the main process and worker processes. This typically involves **serialization** (converting Python objects into a byte stream using libraries like `pickle`) and Inter-Process Communication (IPC) mechanisms (e.g., pipes, queues). This data transfer incurs overhead, which can become significant if large amounts of data are passed back and forth frequently. It's often more efficient to pass only necessary data or have workers load data independently if possible.
    *   **Use Cases:** Ideal for "embarrassingly parallel" tasks where a computation can be easily broken down into many independent sub-problems that operate on different data segments (e.g., processing multiple independent files, running the same analysis on different parameter sets, performing independent calculations within a loop).
    *   **Limitations:** Process creation and IPC overhead can be substantial for very short tasks. Managing shared state between processes requires explicit use of synchronization primitives (`Lock`, `Queue`, `Value`, `Array` provided by `multiprocessing`) or specialized managers, adding complexity.

*   **11.2.3 Parallel Execution with `joblib`**
    The **`joblib`** library provides a higher-level, often simpler interface for common parallel processing patterns in Python, particularly for embarassingly parallel tasks often found in scientific computing loops (Varoquaux et al., n.d.). It aims to make parallel execution straightforward with minimal code changes.
    *   **Core Functionality (`Parallel`, `delayed`):** The main components are the `Parallel` class and the `delayed` function.
        *   `delayed(func)`: Wraps a function call, delaying its execution.
        *   `Parallel(n_jobs=<N>)`: Creates a context manager for parallel execution. `n_jobs` specifies the number of worker processes (or threads) to use; `n_jobs=-1` typically uses all available CPU cores.
        *   The common pattern involves creating a list of delayed function calls within a generator expression or list comprehension and passing this list to the `Parallel` object for execution:
            `results = Parallel(n_jobs=-1)(delayed(my_function)(i) for i in range(N))`
    *   **Backends:** `joblib` supports different backends for parallel execution, including `loky` (default, robust process-based backend, avoids GIL), `multiprocessing` (uses the standard module), and `threading` (subject to GIL for CPU-bound tasks).
    *   **Advantages:** Often requires minimal changes to existing serial code (primarily wrapping the function call in `delayed` and the loop in `Parallel`). Handles data transfer and worker management transparently. Includes optimizations like memory mapping for large NumPy arrays passed between processes (`memmapping`).
    *   **Integration:** Widely used within the scientific Python ecosystem, particularly by `scikit-learn` for parallelizing model training and cross-validation.
    *   **Use Cases:** Excellent for parallelizing simple `for` loops where each iteration is independent and computationally significant enough to outweigh the overhead.

*   **11.2.4 Message Passing Interface (MPI) Concepts (`mpi4py`)**
    For large-scale parallel computations spanning multiple compute nodes in an HPC cluster (a **distributed memory** environment where each node has its own private memory), the **Message Passing Interface (MPI)** is the de facto standard communication protocol (Gropp et al., 1999; Forum, 1994). MPI defines a library of functions that allow processes running on different nodes to explicitly send and receive messages, enabling data exchange and coordination.
    *   **Programming Model (SPMD):** MPI applications typically follow a Single Program, Multiple Data (SPMD) model. The same program code is launched simultaneously on multiple processor cores (often across different nodes), creating multiple MPI processes (called "ranks"). Each rank is assigned a unique integer ID (from 0 to $N_{ranks}-1$), and the total number of ranks ($N_{ranks}$) is known. The program code uses conditional logic (e.g., `if rank == 0: ... else: ...`) to differentiate the tasks performed by different ranks (e.g., rank 0 might handle I/O or distribute work, while other ranks perform calculations).
    *   **Communication:** MPI provides functions for:
        *   **Point-to-Point Communication:** Sending (`MPI.Send`) and receiving (`MPI.Recv`) data explicitly between two specific ranks. Operations can be blocking (wait for completion) or non-blocking (initiate communication and check later).
        *   **Collective Communication:** Operations involving a group of ranks simultaneously. Examples include:
            *   `MPI.Bcast`: Broadcast data from one rank (root) to all other ranks in a communicator group.
            *   `MPI.Scatter`: Distribute different chunks of an array from the root rank to all other ranks.
            *   `MPI.Gather`: Collect data chunks from all ranks onto the root rank.
            *   `MPI.Reduce`: Combine data from all ranks using a specified operation (e.g., sum, max, min) and place the result on the root rank.
            *   `MPI.Allreduce`: Like `Reduce`, but the result is distributed back to all ranks.
            *   `MPI.Barrier`: Synchronizes all ranks in a group; processes wait until all have reached the barrier.
    *   **`mpi4py`:** Provides Python bindings for standard MPI implementations (like Open MPI, MPICH), allowing Python programs to leverage MPI for distributed computing (Dalcin et al., 2011). It exposes MPI functionality through a Pythonic interface, typically operating on NumPy arrays or pickleable Python objects. Running an `mpi4py` script usually involves using a command like `mpiexec -n <num_processes> python your_script.py`.
    *   **Advantages:** The standard for large-scale distributed memory parallelism on HPC clusters. Offers fine-grained control over communication patterns. Highly scalable to thousands or millions of cores. Enables solving problems that far exceed the memory capacity of a single node.
    *   **Disadvantages:** MPI programming has a steeper learning curve than shared-memory parallelism using `multiprocessing` or `joblib`. Requires careful explicit management of data distribution and communication, which can be complex and prone to deadlocks if not implemented correctly. Performance is highly sensitive to network latency and bandwidth. Requires an MPI library to be installed on the system/cluster.

Choosing the appropriate CPU parallelization strategy depends on the problem structure ("embarrassingly parallel" vs. communication-intensive), the scale of the computation (single multi-core machine vs. multi-node cluster), the amount of data transfer required, and the programmer's familiarity with different paradigms. `joblib` offers simplicity for common loop parallelization, `multiprocessing` provides more flexibility on a single machine, and `mpi4py` enables large-scale distributed computing on HPC systems.

**11.3 Distributed Computing Frameworks**

While MPI provides low-level control for distributed computing, higher-level frameworks have emerged to simplify the development and execution of parallel and distributed applications in Python, particularly for data science and machine learning workloads. These frameworks often aim to parallelize existing familiar APIs (like NumPy, Pandas, Scikit-learn) across multiple cores on a single machine or distributed across multiple nodes in a cluster, hiding much of the complexity of task scheduling, data movement, and fault tolerance from the user. Two prominent examples are Dask and Ray.

*   **11.3.1 `Dask`: Scaling NumPy, Pandas, Scikit-Learn**
    **Dask** is a flexible parallel computing library for Python that scales familiar NumPy array, Pandas DataFrame, and increasingly Scikit-learn APIs to larger-than-memory datasets and distributed environments (Dask Development Team, 2016; Rocklin, 2015). Dask achieves this through two main components:
    1.  **Task Scheduling:** Dask represents computations as **task graphs** (Directed Acyclic Graphs - DAGs), where nodes represent operations (e.g., NumPy function calls, Pandas operations) and edges represent data dependencies. Dask includes dynamic task schedulers optimized for different environments:
        *   **Single-Machine Schedulers:** Use threads (`dask.threaded`) or processes (`dask.multiprocessing`) to execute tasks in parallel on a local machine, suitable for leveraging multi-core CPUs.
        *   **Distributed Scheduler (`dask.distributed`):** Enables scaling computations across multiple machines (nodes) in a cluster. It involves a central scheduler coordinating work and multiple worker processes spread across the cluster nodes performing the actual computations. This allows handling datasets that don't fit into the memory of a single machine and leveraging the combined compute power of the cluster.
    2.  **Parallel Collections:** Dask provides high-level data structures that mimic popular libraries but operate lazily and in parallel using the task schedulers:
        *   **Dask Array:** Implements a large subset of the NumPy `ndarray` interface but operates on arrays partitioned into smaller NumPy arrays (chunks). Operations on Dask arrays generate task graphs that are executed lazily only when a result is explicitly requested (e.g., via `.compute()`). This allows operating on arrays larger than available RAM, as only necessary chunks are loaded into memory at any given time. Ideal for large N-dimensional datasets like image cubes or simulation outputs.
        *   **Dask DataFrame:** Implements a large part of the Pandas `DataFrame` interface but operates on DataFrames partitioned into smaller Pandas DataFrames (often partitioned by row). Similar to Dask Array, operations are lazy and generate task graphs, enabling analysis of tabular datasets larger than memory. Suitable for large astronomical catalogs.
        *   **Dask Bag:** A parallel list implementation suitable for unstructured or semi-structured data (e.g., processing lists of files, JSON records).
        *   **Dask-ML:** Provides parallel and distributed implementations of some Scikit-learn algorithms (e.g., preprocessing, cross-validation, certain estimators like linear models, K-Means, and integration with XGBoost/LightGBM) that can operate on Dask arrays or DataFrames, enabling ML on large datasets.

    **Advantages:** Provides a relatively seamless transition for users familiar with NumPy/Pandas/Scikit-learn. Handles task scheduling, data locality optimization, and parallel execution across cores or clusters largely transparently. Enables working with datasets larger than local RAM. Integrates well with the existing Python scientific ecosystem. `dask.distributed` provides sophisticated diagnostics and cluster management capabilities.
    **Disadvantages:** Introduces some overhead compared to direct NumPy/Pandas operations on in-memory data. Lazy evaluation can sometimes make debugging more complex. Performance can depend on optimal chunking strategies. Not all NumPy/Pandas operations are fully implemented or efficiently parallelized. Communication costs can still be a factor in distributed environments, though Dask attempts to minimize them.
    **Use Cases in Astronomy:** Processing large image mosaics or data cubes, analyzing massive catalogs (e.g., Gaia, LSST), parallelizing custom analysis pipelines involving NumPy/Pandas operations, scaling certain ML workflows.

*   **11.3.2 `Ray`: Brief Overview**
    **Ray** is another popular open-source framework for building distributed applications in Python, particularly focused on scaling ML workloads and reinforcement learning, but also applicable to general distributed computing tasks (Moritz et al., 2018; Ray Team, n.d.). Ray provides a simpler, more general API for distributed computing compared to Dask's focus on mimicking specific data structures.
    *   **Core Concepts:** Ray allows users to easily parallelize Python functions and classes:
        *   **Remote Functions (`@ray.remote`):** Decorating a Python function with `@ray.remote` turns it into a remote task that can be executed asynchronously on any available worker process in the Ray cluster. Calls to the function return "futures" (object IDs), and results can be retrieved later using `ray.get()`.
        *   **Remote Actors (`@ray.remote` class):** Decorating a Python class turns it into an actor, a stateful service that can be instantiated on a worker process and whose methods can be invoked remotely. Actors are useful for maintaining state across parallel tasks (e.g., holding a shared model, managing resources).
    *   **Task Scheduling:** Ray manages a cluster of nodes and includes a distributed scheduler that efficiently assigns tasks and actors to available resources, handling object serialization and data transfer.
    *   **Ecosystem (`Ray AI Runtime` - AIR):** Ray has a growing ecosystem of higher-level libraries built on its core functionality, aimed specifically at distributed ML tasks:
        *   `Ray Data`: Distributed data loading and preprocessing.
        *   `Ray Train`: Distributed training for various ML frameworks (TensorFlow, PyTorch, XGBoost, Scikit-learn).
        *   `Ray Tune`: Scalable hyperparameter tuning.
        *   `Ray Serve`: Scalable model serving.
    *   **Advantages:** Offers a more general and potentially simpler API for arbitrary distributed tasks compared to Dask's collection-focused approach. Provides strong support for stateful distributed applications via actors. Growing ecosystem specifically tailored for end-to-end distributed ML.
    *   **Disadvantages:** Less direct integration with existing NumPy/Pandas APIs compared to Dask (though libraries like Modin build Pandas on Ray). Might have a slightly steeper learning curve for users solely focused on data analysis tasks compared to Dask Arrays/DataFrames.
    *   **Use Cases in Astronomy:** Large-scale hyperparameter tuning for ML models, distributed training of complex deep learning models, implementing custom distributed simulation or analysis workflows, building scalable inference services.

Both Dask and Ray provide powerful abstractions over lower-level parallelization tools like `multiprocessing` and MPI, significantly simplifying the development of scalable Python applications for handling large astronomical datasets and computationally demanding analyses on multi-core machines and clusters. The choice between them often depends on whether the primary goal is scaling existing NumPy/Pandas/Scikit-learn workflows (favoring Dask) or building more general distributed applications or complex ML pipelines (where Ray might offer advantages).

**11.4 Acceleration with Manycore Processors (GPUs)**

While multi-core CPUs offer parallelism, **Graphics Processing Units (GPUs)** represent a different architectural paradigm, featuring hundreds or thousands of simpler processing cores designed for massively parallel computations. Originally developed for graphics rendering, GPUs have evolved into powerful general-purpose parallel processors (GPGPUs) highly effective at accelerating computationally intensive, data-parallel tasks common in scientific computing, including many found in astronomy (Owens et al., 2008; Nvidia Corporation, n.d.; Wilt, 2013). Tasks that involve applying the same operation independently to many data elements (e.g., element-wise array operations, linear algebra, image processing filters, training deep neural networks) can often achieve speedups of 10x to 100x or more when executed on a GPU compared to a multi-core CPU.

*   **11.4.1 GPU Architecture Fundamentals: CUDA, Parallel Threads**
    Understanding the basic architecture helps in utilizing GPUs effectively:
    *   **Many Simple Cores:** GPUs contain a large number of relatively simple Arithmetic Logic Units (ALUs) grouped into **Streaming Multiprocessors (SMs)**.
    *   **SIMT Execution:** GPUs typically employ a **Single Instruction, Multiple Thread (SIMT)** execution model. Within an SM, groups of threads (often 32 threads, called a "warp" in NVIDIA terminology) execute the same instruction simultaneously, but potentially operate on different data elements. This model is highly efficient for data-parallel algorithms where the same operation is applied across large arrays.
    *   **Memory Hierarchy:** GPUs have their own dedicated high-bandwidth memory (e.g., GDDR or HBM). Data must be explicitly transferred from the host (CPU) RAM to the GPU device memory before the GPU can operate on it, and results must be transferred back. This data transfer over the PCIe bus can be a significant bottleneck if not managed carefully. GPUs also have smaller, faster on-chip memory structures like shared memory (accessible by threads within the same SM block) and registers (private to each thread), which are crucial for performance optimization.
    *   **CUDA (Compute Unified Device Architecture):** Developed by NVIDIA, CUDA is the dominant parallel computing platform and programming model for NVIDIA GPUs. It provides APIs (primarily C/C++, but with wrappers for Python, Fortran, etc.) allowing developers to write "kernels" – functions that execute in parallel across many GPU threads. CUDA organizes threads into grids of blocks, mapping logically onto the GPU hardware. OpenCL is an open standard alternative to CUDA, but CUDA has wider adoption and a more mature ecosystem, especially in scientific computing and deep learning.

*   **11.4.2 `CuPy`: NumPy-like Interface for CUDA GPUs**
    Writing raw CUDA kernels can be complex. **CuPy** is a crucial Python library that provides a NumPy-compatible array interface specifically for NVIDIA GPUs (Okuta et al., 2017; CuPy Developers, n.d.).
    *   **`cupy.ndarray`:** CuPy implements a `cupy.ndarray` object that mirrors the API of `numpy.ndarray`. Arrays created with `cupy` reside directly in the GPU device memory.
    *   **NumPy Compatibility:** A large subset of NumPy functions and operations are implemented in CuPy to operate directly on GPU arrays. This allows users to accelerate existing NumPy code by simply replacing `numpy` imports and array creation calls with `cupy` equivalents (e.g., `import cupy as cp; x_gpu = cp.array(x_cpu)`).
    *   **GPU Execution:** When operations are performed on `cupy.ndarray` objects, CuPy automatically compiles and executes corresponding CUDA kernels on the GPU behind the scenes, leveraging the GPU's massive parallelism.
    *   **Data Transfer:** Explicit transfer between host (NumPy) and device (CuPy) memory is handled via functions like `cp.asarray(numpy_array)` (CPU to GPU) and `cp.asnumpy(cupy_array)` or `cupy_array.get()` (GPU to CPU). Minimizing these transfers is key to performance.
    *   **Custom Kernels:** CuPy also allows users to define and compile custom CUDA kernels directly within Python using `cupy.RawKernel` for performance-critical sections not covered by standard functions.
    **Use Cases:** Accelerating array-based computations, linear algebra, FFTs, random number generation, and other tasks heavily reliant on NumPy operations, especially when dealing with large arrays that fit within GPU memory. Example 11.7.3 demonstrates its use.

*   **11.4.3 `Numba`: Just-in-Time Compilation for CPU/GPU**
    **Numba** is a just-in-time (JIT) compiler for Python that translates a subset of Python and NumPy code into fast machine code, often achieving performance comparable to C or Fortran without requiring explicit code rewriting in those languages (Lam et al., 2015; Numba Developers, n.d.).
    *   **JIT Compilation:** Numba uses decorators (like `@numba.jit` or `@numba.vectorize`) to flag Python functions for compilation. When the function is first called, Numba analyzes the input types and compiles optimized machine code specific to those types using the LLVM compiler infrastructure. Subsequent calls use the compiled version directly.
    *   **NumPy Awareness:** Numba understands many NumPy array operations and can often generate highly optimized, parallel code (utilizing CPU SIMD instructions or multi-threading implicitly) for numerical loops operating on arrays.
    *   **GPU Acceleration (`@numba.cuda.jit`):** Crucially, Numba also supports compiling functions specifically for execution on NVIDIA GPUs using the `@numba.cuda.jit` decorator. This allows writing GPU kernels directly in Python syntax (a restricted subset), providing fine-grained control over thread indexing, shared memory usage, and atomic operations, similar to writing raw CUDA C++ but with Python syntax. Numba handles the compilation to PTX (GPU assembly) and kernel launch.
    *   **Advantages:** Can significantly accelerate pure Python loops and numerical code involving NumPy arrays with minimal code changes (just adding decorators). Provides a Pythonic way to write custom GPU kernels for tasks not directly covered by CuPy or other libraries. Offers both CPU and GPU acceleration capabilities within a single framework.
    *   **Disadvantages:** Compiles only a subset of Python/NumPy; performance gains depend heavily on whether the code fits Numba's optimization capabilities (e.g., typed numerical loops). GPU programming with `@numba.cuda.jit` still requires understanding CUDA concepts (threads, blocks, memory spaces). JIT compilation adds a small overhead on the first function call.
    **Use Cases:** Accelerating computationally intensive loops in custom analysis functions, implementing custom image processing filters or algorithms on CPU or GPU, writing specialized numerical kernels where fine-grained control is needed. Examples 11.7.1 and 11.7.6 demonstrate its use.

*   **11.4.4 GPU Utilization in Deep Learning Frameworks**
    Deep Learning model training (Section 10.5) is one of the most prominent applications of GPUs in scientific computing today. Training large neural networks involves massive numbers of matrix multiplications, convolutions, and element-wise operations, which are inherently data-parallel and map extremely well to GPU architectures.
    *   **Framework Integration:** Deep Learning frameworks like **TensorFlow** (via `tf.device('/GPU:0')`) and **PyTorch** (via `.to('cuda')`) provide seamless integration with NVIDIA GPUs using CUDA (and underlying libraries like cuDNN for optimized deep learning primitives).
    *   **Automatic Parallelization:** These frameworks automatically handle the distribution of computations across the GPU's many cores, the management of GPU memory, and the necessary data transfers between host and device memory. Users typically only need to ensure their data tensors and model parameters are placed on the GPU device.
    *   **Performance Gains:** Training deep learning models on GPUs typically results in speedups of one to several orders of magnitude compared to CPU-only training, making it feasible to train complex models on large astronomical datasets (images, spectra, simulations) in reasonable timescales. Example 11.7.7 involves GPU training.

GPUs offer a powerful pathway for accelerating suitable data-parallel workloads in astronomy. Libraries like CuPy provide a high-level NumPy-like interface, Numba enables JIT compilation and custom GPU kernel writing in Python, and Deep Learning frameworks automatically leverage GPU power for model training. Effective GPU utilization requires understanding data locality, minimizing host-device transfers, and choosing algorithms that map well to the GPU's massively parallel SIMT architecture.

**11.5 Introduction to Tensor Processing Units (TPUs)**

Beyond GPUs, Google has developed specialized hardware accelerators called **Tensor Processing Units (TPUs)** specifically designed to accelerate large-scale Machine Learning computations, particularly those involving large matrix multiplications and neural network inference and training (Jouppi et al., 2017). While GPUs are general-purpose parallel processors, TPUs have hardware optimized for the specific types of operations dominant in deep learning.
*   **Architecture:** TPUs feature large matrix multiply units (MXUs) capable of performing tens or hundreds of thousands of multiply-accumulate operations in parallel with high efficiency and lower precision (often `bfloat16`). They are designed for high throughput and power efficiency for ML workloads.
*   **Connectivity:** TPUs are often deployed in "pods" – large interconnected clusters of TPU devices – enabling massive distributed training scenarios.
*   **Software Ecosystem:** TPUs are primarily accessible through Google Cloud Platform and are integrated with frameworks like TensorFlow, PyTorch, and JAX. Programming typically involves using high-level APIs within these frameworks, which compile the computations to run efficiently on the TPU hardware.
*   **Use Cases in Astronomy:** While less ubiquitous than GPUs in individual research groups or university clusters, TPUs are increasingly used via cloud platforms for training extremely large deep learning models on massive astronomical datasets (e.g., large image classification models, generative models for simulations) where their specialized architecture and scalability can offer significant performance advantages over GPU clusters for specific ML tasks (e.g., Modi et al., 2021; Hausen & Robertson, 2020). Access usually involves utilizing cloud computing resources.

TPUs represent another frontier in hardware acceleration, highly specialized for ML, and offer potential for tackling the most demanding deep learning challenges in astronomy, often through cloud-based platforms.

**11.6 Code Profiling and Optimization**

Writing parallel or accelerated code is only part of the story; achieving significant speedups often requires careful **profiling** to identify performance bottlenecks and applying targeted **optimization** strategies. Simply parallelizing code without understanding where the time is actually spent can lead to disappointing results if the bottleneck lies elsewhere (e.g., I/O, memory access, serial sections).

**Profiling:** Measuring the execution time and resource usage of different parts of a program.
*   **Basic Timing:** Simple methods involve using Python's `time` module (`time.time()` or `time.perf_counter()`) to measure the wall-clock time taken by specific code blocks or functions. Useful for quick checks but doesn't provide detailed breakdowns.
*   **CPU Profilers (`cProfile`, `line_profiler`):**
    *   **`cProfile`:** Python's built-in profiler. It measures the time spent within each function call, the number of calls, and cumulative time. It provides a high-level overview of which functions dominate execution time. Output can be analyzed using the `pstats` module.
    *   **`line_profiler`:** Provides more granular information by measuring the time spent on *each individual line* within specified functions (decorated with `@profile`). This is extremely useful for pinpointing slow lines within computationally intensive functions, such as loops or specific calculations. Requires installing the `line_profiler` package.
*   **Memory Profilers (`memory_profiler`):** Help diagnose memory bottlenecks by tracking the memory consumption of a process over time or measuring the memory usage of specific functions line-by-line (using `@profile` decorator after installation). Crucial for identifying memory leaks or unexpectedly large memory allocations.
*   **GPU Profilers (e.g., `nvprof`, Nsight Systems/Compute):** NVIDIA provides command-line tools (`nvprof` - older, `nsys`/`ncu` - newer) and visual profilers (Nsight Systems, Nsight Compute) for detailed performance analysis of CUDA applications running on GPUs. These tools can measure kernel execution times, memory transfer times, GPU utilization, occupancy, memory throughput, identify pipeline stalls, and provide detailed performance metrics and guidance for optimization. Profiling GPU code is essential for identifying bottlenecks related to kernel performance, memory access patterns, or host-device data transfers.
*   **Dask/Ray Dashboards:** Distributed frameworks like Dask (`dask.distributed`) and Ray often come with web-based diagnostic dashboards that provide real-time insights into task scheduling, worker utilization, memory usage across the cluster, and data transfer patterns, aiding in debugging and optimizing distributed applications.

**Optimization Strategies:** Once bottlenecks are identified via profiling, various optimization techniques can be applied:
*   **Algorithmic Optimization:** Replacing an inefficient algorithm with a more efficient one (e.g., using an $O(N \log N)$ algorithm instead of $O(N^2)$) often yields the largest performance gains.
*   **Vectorization (NumPy):** Replacing explicit Python `for` loops over array elements with optimized, C-level NumPy array operations (ufuncs, slicing, broadcasting). NumPy operations often release the GIL and utilize optimized CPU instructions (SIMD), providing significant speedups over pure Python loops. This is the most fundamental optimization for numerical Python code.
*   **Efficient Memory Access:** Arranging computations to access memory sequentially (maximizing cache utilization) rather than randomly can dramatically improve performance, especially for memory-bound tasks. Understanding array storage order (C/row-major vs. Fortran/column-major) can be important.
*   **Just-in-Time Compilation (`Numba`):** As discussed (Section 11.4.3), using Numba (`@jit`) to compile critical Python loops or numerical functions to machine code can provide substantial speedups. `@numba.vectorize` can create efficient NumPy ufuncs from scalar functions.
*   **Parallelization (CPU/GPU):** Applying the techniques discussed earlier (`multiprocessing`, `joblib`, `mpi4py`, `dask`, `cupy`, `numba.cuda`) to execute computationally intensive parts in parallel.
*   **Minimize Data Transfers (GPU/MPI):** For GPU computing, reducing the frequency and volume of data transferred between CPU host memory and GPU device memory is critical. For MPI applications, minimizing inter-process communication (message size and frequency) is key. Algorithms may need restructuring to improve data locality.
*   **Choosing Appropriate Data Structures:** Using efficient data structures (e.g., NumPy arrays over Python lists for numerical data, sparse matrices if applicable) can impact performance.
*   **Code Specialization:** Sometimes, rewriting the most performance-critical kernels in a lower-level language like C, C++, Fortran, or CUDA (and wrapping them for Python using tools like Cython, f2py, or ctypes) may be necessary to achieve maximum performance, although libraries like Numba and CuPy significantly reduce the need for this.

Effective optimization is often an iterative process: profile, identify bottleneck, apply optimization, profile again, and repeat until satisfactory performance is achieved or bottlenecks shift elsewhere.

**11.7 Examples in Practice (Python): HPC Applications**

The following examples illustrate the practical application of various High-Performance Computing (HPC) techniques discussed in this chapter to accelerate common astronomical computations. They showcase the usage of `Numba` for JIT compilation, `multiprocessing`/`joblib` for local CPU parallelism, `CuPy` for GPU array acceleration, `Dask` for distributed computations, `mpi4py` for cluster-based parallelism, and the integration of GPUs in Deep Learning training, each applied to a relevant problem from a different astronomical subfield.

**11.7.1 Solar: `Numba` (`@jit`) Acceleration of Custom Image Processing**
Solar physics often involves applying custom algorithms or filters to large sequences of images (e.g., from SDO). While many standard operations are optimized in NumPy/SciPy, custom functions involving explicit Python loops can become bottlenecks. Numba's Just-in-Time (JIT) compiler provides an excellent way to accelerate such functions with minimal code modification. This example demonstrates applying Numba's `@jit` decorator to accelerate a simple custom image filtering function (e.g., a local contrast enhancement filter) operating on a simulated solar image represented as a NumPy array.

The introductory paragraph should explain the context: analyzing large solar image sequences often requires custom processing. Mention that pure Python loops can be slow. State the goal: demonstrate how Numba's `@jit` decorator can significantly speed up a custom Python function involving pixel-level loops applied to a solar image, with minimal code changes.

```python
import numpy as np
import time
# Requires Numba: pip install numba
try:
    import numba
    numba_available = True
except ImportError:
    print("Numba not found, skipping Solar Numba example.")
    numba_available = False
from astropy.io import fits # For reading/writing FITS
import matplotlib.pyplot as plt
import os

# --- Simulate Solar Image Data ---
solar_image_file = 'solar_image_for_numba.fits'
if not os.path.exists(solar_image_file):
     print(f"Creating dummy file: {solar_image_file}")
     im_size = (512, 512)
     # Simulate some features on a background
     yy, xx = np.indices(im_size)
     data = 100 + 50 * np.sin(xx / 30.0) + 50 * np.cos(yy / 25.0)
     data += np.random.normal(0, 5, size=im_size)
     fits.PrimaryHDU(data.astype(np.float32)).writeto(solar_image_file, overwrite=True)

# --- Define Custom Image Processing Function (Pure Python) ---
# Example: Local standard deviation filter (simple contrast measure)
def local_stddev_filter_python(image, window_size):
    rows, cols = image.shape
    half_window = window_size // 2
    output = np.zeros_like(image, dtype=np.float64) # Use float64 for precision
    # Pad image to handle boundaries
    padded_image = np.pad(image.astype(np.float64), half_window, mode='reflect')

    for r in range(rows):
        for c in range(cols):
            # Extract local window
            window = padded_image[r : r + window_size, c : c + window_size]
            # Calculate standard deviation in the window
            output[r, c] = np.std(window)
    return output

# --- Define Numba-Accelerated Version ---
if numba_available:
    # Apply the @numba.jit decorator
    # nopython=True: Compiles fully to machine code (no Python object mode fallback) - Recommended for performance
    # parallel=True: Enables Numba's auto-parallelization (requires careful usage, may not always help)
    # fastmath=True: Allows less precise but potentially faster math operations
    @numba.jit(nopython=True, parallel=False, fastmath=True) # Disable parallel initially
    def local_stddev_filter_numba(image, window_size):
        # Numba works best with simple loops and NumPy functions/math it understands
        rows, cols = image.shape
        half_window = window_size // 2
        # Numba requires explicit type declaration for output array sometimes
        output = np.empty((rows, cols), dtype=numba.float64)
        # Padding - need to handle potentially unsupported 'reflect' mode manually or use simpler padding if needed
        # Using manual reflect padding for Numba compatibility:
        padded_image = np.empty((rows + 2*half_window, cols + 2*half_window), dtype=numba.float64)
        padded_image[half_window:rows+half_window, half_window:cols+half_window] = image.astype(numba.float64)
        # Reflect top/bottom
        for i in range(half_window):
             padded_image[i, half_window:-half_window] = image[half_window-1-i, :]
             padded_image[rows+half_window+i, half_window:-half_window] = image[rows-1-i, :]
        # Reflect left/right (including corners)
        for j in range(half_window):
             padded_image[:, j] = padded_image[:, 2*half_window-1-j]
             padded_image[:, cols+half_window+j] = padded_image[:, cols+half_window-1-j]


        # Use numba.prange for explicit parallel loops if parallel=True
        # for r in numba.prange(rows): # Example if parallel=True used
        for r in range(rows):
            for c in range(cols):
                window = padded_image[r : r + window_size, c : c + window_size]
                # Use np.std supported by Numba
                output[r, c] = np.std(window)
        return output

# --- Load Data and Run Timing Comparison ---
try:
    print(f"Loading image: {solar_image_file}")
    image_data = fits.getdata(solar_image_file).astype(np.float64) # Ensure float type
    window_size = 5 # Size of the local window

    # --- Time Pure Python Version ---
    print(f"\nTiming pure Python version (window_size={window_size})...")
    start_time = time.time()
    result_python = local_stddev_filter_python(image_data, window_size)
    end_time = time.time()
    time_python = end_time - start_time
    print(f"  Pure Python time: {time_python:.4f} seconds")

    # --- Time Numba Version ---
    if numba_available:
        print(f"\nTiming Numba version (window_size={window_size})...")
        # First call includes compilation time
        print("  (First call includes JIT compilation...)")
        start_time_compile = time.time()
        result_numba_compile = local_stddev_filter_numba(image_data, window_size)
        end_time_compile = time.time()
        time_numba_compile = end_time_compile - start_time_compile
        print(f"  Numba time (incl. compile): {time_numba_compile:.4f} seconds")
        # Second call uses cached compiled code
        print("  (Second call uses cached code...)")
        start_time_run = time.time()
        result_numba = local_stddev_filter_numba(image_data, window_size)
        end_time_run = time.time()
        time_numba_run = end_time_run - start_time_run
        print(f"  Numba time (run only): {time_numba_run:.4f} seconds")

        # Calculate Speedup
        if time_python > 0 and time_numba_run > 0:
             speedup = time_python / time_numba_run
             print(f"\n  Approximate Speedup (Python / Numba run): {speedup:.2f}x")

        # Verify results are close (allow for small floating point differences)
        # np.testing.assert_allclose(result_python, result_numba, rtol=1e-5)
        # print("  Numba result verified against Python result.")

        # --- Optional: Visualize Results ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        axes[0].imshow(image_data, cmap='gray', origin='lower')
        axes[0].set_title('Original Image')
        axes[1].imshow(result_python, cmap='viridis', origin='lower')
        axes[1].set_title('Local StdDev (Python)')
        axes[2].imshow(result_numba, cmap='viridis', origin='lower')
        axes[2].set_title('Local StdDev (Numba)')
        plt.tight_layout()
        plt.show()

    else:
        print("\nSkipping Numba timing comparison as Numba is not available.")

except FileNotFoundError:
    print(f"Error: Input file {solar_image_file} not found.")
except Exception as e:
    print(f"An unexpected error occurred in the Numba example: {e}")

```

This Python script demonstrates the significant performance improvement achievable using Numba's Just-in-Time (JIT) compiler for accelerating custom numerical Python code, specifically applied to a solar image processing task. It defines a function `local_stddev_filter_python` that calculates the standard deviation within a moving window across an image using standard Python loops and NumPy – a potentially slow operation for large images. A second version, `local_stddev_filter_numba`, implements the identical logic but is decorated with `@numba.jit(nopython=True)`. This decorator instructs Numba to compile the function to optimized machine code the first time it's called. The script loads a simulated solar image, then times the execution of both the pure Python and the Numba-compiled versions of the filter function. The timing results clearly show that after an initial compilation overhead, the Numba version runs substantially faster (often 10x-100x or more) than the pure Python equivalent, highlighting Numba's effectiveness in accelerating computationally intensive loops with minimal code modification. The visualization compares the output of both versions.

**11.7.2 Planetary: Parallelizing Asteroid Orbital Integrations (`multiprocessing`/`joblib`)**
Planetary science often involves simulating the orbits of numerous small bodies (asteroids, comets, Kuiper Belt Objects) over long timescales to study their dynamical evolution, stability, or potential interactions. Integrating the equations of motion for thousands or millions of bodies independently is an "embarrassingly parallel" problem, well-suited for parallelization across multiple CPU cores. This example simulates performing orbital integrations for a set of test particles (representing asteroids) under the gravitational influence of the Sun and major planets. It shows how to parallelize the integration loop using either Python's `multiprocessing.Pool` or the higher-level `joblib.Parallel` interface to distribute the independent integrations across available CPU cores, significantly reducing the total computation time compared to a serial execution.

```python
import numpy as np
import time
# Requires multiprocessing (standard library) or joblib: pip install joblib
from multiprocessing import Pool
try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    print("joblib not found, will use multiprocessing only.")
    joblib_available = False
# Requires rebound for N-body integration: pip install rebound
try:
    import rebound
    rebound_available = True
except ImportError:
    print("rebound not found, skipping Planetary orbital integration example.")
    rebound_available = False
import os # For setting number of threads potentially

# --- Define Orbital Integration Function ---
# This function integrates ONE test particle's orbit
def integrate_orbit(particle_state, integration_time_years):
    # particle_state: initial [x, y, z, vx, vy, vz] in AU, AU/day
    if not rebound_available: return None # Return if library missing

    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun') # Use astronomical units
    # Add Sun
    sim.add(m=1.0)
    # Add major planets (use rebound built-in setup)
    sim.add_planets() # Adds Jupiter, Saturn, Uranus, Neptune by default relative to Solar System Barycenter
    # Need to switch integrator for planets, IAS15 is good
    sim.integrator = "ias15"
    # Add the test particle
    sim.add(x=particle_state[0], y=particle_state[1], z=particle_state[2],
            vx=particle_state[3]*(365.25), vy=particle_state[4]*(365.25), vz=particle_state[5]*(365.25)) # Convert velocity to AU/yr
    # Integrate
    try:
         sim.integrate(integration_time_years)
         # Return the final state of the test particle (index -1)
         final_particle = sim.particles[-1]
         return [final_particle.x, final_particle.y, final_particle.z,
                 final_particle.vx/365.25, final_particle.vy/365.25, final_particle.vz/365.25] # Convert back to AU/day
    except Exception as e:
         print(f"Warning: Rebound integration failed: {e}")
         return None # Return None on failure


# --- Simulate Initial Asteroid States ---
if rebound_available:
    n_asteroids = 100 # Number of asteroids to integrate (increase for real test)
    print(f"Simulating initial states for {n_asteroids} asteroids...")
    # Simulate asteroids in the outer solar system (example)
    initial_states = []
    for _ in range(n_asteroids):
        a = np.random.uniform(30, 50) # Semi-major axis (AU)
        e = np.random.uniform(0, 0.2)
        inc = np.random.uniform(0, 10) # degrees
        # Use rebound to get cartesian state from orbital elements
        sim_setup = rebound.Simulation()
        sim_setup.units = ('yr', 'AU', 'Msun')
        sim_setup.add(m=1.0) # Sun
        # Velocity unit needs care, rebound uses AU/yr with these units
        vel_factor = 365.25 # Convert AU/day to AU/yr
        try:
            sim_setup.add(a=a, e=e, inc=np.radians(inc), primary=sim_setup.particles[0])
            p = sim_setup.particles[-1]
            initial_states.append([p.x, p.y, p.z, p.vx/vel_factor, p.vy/vel_factor, p.vz/vel_factor])
        except Exception as add_err:
             print(f"Warning: Failed to add particle state: {add_err}")
             # Add dummy state if failed
             initial_states.append(np.random.rand(6))


    integration_time = 1000.0 # Integrate for 1000 years

    # --- Serial Execution (for comparison) ---
    print("\nRunning serial integration...")
    start_time_serial = time.time()
    results_serial = [integrate_orbit(state, integration_time) for state in initial_states]
    end_time_serial = time.time()
    time_serial = end_time_serial - start_time_serial
    print(f"  Serial execution time: {time_serial:.4f} seconds")

    # --- Parallel Execution using multiprocessing.Pool ---
    # Determine number of cores to use
    num_cores = os.cpu_count() or 1 # Get number of available CPU cores
    print(f"\nRunning parallel integration using multiprocessing.Pool ({num_cores} cores)...")
    start_time_mp = time.time()
    # Create a pool of worker processes
    with Pool(processes=num_cores) as pool:
        # Use pool.starmap to pass arguments (state, integration_time)
        # Create list of argument tuples for starmap
        mp_args = [(state, integration_time) for state in initial_states]
        results_mp = pool.starmap(integrate_orbit, mp_args)
    end_time_mp = time.time()
    time_mp = end_time_mp - start_time_mp
    print(f"  Multiprocessing time: {time_mp:.4f} seconds")
    if time_serial > 0 and time_mp > 0:
         print(f"  Speedup (Serial / MP): {(time_serial / time_mp):.2f}x")
    # Verify results match (optional)
    # np.testing.assert_allclose(results_serial, results_mp)

    # --- Parallel Execution using joblib.Parallel ---
    if joblib_available:
        print(f"\nRunning parallel integration using joblib.Parallel ({num_cores} cores)...")
        start_time_jl = time.time()
        # Use Parallel context manager with delayed function calls
        results_jl = Parallel(n_jobs=num_cores)(delayed(integrate_orbit)(state, integration_time) for state in initial_states)
        end_time_jl = time.time()
        time_jl = end_time_jl - start_time_jl
        print(f"  Joblib time: {time_jl:.4f} seconds")
        if time_serial > 0 and time_jl > 0:
             print(f"  Speedup (Serial / Joblib): {(time_serial / time_jl):.2f}x")
        # Verify results match (optional)
        # np.testing.assert_allclose(results_serial, results_jl)

else:
    print("Skipping Planetary orbital integration example: rebound unavailable.")

```

This Python script demonstrates how to significantly speed up the computationally intensive task of integrating the orbits of multiple asteroids using CPU parallelization. It first defines a function `integrate_orbit` that performs the N-body integration for a *single* asteroid using the `rebound` library, taking the initial state vector and integration time as input. After simulating initial states for a number of asteroids, the script first runs the integrations serially using a standard Python list comprehension for baseline timing. Then, it shows two parallel approaches: first, using Python's built-in `multiprocessing.Pool`, it creates a pool of worker processes (typically one per CPU core) and uses `pool.starmap` to distribute the `integrate_orbit` calls for each asteroid across these workers. Second, it demonstrates the equivalent parallelization using the `joblib` library's `Parallel` and `delayed` constructs, which often provides a more concise syntax for parallelizing loops. Both parallel methods achieve significant speedups compared to the serial execution by utilizing multiple CPU cores simultaneously to perform the independent orbital integrations, showcasing effective parallelization for "embarrassingly parallel" scientific computations.

**11.7.3 Stellar: `CuPy` Acceleration of PSF Photometry Calculations**
PSF photometry, particularly iterative or simultaneous fitting in crowded fields (Section 6.4), involves numerous array operations and potentially linear algebra calculations (e.g., solving linear systems during fitting). These operations can often be significantly accelerated by performing them on a GPU. The `CuPy` library allows NumPy-like operations to be executed directly on NVIDIA GPUs. This example provides a conceptual illustration of how `CuPy` could be used to accelerate a hypothetical, simplified core calculation within a PSF fitting routine (e.g., evaluating the PSF model over a grid or performing a matrix operation), assuming the relevant data arrays (image cutout, PSF model) have been transferred to the GPU.

```python
import numpy as np
import time
# Requires CuPy (and compatible NVIDIA GPU + CUDA toolkit): pip install cupy
try:
    import cupy as cp
    # Check if GPU is available and CuPy is working
    try:
         cp.cuda.Device(0).use() # Select GPU 0
         print(f"CuPy found and GPU device is available: {cp.cuda.Device(0).name}")
         cupy_available = True
    except cp.cuda.runtime.CUDARuntimeError as e:
         print(f"CuPy found, but CUDA error occurred: {e}")
         print("GPU acceleration will not be possible.")
         cupy_available = False
except ImportError:
    print("CuPy not found, skipping Stellar GPU PSF example.")
    cupy_available = False

# --- Simulate Data (Small Image Cutout and PSF Model) ---
if cupy_available:
    im_size = 64 # Size of cutout
    psf_sigma = 1.5
    # Simulate a star cutout image on CPU (NumPy)
    yy_cpu, xx_cpu = np.indices((im_size, im_size))
    star_flux = 10000.0
    center_x, center_y = im_size/2.0 - 0.5, im_size/2.0 - 0.5
    image_cutout_cpu = (star_flux * np.exp(-0.5 * (((xx_cpu - center_x)/psf_sigma)**2 + ((yy_cpu - center_y)/psf_sigma)**2)) +
                        np.random.normal(10, 2, size=(im_size, im_size))).astype(np.float32)

    # Simulate a PSF model grid (e.g., evaluated analytical model) on CPU
    psf_model_cpu = (np.exp(-0.5 * (((xx_cpu - center_x)/psf_sigma)**2 + ((yy_cpu - center_y)/psf_sigma)**2))
                    ).astype(np.float32)
    psf_model_cpu /= np.sum(psf_model_cpu) # Normalize sum to 1

    # --- Define Hypothetical Calculation (e.g., weighted sum for fitting residual) ---
    # This is a simplified stand-in for a real calculation in PSF fitting
    def calculate_weighted_residual_sum_cpu(image, model, weight):
        residual = image - model * np.sum(image) # Simple scaling approximation
        weighted_residual = residual * weight
        return np.sum(weighted_residual**2) # Sum of squared weighted residuals

    # Define equivalent function using CuPy for GPU execution
    def calculate_weighted_residual_sum_gpu(image_gpu, model_gpu, weight_gpu):
        residual = image_gpu - model_gpu * cp.sum(image_gpu)
        weighted_residual = residual * weight_gpu
        return cp.sum(weighted_residual**2)

    # Define a weight map (e.g., inverse variance)
    weights_cpu = np.ones_like(image_cutout_cpu) / (2.0**2) # Uniform weights for simplicity

    # --- Time CPU Calculation ---
    print("Timing calculation on CPU (NumPy)...")
    start_time_cpu = time.time()
    # Run calculation many times to get measurable time
    n_repeats = 500
    for _ in range(n_repeats):
        result_cpu = calculate_weighted_residual_sum_cpu(image_cutout_cpu, psf_model_cpu, weights_cpu)
    cp.cuda.Stream.null.synchronize() # Ensure CPU computation is finished (though likely not needed here)
    end_time_cpu = time.time()
    time_cpu = end_time_cpu - start_time_cpu
    print(f"  CPU time ({n_repeats} repeats): {time_cpu:.4f} seconds")
    print(f"  CPU Result (last): {result_cpu}")

    # --- Time GPU Calculation using CuPy ---
    print("\nTiming calculation on GPU (CuPy)...")
    # 1. Transfer data from CPU (NumPy) to GPU (CuPy)
    start_time_transfer = time.time()
    image_cutout_gpu = cp.asarray(image_cutout_cpu)
    psf_model_gpu = cp.asarray(psf_model_cpu)
    weights_gpu = cp.asarray(weights_cpu)
    cp.cuda.Stream.null.synchronize() # Wait for transfers to complete
    end_time_transfer = time.time()
    time_transfer = end_time_transfer - start_time_transfer
    print(f"  Data transfer time (CPU->GPU): {time_transfer:.4f} seconds")

    # 2. Perform calculation on GPU
    start_time_gpu = time.time()
    for _ in range(n_repeats):
        result_gpu_device = calculate_weighted_residual_sum_gpu(image_cutout_gpu, psf_model_gpu, weights_gpu)
    # Synchronize the GPU stream to ensure calculation is finished before stopping timer
    cp.cuda.Stream.null.synchronize()
    end_time_gpu = time.time()
    time_gpu = end_time_gpu - start_time_gpu
    print(f"  GPU time ({n_repeats} repeats): {time_gpu:.4f} seconds")

    # 3. Transfer result back from GPU to CPU (if needed)
    result_gpu_host = cp.asnumpy(result_gpu_device) # or result_gpu_device.get()
    print(f"  GPU Result (last, on host): {result_gpu_host}")

    # Calculate Speedup (Computation Only)
    if time_cpu > 0 and time_gpu > 0:
         speedup_compute = time_cpu / time_gpu
         print(f"\n  Approx Compute Speedup (CPU / GPU compute): {speedup_compute:.2f}x")
         speedup_total = time_cpu / (time_gpu + time_transfer) # Include transfer time
         print(f"  Approx Overall Speedup (CPU / GPU total): {speedup_total:.2f}x (includes CPU->GPU transfer)")

    # Verify results are close
    # np.testing.assert_allclose(result_cpu, result_gpu_host, rtol=1e-5)
    # print("  GPU result verified against CPU result.")

else:
    print("Skipping Stellar GPU PSF example: CuPy unavailable or no GPU found.")
```

This Python script provides a conceptual demonstration of accelerating a core computational step within PSF photometry using the `CuPy` library for GPU execution. It simulates a small image cutout containing a star and a corresponding PSF model as NumPy arrays on the CPU. A simple function `calculate_weighted_residual_sum_cpu` representing a hypothetical calculation (like summing weighted squared residuals, common in fitting) is defined using standard NumPy operations. An equivalent function `calculate_weighted_residual_sum_gpu` is defined using `cupy` functions, designed to operate on GPU arrays. The script times the execution of the NumPy version on the CPU over many repeats. It then explicitly transfers the input NumPy arrays to the GPU using `cp.asarray()`, times this transfer, executes the `cupy` version of the function on the GPU (again repeated for timing accuracy, ensuring GPU synchronization), and optionally transfers the result back. By comparing the CPU execution time to the GPU execution time (potentially including data transfer time), the script illustrates the significant speedup potential offered by GPUs for array-based numerical computations typical in image analysis and fitting routines, leveraging CuPy's NumPy-compatible interface.

**11.7.4 Exoplanetary: `Dask` Distribution of Lomb-Scargle**
Searching for periodic signals (like transits or stellar pulsations) in massive time-domain datasets, such as light curves for millions of stars from Kepler, TESS, or LSST, can be computationally demanding. Calculating the Lomb-Scargle periodogram (Section 8.2.2) for each star over a fine grid of frequencies requires significant computation. Dask provides a way to distribute this task across multiple cores or even nodes in a cluster. This example conceptually outlines how Dask could parallelize the computation of Lomb-Scargle periodograms for a large number of simulated light curves. It simulates generating many light curves, wraps the Lomb-Scargle calculation for a single light curve in a function, and uses `dask.delayed` or `dask.bag` to compute the periodograms in parallel using Dask's scheduler.

```python
import numpy as np
import time
# Requires Dask and Dask.distributed: pip install dask distributed
try:
    import dask
    import dask.array as da
    from dask.delayed import delayed
    from dask.distributed import Client, LocalCluster
    dask_available = True
except ImportError:
    print("Dask not found, skipping Exoplanet Dask Lomb-Scargle example.")
    dask_available = False
# Requires astropy.timeseries: pip install astropy
try:
    from astropy.timeseries import LombScargle
    astropy_timeseries_available = True
except ImportError:
    print("astropy.timeseries not found, cannot run LombScargle.")
    astropy_timeseries_available = False
import astropy.units as u

# --- Simulate Multiple Light Curves ---
# (Simplified simulation)
def simulate_light_curve(lc_id, n_points=500, period=None, transit_depth=0.01):
    """Simulates a simple light curve with optional transit."""
    times = np.sort(np.random.rand(n_points)) * 10.0 # 10 days baseline
    flux = np.random.normal(1.0, 0.005, size=n_points) # Baseline flux = 1 + noise
    if period is not None:
        # Add transit signal
        transit_duration = 0.1 # days
        phase = (times / period) % 1.0
        in_transit = (phase > (0.5 - transit_duration / (2 * period))) & \
                     (phase < (0.5 + transit_duration / (2 * period)))
        flux[in_transit] -= transit_depth
    return {'id': lc_id, 'time': times, 'flux': flux, 'flux_err': 0.005}

# Generate a list of simulated light curve data (as dictionaries)
n_light_curves = 200 # Number of light curves to process (increase for real test)
print(f"Simulating {n_light_curves} light curves...")
# Add some with transits, some without
periods_sim = np.random.uniform(1.0, 5.0, n_light_curves)
has_transit = np.random.rand(n_light_curves) < 0.1 # 10% have transits
light_curve_data = []
for i in range(n_light_curves):
    p = periods_sim[i] if has_transit[i] else None
    lc_data = simulate_light_curve(lc_id=i, period=p)
    light_curve_data.append(lc_data)

# --- Define Lomb-Scargle Calculation Function ---
# Function to process ONE light curve
def calculate_best_ls_period(lc_dict):
    if not astropy_timeseries_available: return None
    try:
        ls = LombScargle(lc_dict['time'], lc_dict['flux'], dy=lc_dict['flux_err'])
        # Define frequency grid or use autopower
        frequency, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=1.0, # Freq in 1/day
                                        samples_per_peak=10)
        best_freq = frequency[np.argmax(power)]
        best_period = (1 / best_freq) if best_freq > 0 else np.nan
        return {'id': lc_dict['id'], 'best_period_LS': best_period, 'max_power_LS': power.max()}
    except Exception as e:
         # print(f"LS Error for ID {lc_dict.get('id', -1)}: {e}")
         return {'id': lc_dict.get('id', -1), 'best_period_LS': np.nan, 'max_power_LS': np.nan}

# --- Execute with Dask ---
if dask_available and astropy_timeseries_available:
    print("\nSetting up Dask local cluster...")
    # Set up a local cluster using available cores
    # n_workers=number of processes, threads_per_worker=threads within each process
    cluster = LocalCluster(n_workers=os.cpu_count() or 2, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")

    # --- Method 1: Using dask.delayed ---
    print("\nProcessing light curves in parallel using dask.delayed...")
    start_time_dask = time.time()
    # Create lazy computation graph
    lazy_results = [delayed(calculate_best_ls_period)(lc) for lc in light_curve_data]
    # Trigger computation and collect results
    results_dask = dask.compute(*lazy_results)
    end_time_dask = time.time()
    time_dask = end_time_dask - start_time_dask
    print(f"  Dask computation time: {time_dask:.4f} seconds")

    # Process results (e.g., create table)
    results_table = Table(rows=[r for r in results_dask if r is not None])
    print("\nResults (first 10):")
    print(results_table[:10])

    # --- Method 2: Using dask.bag (Alternative for list processing) ---
    # import dask.bag as db
    # print("\nProcessing light curves in parallel using dask.bag...")
    # start_time_bag = time.time()
    # Create a Dask Bag from the list of dictionaries
    # b = db.from_sequence(light_curve_data, npartitions=os.cpu_count())
    # Apply the function to each element in the bag
    # results_bag = b.map(calculate_best_ls_period).compute()
    # end_time_bag = time.time()
    # time_bag = end_time_bag - start_time_bag
    # print(f"  Dask Bag computation time: {time_bag:.4f} seconds")


    # Shutdown Dask client and cluster
    print("\nShutting down Dask client and cluster...")
    client.close()
    cluster.close()

else:
    print("Skipping Exoplanet Dask Lomb-Scargle example: Dask or Astropy unavailable.")

```

This Python script demonstrates how the Dask library can be used to parallelize the computationally intensive task of calculating Lomb-Scargle periodograms for a large number of exoplanetary light curves. It first simulates a list of light curve datasets (represented as dictionaries containing time, flux, and error arrays). A function `calculate_best_ls_period` is defined to perform the Lomb-Scargle analysis (using `astropy.timeseries.LombScargle`) for a single light curve and return the period corresponding to the highest peak. The key parallelization step uses `dask.delayed`: the processing function is wrapped with `delayed()`, and this delayed function is applied to each light curve in the input list, creating a list of `lazy_results`. This builds a Dask task graph representing the computation without immediately executing it. Calling `dask.compute(*lazy_results)` triggers the execution of this graph, distributing the independent Lomb-Scargle calculations across multiple worker processes managed by a Dask scheduler (here, a `LocalCluster` utilizing multiple CPU cores). This parallel execution significantly reduces the total time required to analyze all light curves compared to a serial loop. The results (best period and peak power for each light curve) are collected and can be compiled into a table for further analysis.

**11.7.5 Galactic: `mpi4py` Parallelization of Parameter Space Search**
In Galactic dynamics or stellar population synthesis, researchers often compare observational data to large grids of theoretical models, searching for the best-fit parameters. If the model grid is very large and evaluating the goodness-of-fit (e.g., chi-squared, likelihood) for each model point is computationally non-trivial, this parameter space search becomes a bottleneck. For large High-Performance Computing (HPC) clusters with distributed memory, the Message Passing Interface (MPI) provides a standard way to parallelize such tasks. This conceptual example outlines using `mpi4py` to distribute the calculation of chi-squared values over a grid of model parameters across multiple MPI processes (ranks). Each rank calculates chi-squared for a subset of the parameter grid, and the results are gathered back to a root rank to find the overall minimum.

```python
# Conceptual Example: Parallel Parameter Grid Search using mpi4py
# This script is intended to be run with mpiexec:
# mpiexec -n <N_PROCESSES> python this_script.py

import numpy as np
import time
# Requires mpi4py and an MPI implementation (e.g., OpenMPI, MPICH)
# pip install mpi4py
try:
    from mpi4py import MPI
    mpi4py_available = True
except ImportError:
    print("mpi4py not found, skipping Galactic MPI example.")
    mpi4py_available = False

# --- Setup MPI ---
if mpi4py_available:
    comm = MPI.COMM_WORLD # Default communicator
    rank = comm.Get_rank() # Rank (ID) of the current process
    size = comm.Get_size() # Total number of processes
else: # Define dummy values if mpi4py not available for script structure
    comm = None
    rank = 0
    size = 1

# --- Define Problem (Dummy Model and Data) ---
# Simulate observational data (e.g., CMD points for a cluster)
n_data_points = 100
observed_color = np.random.normal(1.0, 0.3, n_data_points)
observed_mag = 5.0 * observed_color + 15.0 + np.random.normal(0, 0.5, n_data_points) # Simple linear relation + noise
observed_data = np.vstack((observed_color, observed_mag)).T

# Define a computationally intensive function to evaluate model goodness-of-fit (chi-squared)
# This function simulates calculating chi-squared for ONE model parameter set
def calculate_chi_squared(param1, param2, obs_data):
    # param1, param2: Model parameters (e.g., age, metallicity)
    # obs_data: Observational data points
    # Simulate model prediction (e.g., isochrone points) - replace with real model eval
    model_color = np.linspace(0, 2, 50)
    # Model depends on parameters (e.g., slope changes with param1, intercept with param2)
    model_mag = (4.0 + param1) * model_color + (14.0 + param2)
    # Simulate a time-consuming part
    time.sleep(0.001) # Simulate 1 ms calculation time per model point

    # Calculate chi-squared (simplified nearest neighbor distance sum)
    chi2 = 0
    for data_point in obs_data:
        distances_sq = (model_color - data_point[0])**2 + (model_mag - data_point[1])**2
        chi2 += np.min(distances_sq) # Simple distance metric
    return chi2

# --- Define Parameter Grid ---
# Define the grid of parameters to search
param1_values = np.linspace(0.5, 1.5, 50) # Example: 50 steps for param1
param2_values = np.linspace(0.0, 2.0, 50) # Example: 50 steps for param2
total_grid_points = len(param1_values) * len(param2_values)

if rank == 0: # Master process prints info
    print(f"Starting MPI Parameter Grid Search on {size} processes.")
    print(f"Parameter Grid Size: {len(param1_values)} x {len(param2_values)} = {total_grid_points} points")

# --- Distribute Work among MPI Ranks ---
# Divide the total grid points among the available processes (ranks)
points_per_rank = total_grid_points // size
start_index = rank * points_per_rank
# The last rank might get slightly more points if not evenly divisible
end_index = (rank + 1) * points_per_rank if rank != size - 1 else total_grid_points

# Each rank processes its assigned chunk of the grid
local_results = [] # List to store results for this rank: [(param1, param2, chi2), ...]
start_time_local = time.time()
grid_idx = 0
for i, p1 in enumerate(param1_values):
    for j, p2 in enumerate(param2_values):
        if start_index <= grid_idx < end_index:
            # This grid point belongs to the current rank
            chi2_value = calculate_chi_squared(p1, p2, observed_data)
            local_results.append({'p1': p1, 'p2': p2, 'chi2': chi2_value})
        grid_idx += 1

end_time_local = time.time()
time_local = end_time_local - start_time_local
print(f"Rank {rank} processed {len(local_results)} grid points in {time_local:.4f} seconds.")

# --- Gather Results on Root Rank (Rank 0) ---
# Use comm.gather to collect the list of dictionaries from each rank
# Only rank 0 will receive the full list; other ranks receive None
if comm is not None: # Check if MPI is actually available
     all_results_list = comm.gather(local_results, root=0)
else: # Simulate gathering for single process run
     all_results_list = [local_results] if rank == 0 else None


# --- Process Results on Root Rank ---
if rank == 0:
    print("\nGathering results on Rank 0...")
    # Combine the lists from all ranks into one large list
    final_results = []
    if all_results_list: # Check if gathering returned data
        for rank_results in all_results_list:
             if rank_results: # Check if the list from a rank is not None/empty
                  final_results.extend(rank_results)

    if not final_results:
        print("Error: No results were gathered.")
    else:
        # Find the best-fit parameters (minimum chi-squared)
        best_fit = min(final_results, key=lambda x: x['chi2'])
        print("\nBest Fit Parameters (Minimum Chi-squared):")
        print(f"  Param1 = {best_fit['p1']:.3f}")
        print(f"  Param2 = {best_fit['p2']:.3f}")
        print(f"  Min Chi^2 = {best_fit['chi2']:.4f}")

        # Convert results to a table for potential saving (optional)
        # results_table = Table(rows=final_results)
        # print("\nFull Results Table (first 5 rows):")
        # print(results_table[:5])

# --- Final MPI Barrier (Optional but good practice) ---
if comm is not None:
    comm.Barrier() # Ensure all processes finish before exiting

```

This Python script provides a conceptual framework for parallelizing a parameter space search across multiple nodes of an HPC cluster using the Message Passing Interface (MPI) via the `mpi4py` library. **Note:** This script is intended to be executed using `mpiexec -n N python your_script.py`, where N is the number of processes. The script starts by initializing MPI, obtaining the rank (unique ID) of the current process and the total number of processes (`size`). It defines a computationally intensive placeholder function `calculate_chi_squared` that evaluates the goodness-of-fit for a given set of model parameters against simulated observational data. A grid of model parameters is defined. The core parallelization logic involves dividing the total number of grid points among the available MPI ranks. Each rank then iterates through *only its assigned subset* of the parameter grid, calculating the chi-squared value for each point and storing the results locally. After all ranks complete their assigned calculations, the `comm.gather` collective operation is used to collect the lists of local results from all ranks onto the root rank (rank 0). The root rank then combines these results and identifies the parameter set corresponding to the overall minimum chi-squared value, representing the best fit found across the entire grid. This approach effectively distributes the computational load of the parameter search across the cluster nodes, significantly reducing the time required for exploring large model grids.

**11.7.6 Extragalactic: `Numba` (`@vectorize`) for Faster Catalog Calculations**
Analyzing large astronomical catalogs often involves calculating derived quantities for millions or billions of sources based on their measured properties. While Pandas and NumPy are efficient for many vectorized operations, applying custom mathematical functions row-by-row using Python loops can be slow. Numba's `@vectorize` decorator provides a powerful way to create fast, NumPy-style Universal Functions (ufuncs) directly from simple Python functions that operate on scalars. These compiled ufuncs can execute efficiently on arrays, leveraging CPU SIMD instructions or even GPU parallelism with appropriate signatures. This example demonstrates using `@numba.vectorize` to accelerate the calculation of a derived quantity (e.g., absolute magnitude) for a large catalog of galaxies, operating directly on NumPy arrays extracted from the catalog table.

```python
import numpy as np
import pandas as pd # Often used for catalogs, but operate on NumPy arrays for Numba
import time
# Requires Numba: pip install numba
try:
    import numba
    numba_available = True
except ImportError:
    print("Numba not found, skipping Extragalactic Numba vectorize example.")
    numba_available = False
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM # Example cosmology

# --- Simulate Large Galaxy Catalog Data ---
n_galaxies = 1_000_000 # Large number of galaxies
print(f"Simulating catalog data for {n_galaxies:,} galaxies...")
# Simulate apparent magnitude and redshift
app_mag = np.random.uniform(18.0, 24.0, n_galaxies)
redshift = np.random.uniform(0.1, 2.0, n_galaxies)
# Create NumPy arrays (Numba vectorize works well on these)
catalog_data = {'app_mag': app_mag, 'redshift': redshift}

# --- Define Calculation Function (Scalar Python) ---
# Function to calculate absolute magnitude from apparent mag and redshift
# Requires a cosmology object
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # Example cosmology
def calculate_abs_mag_scalar(app_mag, z, cosmo_model):
    if z <= 0: # Avoid issues with log(0) or negative distance
        return np.nan
    try:
        dist_mod = cosmo_model.distmod(z).value # Distance modulus
        abs_mag = app_mag - dist_mod
        return abs_mag
    except Exception: # Handle potential cosmology calculation errors
        return np.nan

# --- Numba Vectorized Version ---
if numba_available:
    # Use @numba.vectorize decorator
    # Specify input/output types for compilation ('float64(float64, float64)')
    # Or use type signatures for more flexibility, e.g., allow float32
    # target='parallel' can parallelize over CPU cores
    # target='cuda' can compile for GPU (requires CUDA toolkit)
    @numba.vectorize(['float64(float64, float64)'], target='parallel') # Example signature for CPU parallel
    def calculate_abs_mag_vectorized(app_mag, z):
        # NOTE: Cannot pass complex objects like astropy cosmology directly
        # into nopython mode or vectorized functions easily.
        # Need to pass essential cosmological parameters or use a simplified
        # calculation within the Numba function if possible.
        # ---- Simplified distance calculation for Numba example ----
        # This is NOT cosmologically accurate, just for demonstrating vectorization
        # Replace with lookup table interpolation or precompute distance modulus if needed
        dist_mod_approx = 5 * np.log10(z / 0.1) + 40.0 # Very rough approximation!
        if z <= 0: return np.nan
        # ---------------------------------------------------------
        abs_mag = app_mag - dist_mod_approx
        return abs_mag

# --- Time Calculations ---
# Extract NumPy arrays from dictionary/DataFrame
app_mag_array = catalog_data['app_mag']
redshift_array = catalog_data['redshift']

# --- Time Standard Python Loop (Inefficient) ---
print("\nTiming calculation using Python loop (Inefficient - DO NOT DO THIS)...")
start_time_loop = time.time()
abs_mag_loop = np.zeros_like(app_mag_array)
for i in range(n_galaxies):
     # Call original scalar function, passing the cosmology object
     # This is slow due to loop overhead and repeated function calls
     abs_mag_loop[i] = calculate_abs_mag_scalar(app_mag_array[i], redshift_array[i], cosmo)
end_time_loop = time.time()
time_loop = end_time_loop - start_time_loop
print(f"  Python loop time: {time_loop:.4f} seconds")

# --- Time Standard NumPy/Astropy (Vectorized if possible) ---
# Try using astropy cosmology directly on arrays (may be optimized internally)
print("\nTiming calculation using Astropy Cosmology (vectorized)...")
start_time_astropy = time.time()
# Ensure redshift array is valid for distmod
valid_z = redshift_array > 0
distmod_astropy = np.full_like(redshift_array, np.nan)
distmod_astropy[valid_z] = cosmo.distmod(redshift_array[valid_z]).value
abs_mag_astropy = app_mag_array - distmod_astropy
end_time_astropy = time.time()
time_astropy = end_time_astropy - start_time_astropy
print(f"  Astropy vectorized time: {time_astropy:.4f} seconds")

# --- Time Numba Vectorized Version ---
if numba_available:
    print("\nTiming calculation using Numba @vectorize...")
    # Numba vectorized function is called like a NumPy ufunc
    start_time_numba = time.time()
    # Pass only NumPy arrays to the Numba function
    abs_mag_numba = calculate_abs_mag_vectorized(app_mag_array, redshift_array)
    end_time_numba = time.time()
    time_numba = end_time_numba - start_time_numba
    print(f"  Numba vectorized time: {time_numba:.4f} seconds")

    # Calculate Speedup (vs Astropy/NumPy baseline)
    if time_astropy > 0 and time_numba > 0:
         speedup = time_astropy / time_numba
         print(f"\n  Approximate Speedup (Astropy / Numba vectorized): {speedup:.2f}x")
         # Note: Speedup depends heavily on complexity of the function and whether
         # the NumPy/Astropy version was already highly optimized.
         # The simplified Numba function here might be faster than accurate cosmology calc.

    # Compare results (Numba vs Astropy - expect differences due to approx calc)
    print("\nComparing results (first 10):")
    print("Abs Mag (Astropy):", abs_mag_astropy[:10])
    print("Abs Mag (Numba Approx):", abs_mag_numba[:10])

else:
    print("\nSkipping Numba vectorized calculation as Numba is not available.")

```

This Python script demonstrates how Numba's `@vectorize` decorator can accelerate element-wise calculations on large astronomical catalogs. It simulates a large catalog containing apparent magnitudes and redshifts for many galaxies. A standard Python function `calculate_abs_mag_scalar` is defined to compute absolute magnitude for a single galaxy using `astropy.cosmology`. The script first times the (inefficient) pure Python loop applying this scalar function to every row. Then, it times the standard vectorized approach using `astropy.cosmology` directly on the NumPy arrays, which is typically well-optimized. Finally, it defines a Numba-accelerated version `calculate_abs_mag_vectorized` using the `@numba.vectorize` decorator with appropriate type signatures and the `target='parallel'` option to enable multi-core CPU execution. **Crucially**, this Numba version uses a *simplified*, approximate distance calculation because complex objects like the `astropy.cosmology` object cannot be easily passed into Numba's `nopython` mode. The script times the execution of this Numba ufunc applied directly to the input NumPy arrays. The comparison highlights that while standard Astropy/NumPy operations are often already fast, `@vectorize` offers a way to significantly speed up custom scalar calculations applied element-wise across large arrays, achieving performance gains through JIT compilation and potential parallelization, provided the function's logic can be expressed within Numba's constraints.

**11.7.7 Cosmology: GPU Training of CNN for Parameter Estimation**
Estimating cosmological parameters (like $\Omega_m$, $\sigma_8$) from observational data like weak lensing maps or large-scale structure density fields is a key goal in cosmology. Deep Learning, particularly CNNs, can learn to extract relevant features directly from these map-like datasets and perform regression to estimate parameters, potentially capturing non-Gaussian information missed by traditional summary statistics (e.g., Villaescusa-Navarro et al., 2021; Jeffrey et al., 2021). Training these deep CNNs on large sets of simulated maps is computationally intensive and significantly benefits from GPU acceleration. This example conceptually outlines defining a CNN model using `tensorflow.keras` (similar to Example 10.6.7 but for regression) and highlights the code modifications needed to ensure training occurs on an available GPU, leveraging the framework's built-in GPU support.

```python
import numpy as np
# Requires tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tensorflow_available = True
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow found {len(gpus)} GPU(s): {gpus}")
            gpu_available = True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"TensorFlow GPU setup error: {e}")
            gpu_available = False
    else:
        print("TensorFlow did not find any GPUs. Training will use CPU.")
        gpu_available = False

except ImportError:
    print("TensorFlow/Keras not found, skipping Cosmology CNN regression example.")
    tensorflow_available = False
    gpu_available = False
import matplotlib.pyplot as plt
import time

# --- Conceptual Example: CNN for Cosmological Parameter Regression ---
# Focuses on model definition and noting GPU usage.
# Assumes input data X: image maps (e.g., weak lensing convergence maps)
# Assumes output data y: corresponding cosmological parameters (e.g., Omega_m, sigma_8)

if tensorflow_available:
    print("\nDefining conceptual CNN model for Cosmological Parameter Regression...")

    # Example input shape: (map_size, map_size, n_channels)
    input_shape_example = (128, 128, 1) # Example: 128x128 map, 1 channel (e.g., density)

    # Example output shape: number of parameters to predict
    n_output_params = 2 # Example: Predict Omega_m and sigma_8

    # --- Define CNN Model Architecture (Example Regression Model) ---
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape_example),
            # Convolutional Blocks
            layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            # Output Layer: Linear activation for regression, units = n_output_params
            layers.Dense(units=n_output_params, activation="linear"),
        ]
    )
    model.summary()

    # --- Compile the Model for Regression ---
    # Use appropriate loss function (e.g., Mean Squared Error) and metrics
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"]) # Mean Absolute Error
    print("\nModel compiled for regression (MSE loss).")

    # --- Simulate Training Data (Highly Simplified) ---
    # In reality, load large suites of simulation maps and parameters
    print("Simulating dummy training/validation data...")
    n_train = 500
    n_val = 100
    X_train_sim = np.random.rand(n_train, *input_shape_example).astype(np.float32)
    # Simulate parameters related to input mean (very unrealistic, just for demo)
    y_train_sim = np.random.rand(n_train, n_output_params).astype(np.float32) * \
                  np.mean(X_train_sim, axis=(1,2,3), keepdims=True)*0.1 + \
                  np.array([[0.3, 0.8]]) # Omega_m ~ 0.3, sigma_8 ~ 0.8 baseline

    X_val_sim = np.random.rand(n_val, *input_shape_example).astype(np.float32)
    y_val_sim = np.random.rand(n_val, n_output_params).astype(np.float32) * \
                np.mean(X_val_sim, axis=(1,2,3), keepdims=True)*0.1 + \
                np.array([[0.3, 0.8]])

    # --- Train the Model (Utilizing GPU if Available) ---
    print("\nStarting conceptual model training...")
    epochs = 5 # Small number for demo
    batch_size = 16

    # TensorFlow/Keras automatically uses available GPUs if detected and configured correctly.
    # No explicit device placement needed in standard training loop usually.
    start_time_train = time.time()
    history = model.fit(X_train_sim, y_train_sim,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val_sim, y_val_sim),
                        verbose=1) # Print progress
    end_time_train = time.time()
    time_train = end_time_train - start_time_train

    device_used = "GPU" if gpu_available else "CPU"
    print(f"\nTraining complete on {device_used} in {time_train:.2f} seconds.")

    # --- Evaluate (Conceptual) ---
    # loss, mae = model.evaluate(X_test_sim, y_test_sim)
    # print(f"Test Loss (MSE): {loss:.4f}, Test MAE: {mae:.4f}")
    print("(Model evaluation on a separate test set would follow)")

    # --- Plot Training History (Loss) ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"CNN Training History ({device_used})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Often useful for loss plots
    plt.show()

else:
    print("Skipping Cosmology CNN regression example: TensorFlow/Keras unavailable.")

```

This final Python script provides a conceptual demonstration of using a Convolutional Neural Network (CNN) for cosmological parameter estimation from map-like data (e.g., weak lensing maps), highlighting the utilization of GPU acceleration via the `tensorflow.keras` framework. It defines a CNN architecture suitable for regression, with convolutional layers to extract spatial features from input maps and dense layers culminating in an output layer with linear activation predicting multiple continuous cosmological parameters (e.g., $\Omega_m$, $\sigma_8$). The script includes checks for GPU availability using `tf.config.list_physical_devices('GPU')` and enables memory growth for stability. The model is compiled with an appropriate regression loss function (Mean Squared Error). Crucially, the training step, executed using `model.fit()`, **automatically utilizes any detected and configured GPU** without requiring explicit code changes for device placement in standard scenarios. The script simulates training for a few epochs on dummy data and reports the device used (GPU or CPU) and the training time, illustrating how deep learning frameworks seamlessly leverage GPU hardware to drastically reduce the substantial training times required for complex models applied to large cosmological simulation datasets. The training history plot shows the decrease in loss over epochs.

---

**References**

Alsing, J., Charnock, T., Feeney, S., & Wandelt, B. (2019). Fast likelihood-free cosmology with density estimation and active learning. *Monthly Notices of the Royal Astronomical Society, 488*(3), 4440–4456. https://doi.org/10.1093/mnras/stz1961 *(Note: Pre-2020, but relevant LFI example)*
*   *Summary:* Explores likelihood-free inference using neural networks for cosmology. Represents advanced ML/DL applications that heavily rely on accelerated computing (GPUs/TPUs) for training (Sections 11.4, 11.5).

Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tock, S., Lodieu, N., … Astropy Project Contributors. (2022). The Astropy Project: Sustaining and growing a community-oriented Python package for astronomy. *The Astrophysical Journal, 935*(2), 167. https://doi.org/10.3847/1538-4357/ac7c74
*   *Summary:* Describes the Astropy project. While not directly an HPC library, its core data structures and affiliated packages are often the starting point for analyses that require acceleration via tools like Numba, Dask, CuPy, etc. (Sections 11.2-11.4).

Buchner, J. (2021). Nested sampling methods. *Statistics and Computing, 31*(5), 70. https://doi.org/10.1007/s11222-021-10042-z
*   *Summary:* Reviews nested sampling, a Bayesian inference algorithm often used in astrophysics. Complex model evaluations within these algorithms represent a significant computational bottleneck (Section 11.1) that can benefit from parallelization (Section 11.2).

CuPy Developers. (n.d.). *CuPy: A NumPy/SciPy-compatible array library accelerated by CUDA*. Retrieved from https://cupy.dev/ *(Note: Software website)*
*   *Summary:* Official website for CuPy. CuPy provides the NumPy-like interface for GPU array computing discussed in Section 11.4.2 and demonstrated in Example 11.7.3, enabling GPU acceleration with minimal code changes for array operations.

Dalcin, L. D., Paz, R. R., Kler, P. A., & Cosimo, A. (2011). Parallel distributed computing using Python. *Advances in Water Resources, 34*(9), 1124–1139. https://doi.org/10.1016/j.advwatres.2011.04.013 *(Note: Pre-2020, but key mpi4py reference)*
*   *Summary:* Although pre-2020, this paper provides details on `mpi4py`, the essential Python library providing bindings for the Message Passing Interface (MPI), enabling large-scale distributed memory parallelism on HPC clusters (Section 11.2.4, Example 11.7.5).

Dask Development Team. (2016). *Dask: Library for dynamic task scheduling*. https://dask.org *(Note: Software website)*
*   *Summary:* Official website for Dask. Dask provides the high-level framework for parallel and distributed computing, scaling NumPy, Pandas, and Scikit-learn workflows (Section 11.3.1), as demonstrated conceptually in Example 11.7.4.

Di Matteo, T., Perna, R., Davé, R., & Feng, Y. (2023). Computational astrophysics: The numerical exploration of the hidden Universe. *Nature Reviews Physics, 5*(10), 615–634. https://doi.org/10.1038/s42254-023-00624-2
*   *Summary:* This review highlights the importance and challenges of large-scale astrophysical simulations (N-body, hydrodynamics). These simulations represent major computational bottlenecks (Section 11.1) that necessitate the use of HPC techniques like MPI and GPU acceleration (Sections 11.2.4, 11.4).

Jeffrey, N., Alsing, J., & Lanusse, F. (2021). Measuring the density parameter $\Omega_\mathrm{m}$ from weak lensing maps using simulation-based inference. *Monthly Notices of the Royal Astronomical Society, 501*(1), 701–714. https://doi.org/10.1093/mnras/staa3674
*   *Summary:* Applies simulation-based inference with neural networks (likely requiring GPU training, Section 11.4.4/11.7.7) to estimate cosmological parameters from weak lensing maps, demonstrating advanced ML/HPC synergy in cosmology.

Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*. https://doi.org/10.1145/2833157.2833162 *(Note: Foundational Numba paper, pre-2020)*
*   *Summary:* The foundational paper describing Numba. Although pre-2020, Numba remains a key tool for JIT compilation and GPU kernel programming in Python (Section 11.4.3), used in Examples 11.7.1 and 11.7.6 to accelerate Python code.

Modi, C., Feng, Y., & Póczos, B. (2021). Cosmological N-body simulations on TPU. *Machine Learning and the Physical Sciences Workshop, 35th Conference on Neural Information Processing Systems (NeurIPS 2021)*. https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_115.pdf
*   *Summary:* Describes the implementation and performance of cosmological N-body simulations specifically on Google's Tensor Processing Units (TPUs). Provides a direct example of using TPUs (Section 11.5) for computationally demanding astrophysical simulations.

Numba Developers. (n.d.). *Numba: High performance Python compiler*. Retrieved from https://numba.pydata.org/ *(Note: Software website)*
*   *Summary:* Official website for Numba. Numba provides the JIT compiler capabilities (using decorators like `@jit`, `@vectorize`, `@cuda.jit`) discussed in Section 11.4.3 and used in Examples 11.7.1 and 11.7.6.

Nvidia Corporation. (n.d.). *CUDA Toolkit*. Retrieved from https://developer.nvidia.com/cuda-toolkit *(Note: Software website)*
*   *Summary:* The official portal for NVIDIA's CUDA Toolkit, the fundamental software development kit for programming NVIDIA GPUs. Provides the underlying libraries, compiler (NVCC), and APIs used by Python GPU libraries like CuPy and Numba (Section 11.4.1).

Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C. (2017). CuPy: A NumPy-compatible matrix library accelerated by CUDA. *Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*. *(Note: Foundational CuPy paper, pre-2020)*
*   *Summary:* The foundational paper introducing CuPy. While pre-2020, CuPy is the key library providing the NumPy-like API for GPU array computing in Python discussed in Section 11.4.2 and used in Example 11.7.3.

Ray Team. (n.d.). *Ray: A Framework for Scaling AI and Python Apps*. Retrieved from https://www.ray.io/ *(Note: Software website)*
*   *Summary:* Official website for the Ray project. Ray is discussed in Section 11.3.2 as a distributed computing framework alternative to Dask, particularly strong for scaling general Python applications and complex ML workflows.

Reedy, F., & Owen, J. E. (2024). Parallelized radiation hydrodynamics for protoplanetary disc gaps. *Monthly Notices of the Royal Astronomical Society, 527*(4), 11907–11924. https://doi.org/10.1093/mnras/stad3868
*   *Summary:* Presents a parallelized radiation hydrodynamics code for simulating protoplanetary disks. This exemplifies the type of computationally intensive simulation (Section 11.1) that relies heavily on parallelization techniques (likely MPI/OpenMP, Section 11.2) for feasibility.

Turk, M. J., Fielding, D., Groth, C., Hanke, F., Hill, A., Kandola, K., Kumar, R., Lackey, J. R., Lanz, L., Mummert, J., Osborne, S. E., Rockefeller, G., Rosdahl, J., & Thompson, C. (2019). Women in Computational Astrophysics: Introduction. *Computational Astrophysics and Cosmology, 6*(1), 5. https://doi.org/10.1186/s40668-019-0028-5
*   *Summary:* While an introductory article to a special issue, it highlights the broad scope and importance of computational methods (including HPC and simulations) across astrophysics, providing context for computational bottlenecks (Section 11.1).

Varoquaux, G., Grisel, O., Passos, A., & Merriam, P. (n.d.). *Joblib: running Python functions as pipeline jobs*. Retrieved from https://joblib.readthedocs.io/ *(Note: Software documentation)*
*   *Summary:* Official documentation for Joblib. Joblib provides the high-level `Parallel` and `delayed` interface for easy CPU parallelization of Python loops (Section 11.2.3), as demonstrated in Example 11.7.2.

Villaescusa-Navarro, F., Angles-Alcazar, D., Genel, S., Nagai, D., Nelson, D., Pillepich, A., Hernquist, L., Marinacci, F., Pakmor, R., Springel, V., Vogelsberger, M., ZuHone, J., & Weinberger, R. (2023). Splashdown: Representing cosmological simulations through neural networks. *The Astrophysical Journal Supplement Series, 266*(2), 38. https://doi.org/10.3847/1538-4365/accc3e
*   *Summary:* Explores using deep learning (neural representations) to compress and analyze large cosmological simulation outputs. This application often requires significant GPU resources (Section 11.4.4) for training the neural networks.
