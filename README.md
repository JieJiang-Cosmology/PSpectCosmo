```markdown
# PSpectCosmo

`PSpectCosmo` is a C++ program developed to investigate early-universe cosmological dynamics, with a specific emphasis on the inflationary epoch. The program depends on the `fftw3` and `hdf5` libraries for fast Fourier transforms and efficient data storage. Users can configure models and parameters by modifying the `model.h` and `parameters.h` files.

For more details on the methodology and its applications, see my article:  
**"PSpectCosmo: A Pseudo-Spectral Code for Cosmological Dynamics Spanning Inflation and Reheating"**  
Available on arXiv: [arXiv:2411.17658](http://arxiv.org/abs/2411.17658).

## Features

- **Fast Fourier Transform (FFT)**: Utilizes `fftw3` for efficient Fourier transform computations.
- **Data Storage and Management**: Leverages the `hdf5` library for high-performance data storage.
- **Customizable Models and Parameters**: Allows users to define models and adjust parameters through `model.h` and `parameters.h`.

---

## Dependencies

Before compiling and running this program, ensure the following libraries are properly installed:

- **[fftw3](http://www.fftw.org/)**: Fast Fourier Transform library
- **[HDF5](https://www.hdfgroup.org/solutions/hdf5/)**: High-performance data storage library

---

## Installation

Follow these steps to install and run the program:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JieJiang-Cosmology/PSpectCosmo.git
   cd PSpectCosmo
   ```

2. **Install Dependencies**

   Ensure that `fftw3` and `hdf5` libraries are installed on your system. If they are not installed, you can do so as follows:

   - **Ubuntu/Debian Systems**
     ```bash
     sudo apt-get update
     sudo apt-get install libfftw3-dev libhdf5-dev
     ```

   - **MacOS (using Homebrew)**
     ```bash
     brew install fftw hdf5
     ```

   - **Windows**
     Refer to the [fftw](http://www.fftw.org/install/windows.html) and [hdf5](https://www.hdfgroup.org/downloads/hdf5/) official documentation for installation instructions.

3. **Compile the Program**

   Use `make` to compile the program:
   ```bash
   make
   ```

   This will generate an executable file, with the default name `PSpectCosmo`.

4. **Run the Program**

   Run the program from the command line:
   ```bash
   ./PSpectCosmo
   ```

---

## File Structure

- **`psectcosmo.cpp`**: Entry point of the program.
- **`model.h`**: Contains model configurations.
- **`parameters.h`**: Contains parameters for running the program.
- **`Makefile`**: Build configuration for compiling the program.
- **`README.md`**: Documentation for the program (this file).
- **`output/`**: Directory where output data is stored (default).

---

## Configuring Models and Parameters

1. **Model Settings (`model.h`)**

   Modify the `model.h` file to define the models required for the computation. 

2. **Parameter Settings (`parameters.h`)**

   Configure the runtime parameters in the `parameters.h` file. For example:
   ```cpp
   const int N = 64;    // FFT grid size
   const double kIR = 25; // Smallest wave number of the lattice
   ```

---

## Output

After the program completes, the results will be stored in the `output/` directory, with the default format being HDF5.

---

## Example Run

Assuming the `model.h` and `parameters.h` files are configured correctly, you can run the program using the following command:

```bash
./PSpectCosmo
```

Example output:
```
Directory "output" already exists, data will be saved there.
Directory "output/Snapshots" already exists, snapshots will be saved there.

Generating lattice initial conditions for new run at t = 0
The initial kinetic energy (sum of squares of field derivatives / 2) is: 0.332344
The initial potential energy is: 60.5
initial a = 1, ad = 4.50305, add = 19.7235
If you are simulating inflation, No k ^ 2 values are smaller than 10 a'' / a.

Press ENTER to continue
```

---

## FAQ

1. **Missing dependencies at runtime**
   - Ensure that `fftw3` and `hdf5` are properly installed, and their library paths are added to the environment variables.

2. **Compilation errors**
   - Check if the compiler settings in the `Makefile` are compatible with your system (default is `g++`).

3. **Cannot read output files**
   - Use HDF5-compatible tools (e.g., `h5py` or `hdfview`) to read the output files.

---

## Contributing

Feel free to submit issues or create pull requests to improve this project. You can contact me via:

- GitHub: [JieJiang-Cosmology](https://github.com/JieJiang-Cosmology)
- Email: jiejiang@pusan.ac.kr

---

## License

This project is licensed under the [MIT License](LICENSE). For more details, see the `LICENSE` file.
```
