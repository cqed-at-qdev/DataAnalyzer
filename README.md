# Data Analyzer
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=green"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Malthe&color=inactive"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=Malthe&color=inactive"/>
</p>

# Description
Data Analyzer is a plotting and fitting tool made as a wrapper around matplotlib and the iminuit fitting library. It aims to provide a easy to use way to visualize lab data, by automatically transforming the data into the most appropriate units when plotting and must more. The module is centered around ValueClass objects.

The module contains of the following features:

* Valueclass objects
    * A container for data, giving the functionality seen in Labber and more.

* Data conversions tools 
    * Includes conversion functions for JSON, HDF5 (Labber) and more.

* Robust fitting tools
    * With a functions library that can easily be expanded with new models.

* Custom plotting tools
    * As a wrapper around matplotlib, giving the ability to use all matplotlib features.

**Module documentation can be found at [HER]() (Not made yet).**

## Valueclass objects
The valueclass is the main component in the library. It contains values and errors of data. Moreover, it contains the parameters name, unit and information about the data type. Valueclass objects are created automatically in the data conversion functions or manually. They are created manually like this:

```python
from DataAnalyser import Valueclass
freq = Valueclass(name="Frequency", unit="Hz", value=[5.64e9, 5.65e9, 5.65e9])
```

Now you can call the name and unit from the Valueclass as:

```python
freq.name -> "Frequency"
freq.unit -> "Hz"
```

The values can likewise be called:

```python
freq.value -> np.array([5.64e9, 5.65e9, 5.65e9])
```

As the errors are not defined, they will be set to np.nan:

```python
freq.error -> np.array([np.nan, np.nan, np.nan])
```

The data contained within the valueclass can be plotted by calling:

```python
freq.plot() -> matplotlib.pyplot.plot(x=np.arange(len(freq.value)), y=freq.value)
```

## Data conversion
The library contains a number of methods to convert, import and export data. These methods are:
* Valueclass ↔ Dict
* Valueclass ↔ Json
* Valueclass ↔ Labber (hdf5)
* Json ↔ Labber

Conversion of data from a Labber file is done like this:
```python
from DataAnalizer import load_labber_file

labber_file_path = r"somewhere\on\your\computer\dummy_labber_file.hdf5"
parameters, results = load_labber_file(labber_file_path)
```

Here `parameters` is a list of Valueclasses of the parameteres (stepchannels) being sweept in the experiment. Likewise, `results` is a list of Valueclasses of the data from the experiment (logchannels).


# Installation

# Usage

## Running the tests

If you have gotten 'dataanalyzer' from source, you may run the tests locally.

Install `dataanalyzer` along with its test dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r test_requirements.txt
```

Then run `pytest` in the `tests` folder.

## Building the documentation

If you have gotten `dataanalyzer` from source, you may build the docs locally.

Install `dataanalyzer` along with its documentation dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r docs_requirements.txt
```

You also need to install `pandoc`. If you are using `conda`, that can be achieved by

```bash
$ conda install pandoc
```
else, see [here](https://pandoc.org/installing.html) for pandoc's installation instructions.

Then run `make html` in the `docs` folder. The next time you build the documentation, remember to run `make clean` before you run `make html`.
