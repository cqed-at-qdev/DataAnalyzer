"""
Installs the DataAnalyzer[D[D[D[D[D[D[D[D[D[D[d[C[C[C[C[C[C[C[D[D[a[C[C[C[C[C[C[C package
"""

from setuptools import setup, find_packages
from pathlib import Path

import versioneer

readme_file_path = Path(__file__).absolute().parent / "README.md"

required_packages = ['opencensus-ext-azure']
package_data = {"DataAnalyzer[D[D[D[D[D[D[D[D[D[D[d[C[C[C[C[C[C[C[D[D[a[C[C[C[C[C[C[C": ["conf/telemetry.ini"] }


setup(
    name="DataAnalyzer[D[D[D[D[D[D[D[D[D[D[d[C[C[C[C[C[C[C[D[D[a[C[C[C[C[C[C[C",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.9",
    install_requires=required_packages,
    author= "Malthe Asmus Marciniak Nielsen",
    author_email="vpq602@alumni.ku.dk",
    description="Plotter and Fitter for data",
    long_description=readme_file_path.open().read(),
    long_description_content_type="text/markdown",
    license="",
    package_data=package_data,
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
    ],
)
