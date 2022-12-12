"""
Installs the dataanalyzer package
"""

from setuptools import setup, find_packages
from pathlib import Path

import versioneer

readme_file_path = Path(__file__).absolute().parent / "README.md"

required_packages = ['opencensus-ext-azure']
package_data = {"dataanalyzer": ["conf/telemetry.ini"] }


setup(
    name="dataanalyzer",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.9",
    install_requires=required_packages,
    author= "Malthe Asums Marciniak Nielsen",
    author_email="vpq602@alumni.ku.dk",
    description="package_description",
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
