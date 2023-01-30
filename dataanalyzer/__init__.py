from configparser import ConfigParser
from pathlib import Path
import logging

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


import dataanalyzer

# from dataanalyzer.telemetry import start_telemetry


# CONFIG_PATH = Path(Path(dataanalyzer.__file__).parent) / "conf" / "telemetry.ini"

# telemetry_config = ConfigParser()
# telemetry_config.read(CONFIG_PATH)

# if telemetry_config["Telemetry"].getboolean("enabled"):
#     start_telemetry()

logger = logging.getLogger(__name__)
logger.info(f"Imported dataanalyzerversion: {__version__}")


from dataanalyzer.plotter import Plotter
from dataanalyzer.fitter import (
    Fitter,
    Fitparam,
    fitmodels,
    ExternalFunctions,
)
from dataanalyzer.utilities import (
    Valueclass,
    load_json_file,
    load_labber_file,
)
