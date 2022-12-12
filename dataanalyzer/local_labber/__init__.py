# Copyright 2014-2021 Keysight Technologies
# This file is part of the Labber API, a Python interface to Labber.
#
# http://keysight.com/find/labber
#
# All rights reserved

"""
This is a Python interface to Labber.
Software for Instrument Control and Lab Automation.
"""


from __future__ import absolute_import as _ai
from sys import version_info as _info

import os

os.environ["QT_API"] = "pyqt5"

# get import folder depending python version
_VERSION_ERROR_STRING = "The Labber API requires Python >=3.6 and <=3.9"

if _info >= (3, 10) or _info < (3, 9):
    raise ImportError(_VERSION_ERROR_STRING)

# version info
from ._include39._version import version as __version__  # type: ignore
from ._include39._version import info as version  # type: ignore

# script tools
from ._include39 import _ScriptTools as ScriptTools  # type: ignore

# log file
from ._include39._LogFile import (  # type: ignore
    LogFile,
    createLogFile_ForData,
    getTraceDict,
)

# labber client
from ._include39._Client import connectToServer  # type: ignore

# scripting with scenarios
from ._include39 import _config as config  # type: ignore
from ._include39.labber.config.scenario import Scenario  # type: ignore
