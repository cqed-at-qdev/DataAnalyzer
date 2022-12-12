# -----------------------------------------------------------------------------
# Copyright © 2009- The Spyder Development Team
#
# Licensed under the terms of the MIT License
# (see LICENSE.txt for details)
# -----------------------------------------------------------------------------
"""Provides QtQuickWidgets classes and functions."""

# Local imports
from . import PYQT5, PYQT6, PYSIDE6, PYSIDE2, PythonQtError

if PYQT5:
    from PyQt5.QtQuickWidgets import *
elif PYQT6:
    from PyQt6.QtQuickWidgets import *
elif PYSIDE6:
    from PySide6.QtQuickWidgets import *
elif PYSIDE2:
    from PySide2.QtQuickWidgets import *
else:
    raise PythonQtError('No Qt bindings could be found')
