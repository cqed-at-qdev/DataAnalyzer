import Labber
import numpy as np

vTime = np.linspace(0, 1, 501)
vFreq = np.linspace(1, 10, 10)
chTime = dict(name="Time", unit="s", values=vTime)
chFreq = dict(name="Frequency", unit="Hz", values=vFreq)
chSig = dict(name="Signal", unit="V", vector=False)

f = Labber.createLogFile_ForData("TestData", [chSig], [chTime, chFreq])

for freq in vFreq:
    data = {"Signal": np.sin(2 * np.pi * freq * vTime)}
    f.addEntry(data)

f.getFilePath("")
