import os.path

import sys
sys.path.append("./mly/")

from mly.waveforms import cbc

cbc(duration=0.99,
    fs=8192,
    detectors="H",
    massRange=[5,
               95],
    massStep= 1,
    rep=10,
    destinationDirectory="../skywarp_data/validation_injections",
    aproximant="IMRPhenomD",
    test=False)