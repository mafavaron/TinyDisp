#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import configparser
import os
import sys

# System state
global fParticles
global sDataFile
global rEdgeLength


def init():

    # Assume success (will falsify on failure)
    iRetCode = 0

    # Start reading the particle binary file
    try:
        fParticles = open(sDataFile, "rb")
    except:
        iRetCode = 1
        return iRetCode
    try:
        iMaxPart = np.fromfile(fParticles, dtype=np.int32)
    except:
        iRetCode = 2
        fParticles.close()
        return iRetCode

    print(iMaxPart)

    return iRetCode

if __name__ == "__main__":

    # Get parameters
    if len(sys.argv) != 3:
        print("showparts.py - Movie builder for TinyDisp airflow visualizer")
        print()
        print("Usage:")
        print()
        print("  python3 showparts.py <ConfigFile> <OutputMovie.mp4>")
        print()
        print("Copyright 2020 by Servizi Territorio srl")
        print("This is open-source software, covered by the MIT license.")
        print()
        sys.exit(1)
    sCfgFile = sys.argv[1]
    sMp4File = sys.argv[2]

    # Get configuration data
    cfg = configparser.ConfigParser()
    try:
        cfg.read(sCfgFile)
    except:
        print("Error: Configuration file not read")
        sys.exit(2)
    try:
        sDataFile = cfg["General"]["DataFile"]
    except:
        print("Error: Data file name not found in configuration file")
        sys.exit(2)
    try:
        rEdgeLength = float(cfg["General"]["EdgeLength"])
    except:
        print("Error: Edge length not found or invalid in configuration file")
        sys.exit(2)
