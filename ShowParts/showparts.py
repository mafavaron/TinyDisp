#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import configparser
import os
import sys
import struct

# System state
global fParticles
global sDataFile
global rEdgeLength
global bDataOK


def init():

    global fParticles

    # Assume success (will falsify on failure)
    iRetCode = 0
    bDataOK  = False

    # Start reading the particle binary file
    try:
        fParticles = open(sDataFile, "rb")
    except Exception as e:
        print(str(e))
        iRetCode = 1
        return iRetCode
    try:
        ivBuffer = fParticles.read(4)
        iMaxPart = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 2
        fParticles.close()
        return iRetCode

    bDataOK = True
    return iRetCode


def update(iNumFrame):

    global fParticles

    # Assume success (will falsify on failure)
    iRetCode = 0

    # Get additional data
    try:
        ivBuffer = fParticles.read(4)
        iNumPart = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 1
        fParticles.close()
        return iRetCode, None, None, None
    if iNumPart <= 0:
        iRetCode = 2
        return iRetCode, None, None, None
    sRealFmt = "%df" % iNumPart
    sIntFmt  = "%di" % iNumPart
    try:
        bvBuffer = fParticles.read(4 * iNumPart)
        rvX = np.array(struct.unpack(sRealFmt, bvBuffer))
        bvBuffer = fParticles.read(4 * iNumPart)
        rvY = np.array(struct.unpack(sRealFmt, bvBuffer))
        bvBuffer = fParticles.read(4 * iNumPart)
        ivTimeStamp = np.array(struct.unpack(sIntFmt, bvBuffer))
    except:
        iRetCode = -1
        fParticles.close()
        return iRetCode, None, None, None

    return iRetCode, rvX, rvY, ivTimeStamp


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
        sDataFile = cfg["General"]["ParticlesFile"]
    except:
        print("Error: Data file name not found in configuration file")
        sys.exit(2)
    try:
        rEdgeLength = float(cfg["General"]["EdgeLength"])
    except:
        print("Error: Edge length not found or invalid in configuration file")
        sys.exit(2)

    # Define coordinates area
    xMin = -rEdgeLength / 2.0
    xMax = -xMin
    yMin =  xMin
    yMax =  xMax

    # Initialize plotting environment
    plt.style.use('seaborn-pastel')
    figure = plt.figure()
    axes   = plt.axes(xlim=(xMin,xMax), ylim=(yMin,yMax))
    line,  = axes.plot([], [], lw=3)

    iRetCode = init()
    print("Init - Return code: %d" % iRetCode)

    iRetCode = update(1)
    print("Updt - Return code: %d" % iRetCode)
