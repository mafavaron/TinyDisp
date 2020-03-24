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
global iNumParticlePools
global xMin
global xMax
global yMin
global yMax
global fig


def connect(sDataFile):

    global fParticles
    global iNumParticlePools

    # Assume success (will falsify on failure)
    iRetCode = 0

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
    try:
        ivBuffer = fParticles.read(4)
        iNumParticlePools = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 2
        fParticles.close()
        return iRetCode

    return iRetCode


def update(iNumFrame):

    global fParticles
    global xMin
    global xMax
    global yMin
    global yMax
    global fig
    global iNumParticlePools

    # Get additional data
    try:
        ivBuffer = fParticles.read(4)
        iIteration = struct.unpack('i', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        iCurTime = struct.unpack('i', ivBuffer)[0]
    except:
        fParticles.close()
        return
    try:
        ivBuffer = fParticles.read(4)
        rU = struct.unpack('f', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        rV = struct.unpack('f', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        rStdDevU = struct.unpack('f', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        rStdDevV = struct.unpack('f', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        rCovUV = struct.unpack('f', ivBuffer)[0]
    except:
        return
    try:
        ivBuffer = fParticles.read(4)
        iNumPart = struct.unpack('i', ivBuffer)[0]
    except:
        return
    if iNumPart <= 0:
        return
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
        return

    # Plot current particle set
    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots()
    ax.scatter(rvX, rvY, s=0.5, alpha=0.5)
    ax.set_xlim((xMin,xMax))
    ax.set_ylim((yMin,yMax))
    ax.set_aspect('equal')

    # Tell users which step is this
    print("Frame %d of %d generated" % (i, iNumParticlePools))

    return


if __name__ == "__main__":

    global iNumParticlePools
    global fig

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

    # Gather essential preliminary data from output file
    iRetCode = connect(sDataFile)
    if iRetCode != 0:
        print('Error accessing particle file - Return code: %d' % iRetCode)
        sys.exit(3)

    # Run animation
    anim = animation.FuncAnimation(fig, update, interval=20, frmes=iNumParticlePools, blit=True)
    print('Animation completed: generating movie')
    anim.save(sMp4File, 'ffmpeg')
