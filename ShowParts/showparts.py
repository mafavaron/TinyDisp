#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from celluloid import Camera
import configparser
import os
import sys
import struct


def connect_particles_file(sDataFile):
    # Assume success (will falsify on failure)
    iRetCode = 0

    # Start reading the particle binary file
    try:
        fParticles = open(sDataFile, "rb")
    except Exception as e:
        iRetCode = 1
        return iRetCode, None, None
    try:
        ivBuffer = fParticles.read(4)
        iMaxPart = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 2
        fParticles.close()
        return iRetCode, None, None
    try:
        ivBuffer = fParticles.read(4)
        iNumParticlePools = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 2
        fParticles.close()
        return iRetCode, None, None

    return iRetCode, fParticles, iNumParticlePools


def get_next_particles(fParticles):
    # Function, gathering the next particle pool

    # Assume success (will falsify on failure
    iRetCode = 0

    # Get additional data
    try:
        ivBuffer = fParticles.read(4)
        iIteration = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        iCurTime = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        rU = struct.unpack('f', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        rV = struct.unpack('f', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        rStdDevU = struct.unpack('f', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        rStdDevV = struct.unpack('f', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        rCovUV = struct.unpack('f', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    try:
        ivBuffer = fParticles.read(4)
        iNumPart = struct.unpack('i', ivBuffer)[0]
    except:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    if iNumPart <= 0:
        iRetCode = 1
        return iRetCode, None, None, None, None, None, None, None, None, None, None
    sRealFmt = "%df" % iNumPart
    sIntFmt = "%di" % iNumPart
    try:
        bvBuffer = fParticles.read(4 * iNumPart)
        rvX = np.array(struct.unpack(sRealFmt, bvBuffer))
        bvBuffer = fParticles.read(4 * iNumPart)
        rvY = np.array(struct.unpack(sRealFmt, bvBuffer))
        bvBuffer = fParticles.read(4 * iNumPart)
        ivTimeStamp = np.array(struct.unpack(sIntFmt, bvBuffer))
    except:
        iRetCode = 2
        return iRetCode, None, None, None, None, None, None, None, None, None, None

    return iRetCode, iIteration, iCurTime, rU, rV, rStdDevU, rStdDevV, rCovUV, rvX, rvY, ivTimeStamp


if __name__ == "__main__":

    global iNumParticlePools
    global ax

    # Get parameters
    if len(sys.argv) != 3:
        print("showparts.py - Movie builder for TinyDisp airflow visualizer")
        print()
        print("Usage:")
        print()
        print("  python3 showparts.py <ConfigFile> <OutputMovie.gif>")
        print()
        print("Copyright 2020 by Servizi Territorio srl")
        print("This is open-source software, covered by the MIT license.")
        print()
        sys.exit(1)
    sCfgFile = sys.argv[1]
    sGifFile = sys.argv[2]

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
    yMin = xMin
    yMax = xMax

    # Gather essential preliminary data from output file
    iRetCode, fParticles, iNumParticlePools = connect_particles_file(sDataFile)
    if iRetCode != 0:
        print('Error accessing particle file - Return code: %d' % iRetCode)
        sys.exit(3)

    # Generate individual frames
    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots()
    camera = Camera(fig)
    for iNumIter in range(iNumParticlePools):

        # Try getting new particle pool
        iRetCode, iIteration, iCurTime, rU, rV, rStdDevU, rStdDevV, rCovUV, rvX, rvY, ivTimeStamp = get_next_particles(fParticles)

        # Plot current particle pool
        ax.scatter(rvX, rvY, s=0.75, c='red', alpha=0.05)
        ax.set_xlim((xMin, xMax))
        ax.set_ylim((yMin, yMax))
        ax.set_aspect('equal')

        # Take snapshot
        camera.snap()

        # Tell users which step is this
        print("Frame %d of %d generated - Min/Max X: %f/%f - Min/Max Y: %f/%f" % (iNumIter, iNumParticlePools, np.min(rvX), np.max(rvX), np.min(rvY), np.max(rvY)))

    print('Animation completed: generating movie')
    anim = camera.animate(blit=True)

    print('Movie generated: saving it to ' + sGifFile)
    anim.save(sGifFile, writer='PillowWriter', fps=50)

    print("*** END JOB ***")
