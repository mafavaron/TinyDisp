# TinyDisp
A lightweight particle atmospheric dispersion model.

# Core motivations
Extant Lagrangian particle models are complex pieces of software, and their use often demands large scale computing power. TinyDisp has been designed with a small footprint in mind, and this makes the model ideal for teaching, testing and research.

Small footprint does not necessarily mean _lightweight_ however: as any particle model, TinyDisp accuracy gets larger with increasing particle number.

For statistical reasons, a scaling in the order of SQRT(N) may be expected in accuracy, with N the number of particles. This means that if you want to double accuracy, you have to use 4 times the original particle number. This may lead to sensible particle numbers in the order of millions.

# Hardware

To make TinyPart a sensibly quick system something needs being made, and the way I've devised is, using special hardware, namely NVIDIA's CUDA.

The main core code has been developed and tested on a small-size Jetson Nano board, but the code may be recompiled and used on any CUDA-enabled computer, hosting NVIDIA development kit; incidentally, this is the running profile I recommend for real performance.

# Basic requirements
Model equations are designed to operate at values higher than Lagrangian decorrelation time, assuming wind components being uncorrelated. As a consequence, the minimum time step is 300s (5min), to ensure correlation among different momentum components are nearly zero.

Meteorology is restricted to 1-dimensional, composed by one "surface" and one "vertical profile" file (the latter may be missing). Most typically, vertical profiles are considered up to 700m, but may be less - typically, where SODAR measurements are available. Both surface and vertical profile files are allowed to contain gaps: in this case, values are linearly interpolated between consecutive valid values. Vertical profiles are considered defined over a fixed and invariable set of measurement heights.

Sources may be elevated points (stacks) or areas. Area geometry is either rectangular, or circular. Emission may occur at ambient temperature (as often happens with odor dispersion), or at an assigned value. In the latter case, buoyancy effects are modeled.

# Programming language and recommended hardware
Given its experimental, "scientific" nature TinyDisp is conceived to run on a wide set of architectures, ranging in principle to a Raspberry Pi to a supercomputer.

Execution is performed sequentially, and only single-core optimizations (e.g. using SIMD extensions found on many processors) are allowed. This restriction is intentional, to maintain the model architecture as simple as possible.

In future provisions may be taken for porting TinyDisp to a many-core implementation.

To ease this possible transition the TinyDisp model is coded in plain C.

The reference compilers are the GNU C Compiler and the Intel C/C++ compiler (community edition for Linux). 

# Nature and development cycle
TinyDisp is a work-in-progress, and will likely remain so until possible.

Development will occur according to an evolutionary path. At any moment, it is guaranteed the code is compilable and, to the extent allowed by the source as-it-will-be at the moment, testable. Major functional achievements, turning the model into something "useful", will be communicated as soon as possible.

# Relationship to other open-source projects
TinyDisp uses the following open-source code:
* 'ini.h/ini.c' and 'INIReader.h/INIReader.cpp' (https://github.com/benhoyt/inih)
* _pbl_met_ (https://github.com/serv_terr/pbl_met)
I Our appreciation goes to their authors and maintainers.

