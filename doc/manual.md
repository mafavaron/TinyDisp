# TinyDisp parts

The _TinyDisp_ code is composed by two separate and interacting executables, *metpro*, the meteorological data preparer, and *td*, the actual dispersion model.

The former is assigned the task of taking a meteorological file input, in _TinyDisp_ form, and producing the rich meteorological input used by the model. The latter reads the rich meteorological input, plus domain and emission data, and yields a set of files containing ground concentrations and particle statistics, in forms allowing simple use.

A third component is represented by the model configuration file (common to the two executables), and all input files referenced by it.

# Running the TinyDisp parts

# The main TinyDisp configuration file

# Input files

## Meteorological data

## Emission data

### Static emissions

### Dynamic emissions

# Intermediate files

## Intermediate meteo file

# Result files

# Preparing movies

One of the coolest features of _TinyDisp_ is its ability to generate movies, one snapshot a time, showing the time evolution of particles as they move across a domain.


