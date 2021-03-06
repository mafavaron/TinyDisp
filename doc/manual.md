# TinyDisp parts

The _TinyDisp_ code is composed by two separate and interacting executables, *metpro*, the meteorological data preparer, and *td*, the actual dispersion model.

The former is assigned the task of taking a meteorological file input, in _TinyDisp_ form, and producing the rich meteorological input used by the model. The latter reads the rich meteorological input, plus domain and emission data, and yields a set of files containing ground concentrations and particle statistics, in forms allowing simple use. The two executables operate in sequence, exchanging information through a self-documenting meteorological file.

A third component is represented by the model configuration file (common to the two executables), and all input files referenced by it.

# Running the TinyDisp parts

To run the meteorological data preparer the following command should be used from a terminal window:

./metpre <TinyDisp_Ini_File>

To run the dispersion model, the following command should be used (also in a terminal window):

./td <Meteo_File>

<TinyDisp_Ini_File> represents the name of the main input file to TinyDisp, and <Meteo_File> is the meteorological file produced by _metpre_, and referenced in <TinyDisp_Ini_File>.

The two startup commands refer to the UNIX (OS/X and Linux) case. In Windows, he "./" prefix may be omitted.

# The main TinyDisp configuration file

The main TinyDisp configuration file is an INI text file containing all the basic data specifying a TinyDisp run. As any INI file, the main configuration input articulates in _sections_, described in this chapter' sections.

## The _[General]_ section: TinyDisp run environment

The items in this section, used by both _metpre_ and _td_, are:

* "debug_level": Integer value, ranging from 0 (no debug) to 3 (maximum debug information). Intermediate values 1 and 2 signify increasing levels of debug information. The output generated by executables gets larger and larger as the debug level increases.
* "diafile": string, containing the name of the _td_ diagnostic file.
* "frame_interval": Integer value, equal to 0 (no movie made) or the number of sub-steps at which movie snaps are taken. A very common value for movie production is 1.

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


