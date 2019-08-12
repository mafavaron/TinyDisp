# TinyDisp parts

The _TinyDisp_ code is composed by two separate and interacting executables, *metpro*, the meteorological data preparer, and *td*, the actual dispersion model.

The former is assigned the task of taking a meteorological file input, in _TinyDisp_ form, and producing the rich meteorological input used by the model. The latter reads the rich meteorological input, plus domain and emission data, and yields a set of files containing ground concentrations and particle statistics, in forms allowing simple use.

A third component is represented by the model configuration file (common to the two executables), and all input files referenced by it.
