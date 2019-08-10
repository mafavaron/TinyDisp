
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <math.h>
#include <string>

#include "INIReader.h"

#define N (1<<20)

int main(int argc, char** argv) {

	// Get input parameters
	if(argc != 2) {
		std::cerr << "td - The TinyDisp Particle Dispersion Model" << std::endl << std::endl;
		std::cerr << "Usage:" << std::endl << std::endl;
		std::cerr << "  [./]td <InpFileName>" << std::endl << std::endl;
		std::cerr << "Copyright 2019 by Mauri Favaron" << std::endl;
		std::cerr << "                  This is open-source software, covered by the MIT license" << std::endl << std::endl;
		return 1;
	}
	std::string sCfgFileName = argv[1];

	// Read configuration
	INIReader cfg = INIReader(sCfgFileName);
	// -1- General
	std::string sSection = "General";
	std::string sName    = "debug_level";
	int iDebugLevel = cfg.GetInteger(sSection, sName, 4);
	sName = "diafile";
	std::string sDefault = "";
	std::string sDiaFile = cfg.GetString(sSection, sName, sDefault);
	sName = "frame_interval";
	int iFrameInterval = cfg.GetInteger(sSection, sName, 0);
	sName = "frame_path";
	std::string sFramePath = cfg.GetString(sSection, sName, sDefault);
	sName = "exec_mode";
	int iExecMode = cfg.GetInteger(sSection, sName, 0);
	// -1- Timing
	sSection = "Timing";
	sName = "avgtime";
	int iAvgTime = cfg.GetInteger(sSection, sName, 0);
	sName = "nstep";
	int iNumStep = cfg.GetInteger(sSection, sName, 0);
	sName = "npart";
	int iNumPart = cfg.GetInteger(sSection, sName, 0);
	sName = "maxage";
	int iMaxAge = cfg.GetInteger(sSection, sName, 0);
	// -1- Emission
	sSection = "Emission";
	sName = "static";
	std::string sStatic = cfg.GetString(sSection, sName, sDefault);
	sName = "dynamic";
	std::string sDynamic = cfg.GetString(sSection, sName, sDefault);
	// -1- Meteo
	sSection = "Meteo";
	sName = "inpfile";
	std::string sMetInpFile = cfg.GetString(sSection, sName, sDefault);
	sName = "outfile";
	std::string sMetOutFile = cfg.GetString(sSection, sName, sDefault);
	sName = "diafile";
	std::string sMetDiaFile = cfg.GetString(sSection, sName, sDefault);
	sName = "height";
	double rHeight = cfg.GetReal(sSection, sName, 0.0);
	sName = "z0";
	double rZ0 = cfg.GetReal(sSection, sName, 0.0);
	sName = "zr";
	double rZr = cfg.GetReal(sSection, sName, 0.0);
	sName = "zt";
	double rZt = cfg.GetReal(sSection, sName, 0.0);
	sName = "gamma";
	double rGamma = cfg.GetReal(sSection, sName, 0.0);
	sName = "hemisphere";
	int iHemisphere = cfg.GetInteger(sSection, sName, 0);
	// -1- Output
	sSection = "Output";
	sName = "conc";
	std::string sConcFile = cfg.GetString(sSection, sName, sDefault);
	sName = "x0";
	double rY0 = cfg.GetReal(sSection, sName, 0.0);
	sName = "y0";
	double rX0 = cfg.GetReal(sSection, sName, 0.0);
	sName = "nx";
	int iNx = cfg.GetInteger(sSection, sName, 0);
	sName = "ny";
	int iNy = cfg.GetInteger(sSection, sName, 0);
	sName = "dx";
	double rDx = cfg.GetReal(sSection, sName, 0.0);
	sName = "dy";
	double rDy = cfg.GetReal(sSection, sName, 0.0);
	sName = "nz";
	int iNz = cfg.GetInteger(sSection, sName, 0);
	sName = "dz";
	double rDz = cfg.GetReal(sSection, sName, 0.0);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 42ULL);

	thrust::device_vector<float> dvec_x(N);
	thrust::device_vector<float> dvec_y(N);

	float *ptr_x = thrust::raw_pointer_cast(&dvec_x[0]);
	float *ptr_y = thrust::raw_pointer_cast(&dvec_y[0]);

	// Simulate particles emission
	curandGenerateUniform(gen, ptr_x, N);
	curandGenerateUniform(gen, ptr_y, N);
	curandDestroyGenerator(gen);

	// Perform an aggregation function
	int insideCount = thrust::count_if(
		thrust::make_zip_iterator(thrust::make_tuple(dvec_x.begin(), dvec_y.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(dvec_x.end(), dvec_y.end())),
		[]__device__(const thrust::tuple<float, float> &el) {
			return(pow(thrust::get<0>(el), 2) + pow(thrust::get<1>(el), 2)) < 1.f;
		}
	);

	// Result...
	std::cout << "Pi = " << insideCount * 4.f / N << std::endl;

	// Leave
	return 0;

}
