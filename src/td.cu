
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>

#include "cfg.h"
#include "meteodata.h"

int main(int argc, char** argv) {

	// Get input parameters
	if(argc != 2) {
		std::cerr << "td - The TinyDisp Particle Dispersion Model" << std::endl << std::endl;
		std::cerr << "Usage:" << std::endl << std::endl;
		std::cerr << "  [./]td <MeteoFileName>" << std::endl << std::endl;
		std::cerr << "Copyright 2019 by Mauri Favaron" << std::endl;
		std::cerr << "                  This is open-source software, covered by the MIT license" << std::endl << std::endl;
		return 1;
	}
	std::string sMetFileName = argv[1];

	// Read configuration
	std::ifstream fMeteoInputFile;
  fMeteoInputFile.open(sMetFileName, std::ios::in | std::ios::binary);
	Cfg tConfig = Cfg(fMeteoInputFile);
	if(tConfig.GetState() <= 0) {
		std::cerr << "Configuration file read failure" << std::endl;
		return 2;
	}
	int iRetCode = tConfig.Validate();
	if(iRetCode != 0) {
		std::cerr << "Configuration file validation failure, with code " << iRetCode << std::endl;
		return 3;
	}

	// Get emission data
	// For the moment, assume a unit emission from a pointwise source places at domain center
	// and 5m height above ground.
	double rXs = tConfig.GetDomainCenterX();
	double rYs = tConfig.GetDomainCenterY();
	double rZs = 5.0;
	double rEs = 1.0;

	// Generate particle pool, and prepare for simulation
	int N = tConfig.GetPartPoolSize();
	thrust::device_vector<float> rvdPartX(N);
	thrust::device_vector<float> rvdPartY(N);
	thrust::device_vector<float> rvdPartZ(N);

	// Main loop
	MeteoData met(tConfig.GetNumZ());
	while(true) {

		// Get meteo data
		iRetCode = met.Read(fMeteoInputFile, tConfig.GetNumZ());
		if(iRetCode != 0) break;
		time_t iEpoch = (time_t)met.GetTimeStamp();
		struct tm * tStamp = gmtime(&iEpoch);
		char buffer[64];
		strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tStamp);
		std::cout << buffer << std::endl;

		// Move particles

		// Write particles to movie file, if requested

		// Count ground concentrations, if required, and write them to concentration file

	}

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
