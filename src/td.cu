
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


__device__ float windSpeed(float u, float v) {
	return sqrt(u*u + v*v);
}


__global__ void moveKernel(
	const float deltaT, const float Zi, const float H0,
	float* partX, float* partY, float* partZ,
	float* partU, float* partV, float* partW,
	float* partQ, float* partT,
	float* partSh, float* partSz, float* partEmissionTime,
	float* U, float* V, float* T,
	float* sU2, float* sV2, float* sW2, float* dsW2,
	float* eps, float* alfa, float* beta, float* gamma, float* delta,
	float* alfa_u, float* alfa_v,
	float* deltau, float* deltav, float* deltat,
	float* Au, float* Av, float* A, float* B
) {

	// Assign current particle index
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Check something is to be made on this particle
	if(partEmissionTime[i] < 0.f) return;

	// Compute wind directing cosines, and use them to
	// actuate horizontal movement
	float vel = windSpeed(U[i], V[i]);
	float cosa, sina;
	if(vel > 0.f) {
		cosa = U[i] / vel;
		sina = V[i] / vel;
	}
	else {
		cosa = 0.f;
		sina = 0.f;
	}
	partX[i] += (U[i] + partU[i]*cosa - partV[i]*sina) * deltaT;
	partY[i] += (V[i] + partU[i]*sina + partV[i]*cosa) * deltaT;
	partZ[i] += partW[i] * deltaT;

	// Apply reflection on ground and mixing height, assimung no-partial-èenetration
	if(partZ[i] < 0.f) {
		partZ[i] = -partZ[i];
		partW[i] = -partW[i];
	}
	if(partZ[i] > Zi && H0 > 0.f) {
		partZ[i] = 2.f*Zi - partZ[i];
		partW[i] = -partW[i];
	}

}


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
	std::vector<double> rXs;
	rXs.push_back(tConfig.GetDomainCenterX());
	std::vector<double> rYs;
	rYs.push_back(tConfig.GetDomainCenterY());
	std::vector<double> rZs;
	rZs.push_back(5.0);
	std::vector<double> rEs;
	rEs.push_back(1.0);

	// Generate particle pool, and prepare for simulation
	int iPartIdx = -1;
	int iPartNum = 0;
	int N = tConfig.GetPartPoolSize();
	thrust::device_vector<float> rvdPartX(N);
	thrust::device_vector<float> rvdPartY(N);
	thrust::device_vector<float> rvdPartZ(N);
	thrust::device_vector<float> rvdPartU(N);
	thrust::device_vector<float> rvdPartV(N);
	thrust::device_vector<float> rvdPartW(N);
	thrust::device_vector<float> rvdPartQ(N);
	thrust::device_vector<float> rvdPartT(N);
	thrust::device_vector<float> rvdPartSh(N);
	thrust::device_vector<float> rvdPartSz(N);
	thrust::device_vector<float> rvdPartEmissionTime(N);	// -1.0 for not yet filled particles

	// Associate pointers to particle device vectors
	float *ptr_rvdPartX = thrust::raw_pointer_cast(&rvdPartX[0]);
	float *ptr_rvdPartY = thrust::raw_pointer_cast(&rvdPartY[0]);
	float *ptr_rvdPartZ = thrust::raw_pointer_cast(&rvdPartZ[0]);
	float *ptr_rvdPartU = thrust::raw_pointer_cast(&rvdPartU[0]);
	float *ptr_rvdPartV = thrust::raw_pointer_cast(&rvdPartV[0]);
	float *ptr_rvdPartW = thrust::raw_pointer_cast(&rvdPartW[0]);
	float *ptr_rvdPartQ = thrust::raw_pointer_cast(&rvdPartQ[0]);
	float *ptr_rvdPartT = thrust::raw_pointer_cast(&rvdPartT[0]);
	float *ptr_rvdPartSh = thrust::raw_pointer_cast(&rvdPartSh[0]);
	float *ptr_rvdPartSz = thrust::raw_pointer_cast(&rvdPartSz[0]);
	float *ptr_rvdPartEmissionTime = thrust::raw_pointer_cast(&rvdPartEmissionTime[0]);

	// Reserve space for meteo information
  thrust::device_vector<float> rvdU;		  // Wind U components (m/s)
  thrust::device_vector<float> rvdV;		  // Wind V components (m/s)
  thrust::device_vector<float> rvdT;		  // Temperatures (K)
  thrust::device_vector<float> rvdSu2;		// var(U) values (m2/s2)
  thrust::device_vector<float> rvdSv2;		// var(V) values (m2/s2)
  thrust::device_vector<float> rvdSw2;		// var(W) values (m2/s2)
  thrust::device_vector<float> rvdDsw2;		// d var(W) / dz (m/s2)
  thrust::device_vector<float> rvdEps;		// TKE dissipation rate
  thrust::device_vector<float> rvdAlfa;		// Langevin equation coefficient
  thrust::device_vector<float> rvdBeta;		// Langevin equation coefficient
  thrust::device_vector<float> rvdGamma;	// Langevin equation coefficient
  thrust::device_vector<float> rvdDelta;	// Langevin equation coefficient
  thrust::device_vector<float> rvdAlfa_u;	// Langevin equation coefficient
  thrust::device_vector<float> rvdAlfa_v;	// Langevin equation coefficient
  thrust::device_vector<float> rvdDeltau;	// Langevin equation coefficient
  thrust::device_vector<float> rvdDeltav;	// Langevin equation coefficient
  thrust::device_vector<float> rvdDeltat;	// Langevin equation coefficient
  thrust::device_vector<float> rvdAu;			// exp(alfa_u*dt)
  thrust::device_vector<float> rvdAv;			// exp(alfa_v*dt)
  thrust::device_vector<float> rvdA;			// exp(alfa*dt)
  thrust::device_vector<float> rvdB;			// exp(beta*dt)
	thrust::device_vector<float> rvdVel;	  // Widn speed (m/s)

	// Associate host pointers to meteo (device) vectors,
	// to allow easy access during met profile propagation
	float *ptr_rvdU = thrust::raw_pointer_cast(&rvdU[0]);
	float *ptr_rvdV = thrust::raw_pointer_cast(&rvdV[0]);
	float *ptr_rvdT = thrust::raw_pointer_cast(&rvdT[0]);
	float *ptr_rvdSu2 = thrust::raw_pointer_cast(&rvdSu2[0]);
	float *ptr_rvdSv2 = thrust::raw_pointer_cast(&rvdSv2[0]);
	float *ptr_rvdSw2 = thrust::raw_pointer_cast(&rvdSw2[0]);
	float *ptr_rvdDsw2 = thrust::raw_pointer_cast(&rvdDsw2[0]);
	float *ptr_rvdEps = thrust::raw_pointer_cast(&rvdEps[0]);
	float *ptr_rvdAlfa = thrust::raw_pointer_cast(&rvdAlfa[0]);
	float *ptr_rvdBeta = thrust::raw_pointer_cast(&rvdBeta[0]);
	float *ptr_rvdGamma = thrust::raw_pointer_cast(&rvdGamma[0]);
	float *ptr_rvdDelta = thrust::raw_pointer_cast(&rvdDelta[0]);
	float *ptr_rvdAlfa_u = thrust::raw_pointer_cast(&rvdAlfa_u[0]);
	float *ptr_rvdAlfa_v = thrust::raw_pointer_cast(&rvdAlfa_v[0]);
	float *ptr_rvdDeltau = thrust::raw_pointer_cast(&rvdDeltau[0]);
	float *ptr_rvdDeltav = thrust::raw_pointer_cast(&rvdDeltav[0]);
	float *ptr_rvdDeltat = thrust::raw_pointer_cast(&rvdDeltat[0]);
	float *ptr_rvdAu = thrust::raw_pointer_cast(&rvdAu[0]);
	float *ptr_rvdAv = thrust::raw_pointer_cast(&rvdAv[0]);
	float *ptr_rvdA = thrust::raw_pointer_cast(&rvdA[0]);
	float *ptr_rvdB = thrust::raw_pointer_cast(&rvdB[0]);

	// Create random number generator, for use within loop
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 42ULL);

	// Assign data vectors their initial values
	thrust::fill(rvdPartX.begin(), rvdPartX.end(), 0.f);
	thrust::fill(rvdPartY.begin(), rvdPartY.end(), 0.f);
	thrust::fill(rvdPartZ.begin(), rvdPartZ.end(), 0.f);
	thrust::fill(rvdPartU.begin(), rvdPartU.end(), 0.f);
	thrust::fill(rvdPartV.begin(), rvdPartV.end(), 0.f);
	thrust::fill(rvdPartW.begin(), rvdPartW.end(), 0.f);
	thrust::fill(rvdPartQ.begin(), rvdPartQ.end(), 0.f);
	thrust::fill(rvdPartT.begin(), rvdPartT.end(), 0.f);
	thrust::fill(rvdPartSh.begin(), rvdPartSh.end(), 0.f);
	thrust::fill(rvdPartSz.begin(), rvdPartSz.end(), 0.f);
	thrust::fill(rvdPartEmissionTime.begin(), rvdPartEmissionTime.end(), -1.f);

	// Define larger-than-domain particle playground volume
	float rXmin = tConfig.GetX0();
	float rXmax = rXmin + tConfig.GetNx() * tConfig.GetDx();
	float rYmin = tConfig.GetY0();
	float rYmax = rYmin + tConfig.GetNy() * tConfig.GetDy();
	float rZmin = 0.0f;
	float rZmax = rZmin + tConfig.GetNz() * tConfig.GetDz();
	float ampliX = rXmax - rXmin;
	float ampliY = rYmax - rYmin;

	float ampliZ = rZmax - rZmin;
	float x0   = rXmin - ampliX / 2.0;
	float x1   = rXmax + ampliX / 2.0;
	float y0   = rYmin - ampliY / 2.0;
	float y1   = rYmax + ampliY / 2.0;
	float zbot = rZmin - ampliZ / 2.0;
	float ztop = rZmax + ampliZ / 2.0;
	zbot = (zbot < 0.0f)?0.0f:zbot;

	// Main loop
	MeteoData met(tConfig.GetNumZ());
	while(true) {

		// Get meteo data
		iRetCode = met.Read(fMeteoInputFile, tConfig.GetNumZ());
		if(iRetCode != 0) break;

		// Print the current time stamp
		time_t iEpoch = (time_t)met.GetTimeStamp();
		struct tm * tStamp = gmtime(&iEpoch);
		char buffer[64];
		strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tStamp);
		std::cout << buffer << std::endl;

		// Generate new particles
		for(int iSource = 0; iSource < rEs.size(); iSource++) {

			// Generate particles, directly in device memory
			for(int iPart = 0; iPart < tConfig.GetPartToEmitPerSource(); iPart++) {

				// Set counter and next particle index, forcing them to their reasonable constraints
				iPartIdx++;
				iPartIdx = (iPartIdx >= N) ? 0 : iPartIdx;
				iPartNum++;
				iPartNum = (iPartNum > N) ? N : iPartNum;

				// Actual particle generation
				rvdPartX[iPartIdx] = rXs[iSource];
				rvdPartY[iPartIdx] = rYs[iSource];
				rvdPartZ[iPartIdx] = rZs[iSource];
				rvdPartU[iPartIdx] = 0.0f;
				rvdPartV[iPartIdx] = 0.0f;
				rvdPartW[iPartIdx] = 0.0f;
				rvdPartQ[iPartIdx] = (rEs[iSource] / tConfig.GetPartToEmitPerSource()) * tConfig.GetTimeSubstepDuration();
				rvdPartT[iPartIdx] = 0.0f;
				rvdPartSh[iPartIdx] = 0.0f;
				rvdPartSz[iPartIdx] = 0.0f;
				rvdPartEmissionTime[iPartIdx] = met.GetTimeStamp();

			}

		}

		// Associate particles their meteo data
		for(int iPart = 0; iPart < iPartNum; iPart++) {
			iRetCode = met.Evaluate(
				rvdPartZ[iPart], tConfig.GetZ0(), tConfig.GetDz(), iPart,
				ptr_rvdU, ptr_rvdV, ptr_rvdT,
				ptr_rvdSu2, ptr_rvdSv2, ptr_rvdSw2, ptr_rvdDsw2,
				ptr_rvdEps, ptr_rvdAlfa, ptr_rvdBeta, ptr_rvdGamma, ptr_rvdDelta,
				ptr_rvdAlfa_u, ptr_rvdAlfa_v,
				ptr_rvdDeltau, ptr_rvdDeltav, ptr_rvdDeltat,
				ptr_rvdAu, ptr_rvdAv,
				ptr_rvdA, ptr_rvdB
			);
		}

		// Move particles

		// Write particles to movie file, if requested

		// Count ground concentrations, if required, and write them to concentration file

	}

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
