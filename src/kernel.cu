
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Config.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>

struct normal_deviate {

    float mu, sigma;

    __host__ __device__ normal_deviate(float _mu = 0.0f, float _sigma = 1.0f) : mu(_mu), sigma(_sigma) {};

    __device__ float operator()(unsigned int n) {

        thrust::default_random_engine engine;
        thrust::normal_distribution<float> dist(mu, sigma);
        engine.discard(n);

        return dist(engine);

    }

};

int main(int argc, char** argv)
{

    // Get input parameters
    if (argc != 2) {
        std::cerr << "TinyDisp - A simple, local airflow visualizer" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Usage:" << std::endl;
        std::cerr << std::endl;
        std::cerr << "    td <RunConfiguration>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Copyright 2020 by Servizi Territorio srl" << std::endl;
        std::cerr << "                  This is open-source software, covered by the MIT license" << std::endl;
        return 1;
    }
    std::string sCfgFile = argv[1];

    // Gather configuration (and meteo data)
    Config tCfg(sCfgFile);

    // Particle pool
    int iNumPart  = tCfg.GetParticlePoolSize();
    int iNextPart = 0;  // For indexing the generation circular buffer
    thrust::host_vector<int> ivPartTimeStamp(iNumPart); // Time stamp at emission time - for reporting - host-only
    thrust::device_vector<float> rvPartX(iNumPart);
    thrust::device_vector<float> rvPartY(iNumPart);
    thrust::device_vector<float> rvPartU(iNumPart);
    thrust::device_vector<float> rvPartV(iNumPart);

    // Initialize random number generator


    // Main loop: iterate over meteo data
    int iNumData = tCfg.GetNumMeteoData();
    for (auto i = 0; i < iNumData; i++) {

        // Get current meteorology
        int iTimeStamp;
        float rU;
        float rV;
        float rStdDevU;
        float rStdDevV;
        float rCovUV;
        bool lOk = tCfg.GetMeteo(i, iTimeStamp, rU, rV, rStdDevU, rStdDevV, rCovUV);

        // Emit new particles
        thrust::fill(ivPartTimeStamp.begin() + iNextPart, ivPartTimeStamp.begin() + iNextPart + tCfg.GetNumNewParticles(), iTimeStamp);
        thrust::fill(rvPartX.begin() + iNextPart, rvPartX.begin() + iNextPart + tCfg.GetNumNewParticles(), 0.0f);
        thrust::fill(rvPartY.begin() + iNextPart, rvPartY.begin() + iNextPart + tCfg.GetNumNewParticles(), 0.0f);
        thrust::fill(rvPartU.begin() + iNextPart, rvPartU.begin() + iNextPart + tCfg.GetNumNewParticles(), rU);
        thrust::fill(rvPartV.begin() + iNextPart, rvPartV.begin() + iNextPart + tCfg.GetNumNewParticles(), rV);
        iNextPart += tCfg.GetNumNewParticles();
        if (iNextPart >= iNumPart) {
            iNextPart = 0;
        }

        // Move particles

        // Count in cells

        // Inform users of the progress
        std::cout << iTimeStamp << ", " << rU << ", " << rV << ", " << rStdDevU << ", " << rStdDevV << ", " << rCovUV << std::endl;
    }

    // Leave
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}

