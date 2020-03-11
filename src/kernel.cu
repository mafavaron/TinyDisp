
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Config.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

#include <iostream>
#include <math.h>

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
    thrust::device_vector<int> ivPartTimeStamp(iNumPart); // Time stamp at emission time - for reporting - host-only
    thrust::device_vector<float> rvPartX(iNumPart);
    thrust::device_vector<float> rvPartY(iNumPart);
    thrust::device_vector<float> rvPartU(iNumPart);
    thrust::device_vector<float> rvPartV(iNumPart);
    thrust::device_vector<float> rvN1(iNumPart);
    thrust::device_vector<float> rvN2(iNumPart);
    thrust::device_vector<float> rvX1(iNumPart);
    thrust::device_vector<float> rvDeltaU(iNumPart);
    thrust::device_vector<float> rvDeltaV(iNumPart);

    // Main loop: iterate over meteo data
    int iNumData = tCfg.GetNumMeteoData();
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    unsigned int iIteration = 0;
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

        // Generate bivariate normal deviates
        // -1- First of all, generate two sets of random normals, with mu=0 and sigma=1
        thrust::transform(
            index_sequence_begin + iIteration * iNumPart,
            index_sequence_begin + iIteration * (iNumPart + 1),
            rvN1.begin(),
            normal_deviate(0.0f, 1.0f)
        );
        thrust::transform(
            index_sequence_begin + iIteration * (iNumPart + 2),
            index_sequence_begin + iIteration * (iNumPart + 3),
            rvN2.begin(),
            normal_deviate(0.0f, 1.0f)
        );
        iIteration++;
        // -1- Transform the two independent samples in a 2D bivariate sample
        float rho;
        if (rStdDevU > 0.f && rStdDevV > 0.f) {
            rho = rCovUV / (rStdDevU * rStdDevV);
        }
        else {
            rho = 0.f;
        }
        float lambda = (rStdDevV / rStdDevU) * rho;
        float nu = sqrtf((1.0f - rho * rho) * rStdDevV * rStdDevV);
        rvX1 = rvN1;
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(rStdDevU), rvX1.begin(), thrust::multiplies<float>());
        rvDeltaU = rvX1;
        rvDeltaV = rV + lambda * (rvX1 - rU) + nu * rvN2;

        // Move particles

        // Count in cells

        // Inform users of the progress
        std::cout << iIteration << ", " << iTimeStamp << ", " << rU << ", " << rV << ", " << rStdDevU << ", " << rStdDevV << ", " << rCovUV << std::endl;
    }

    // Deallocate manually thrust resources
    // -1- Reclaim workspace
    ivPartTimeStamp.clear();
    rvPartX.clear();
    rvPartY.clear();
    rvPartU.clear();
    rvPartV.clear();
    rvN1.clear();
    rvN2.clear();
    // -1- Clear any other resources
    ivPartTimeStamp.shrink_to_fit();
    rvPartX.shrink_to_fit();
    rvPartY.shrink_to_fit();
    rvPartU.shrink_to_fit();
    rvPartV.shrink_to_fit();
    rvN1.shrink_to_fit();
    rvN2.shrink_to_fit();

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
