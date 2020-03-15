
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Config.h"
#include "FileMgr.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
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

    // Get snapshots pathname and, if non-empty, ensure
    // it exists and is cleaned before to start
    bool lSnapshotsCreated = false;
    FileMgr tSnapshots;
    std::string sSnapshots = tCfg.GetSnapshotsPath();
    std::string sVisItFile = "";
    if (!sSnapshots.empty()) {
        lSnapshotsCreated = true;
        std::string sSearchMask = "snaps*";
        bool lResult = tSnapshots.MapFiles(sSnapshots, sSearchMask);
        bool lOldSnapsRemoved = tSnapshots.CreateAndCleanPath();
        sVisItFile = tSnapshots.GetVisItName();
        std::ofstream fVisIt(sVisItFile);
        fVisIt.close(); // Force a rewrite: all further accesses will be in append mode
    }

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
    thrust::device_vector<float> rvX2(iNumPart);
    thrust::device_vector<float> rvDeltaU(iNumPart);
    thrust::device_vector<float> rvDeltaV(iNumPart);
    thrust::host_vector<float>   rvCellX(iNumPart);
    thrust::host_vector<float>   rvCellY(iNumPart);
    thrust::host_vector<float>   rvTempX(iNumPart);
    thrust::host_vector<float>   rvTempY(iNumPart);

    // Initialize the particles' time stamp to -1, to mean "not yet assigned" the parallel vay
    thrust::fill(ivPartTimeStamp.begin(), ivPartTimeStamp.end(), -1);

    // Main loop: iterate over meteo data
    std::string sOutFileName = tCfg.GetOutputFile();
    auto fOut = std::fstream(sOutFileName, std::ios::out | std::ios::binary);
    int n = tCfg.GetCellsPerEdge();
    auto imNumPartsInCell = new unsigned int[n * n];
    auto rmConc = new float[n * n];
    int iNumData = tCfg.GetNumMeteoData();
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    unsigned int iIteration = 0;
    int iFirstTimeStamp = 0;
    int iNextRandomBlock = 0;
    for (auto i = 0; i < iNumData; i++) {

        // Get current meteorology
        int iTimeStamp;
        float rU;
        float rV;
        float rStdDevU;
        float rStdDevV;
        float rCovUV;
        bool lOk = tCfg.GetMeteo(i, iTimeStamp, rU, rV, rStdDevU, rStdDevV, rCovUV);
        if (iIteration == 0) {
            iFirstTimeStamp = iTimeStamp;
        }

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
            index_sequence_begin + iNextRandomBlock,
            index_sequence_begin + iNextRandomBlock + iNumPart,
            rvN1.begin(),
            normal_deviate(0.0f, 1.0f)
        );
        iNextRandomBlock += iNumPart;
        thrust::transform(
            index_sequence_begin + iNextRandomBlock,
            index_sequence_begin + iNextRandomBlock + iNumPart,
            rvN2.begin(),
            normal_deviate(0.0f, 1.0f)
        );
        iNextRandomBlock += iNumPart;
        iIteration++;
        // -1- Transform the two independent samples in a 2D bivariate sample
        float rho;
        float lambda;
        if (rStdDevU > 0.f && rStdDevV > 0.f) {
            rho = rCovUV / (rStdDevU * rStdDevV);
            lambda = (rStdDevV / rStdDevU) * rho;
        }
        else {
            rho = 0.f;
            lambda = 0.f;
        }
        float nu = sqrtf((1.0f - rho * rho) * rStdDevV * rStdDevV);
        rvX1 = rvN1;
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(rStdDevU), rvX1.begin(), thrust::multiplies<float>());
        rvDeltaU = rvX1;
        rvX2 = rvN2;
        thrust::transform(rvX2.begin(), rvX2.end(), thrust::make_constant_iterator(nu), rvX2.begin(), thrust::multiplies<float>());
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(rU), rvX1.begin(), thrust::minus<float>());
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(lambda), rvX1.begin(), thrust::multiplies<float>());
        thrust::transform(rvX1.begin(), rvX1.end(), rvX2.begin(), rvX1.begin(), thrust::plus<float>());
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(rV), rvX1.begin(), thrust::plus<float>());
        rvDeltaV = rvX1;
        float rN1, rN2;
        for (auto jj = 0; jj < 10; ++jj) {
            rN1 = rvDeltaU[jj];
            rN2 = rvDeltaV[jj];
            std::cout << rN1 << " " << rN2 << std::endl;
        }

        // Move particles
        float rDeltaT = tCfg.GetTimeStep();
        thrust::transform(rvPartU.begin(), rvPartU.end(), rvDeltaU.begin(), rvPartU.begin(), thrust::plus<float>());
        rvX1 = rvPartU;
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(rDeltaT), rvX1.begin(), thrust::multiplies<float>());
        thrust::transform(rvPartX.begin(), rvPartX.end(), rvX1.begin(), rvPartX.begin(), thrust::plus<float>());
        thrust::transform(rvPartV.begin(), rvPartV.end(), rvDeltaV.begin(), rvPartV.begin(), thrust::plus<float>());
        rvX2 = rvPartV;
        thrust::transform(rvX2.begin(), rvX2.end(), thrust::make_constant_iterator(rDeltaT), rvX2.begin(), thrust::multiplies<float>());
        thrust::transform(rvPartY.begin(), rvPartY.end(), rvX2.begin(), rvPartY.begin(), thrust::plus<float>());

        // Count cell contents
        rvX1 = rvPartX;
        rvX2 = rvPartY;
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(tCfg.GetMinX()), rvX1.begin(), thrust::minus<float>());
        thrust::transform(rvX1.begin(), rvX1.end(), thrust::make_constant_iterator(tCfg.GetCellSize()), rvX1.begin(), thrust::divides<float>());
        thrust::transform(rvX2.begin(), rvX2.end(), thrust::make_constant_iterator(tCfg.GetMinY()), rvX2.begin(), thrust::minus<float>());
        thrust::transform(rvX2.begin(), rvX2.end(), thrust::make_constant_iterator(tCfg.GetCellSize()), rvX2.begin(), thrust::divides<float>());
        rvCellX = rvX1;
        rvCellY = rvX2;
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; iy < n; iy++) {
                imNumPartsInCell[n * iy + ix] = 0U;
            }
        }
        for (int j = 0; j < rvCellX.size(); j++) {
            int ix = (int)rvCellX[j];
            int iy = (int)rvCellY[j];
            if (0 <= ix && ix < n && 0 <= iy && iy < n) {
                ++imNumPartsInCell[n * iy + ix];
            }
        }
        int iTotParticles = 0;
        for (int j = 0; j < n * n; j++) {
            iTotParticles += imNumPartsInCell[j];
        }
        float rTotParticles = iTotParticles;
        for (int j = 0; j < n * n; j++) {
            rmConc[j] = (float)imNumPartsInCell[j] / rTotParticles;
        }
        fOut.write((char*)&rmConc[0], (size_t)(n * n)*sizeof(float));

        // Write snapshot, if needed
        if (!sSnapshots.empty()) {
            std::stringstream ssIteration;
            ssIteration << std::setw(6) << std::setfill('0') << iIteration;
            std::string sIteration;
            ssIteration >> sIteration;
            std::string sSnapshotName = tSnapshots.GetFilePath() + "\\snap_" + sIteration + ".dat";
            std::ofstream fVisIt(tSnapshots.GetVisItName(), std::ios_base::app);
            fVisIt << "!TIME" << (float)(iTimeStamp - iFirstTimeStamp) / 3600.0f << std::endl;
            fVisIt << sSnapshotName << std::endl;
            fVisIt.close();
            rvTempX = rvPartX;
            rvTempY = rvPartY;
            std::ofstream fSnap(sSnapshotName);
            fSnap << "x y age\n";
            fSnap << "#coordflag xya\n";
            for (auto i = 0; i < iNumPart; ++i) {
                if(ivPartTimeStamp[i] >= 0) {
                    if (tCfg.GetMinX() <= rvTempX[i] && rvTempX[i] <= -tCfg.GetMinX() && tCfg.GetMinY() <= rvTempY[i] && rvTempY[i] <= -tCfg.GetMinY()) {
                        fSnap << rvTempX[i] << " " << rvTempY[i] << " " << iTimeStamp - ivPartTimeStamp[i] << std::endl;
                    }
                }
            }
            fSnap.close();
        }

        // Inform users of the progress
        std::cout << iIteration << " of " << iNumData << ", " << rU << ", " << rV << ", " << rStdDevU << ", " << rStdDevV << ", " << rCovUV << std::endl;

    }

    // Release OS resources
    fOut.close();

    // Write descriptor file
    std::ofstream fDsc(tCfg.GetDescriptorFile());
    fDsc << "TIME: 1.0\n";
    fDsc << "DATA_FILE: " << tCfg.GetOutputFile() << std::endl;
    fDsc << "DATA_SIZE: " << tCfg.GetCellsPerEdge() << " " << tCfg.GetCellsPerEdge() << " " << iNumData << std::endl;
    fDsc << "DATA_FORMAT: FLOAT\n";
    fDsc << "VARIABLE: Concentration\n";
    fDsc << "DATA_ENDIAN: LITTLE\n";
    fDsc << "CENTERING: zonal\n";
    fDsc << "BRICK_ORIGIN: " << tCfg.GetMinX() << " " << tCfg.GetMinY() << " " << 0.f << std::endl;
    fDsc << "BRICK_SIZE: " << -tCfg.GetMinX()*2.f << " " << -tCfg.GetMinY() * 2.f << " " << (float)iNumData << std::endl;
    fDsc.close();

    // Deallocate manually thrust resources
    // -1- Release count matrices
    delete rmConc;
    delete imNumPartsInCell;
    // -1- Reclaim workspace
    ivPartTimeStamp.clear();
    rvPartX.clear();
    rvPartY.clear();
    rvPartU.clear();
    rvPartV.clear();
    rvN1.clear();
    rvN2.clear();
    rvX1.clear();
    rvX2.clear();
    rvDeltaU.clear();
    rvDeltaV.clear();
    rvCellX.clear();
    rvCellY.clear();
    rvTempX.clear();
    rvTempY.clear();
    // -1- Clear any other resources
    ivPartTimeStamp.shrink_to_fit();
    rvPartX.shrink_to_fit();
    rvPartY.shrink_to_fit();
    rvPartU.shrink_to_fit();
    rvPartV.shrink_to_fit();
    rvN1.shrink_to_fit();
    rvN2.shrink_to_fit();
    rvX1.shrink_to_fit();
    rvX2.shrink_to_fit();
    rvDeltaU.shrink_to_fit();
    rvDeltaV.shrink_to_fit();
    rvCellX.shrink_to_fit();
    rvCellY.shrink_to_fit();
    rvTempX.shrink_to_fit();
    rvTempY.shrink_to_fit();

    // Leave
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    std::cout << "*** End Job ***" << std::endl;

    return 0;
}
