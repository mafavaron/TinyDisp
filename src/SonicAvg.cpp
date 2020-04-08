#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <filesystem>

int main(int argc, char** argv)
{

    // Get parameters
    if (argc != 4) {
        std::cerr << "SonicAvg - Procedure for converting raw 3D ultrasonic" << std::endl;
        std::cerr << "           data to a TinyDisp met input file" << std::endl;
        std::cerr << "\nUsage:\n\n";
        std::cerr << "  ./SonicAvg <DataPath> <AvgTime> <OutFileName>\n" << std::endl;
        return 1;
    }
    std::string sDataPath = argv[1];
    std::string sBuffer   = argv[2];
    std::stringstream data(sBuffer);
    int iAvgTime = 0;
    data >> iAvgTime;
    std::string sOutFile = argv[3];

    // Compute number of blocks in an hour
    int iNumBlocks = 3600 / iAvgTime;
    if (iNumBlocks <= 0) {
        std::cerr << "SonicAvg:: error: No averaging blocks" << std::endl;
        return 2;
    }

    // Iterate over directory and build file list
    std::vector<std::string> svFiles;
    std::vector<std::time_t> ivTimeStamp;
    std::filesystem::path pDataPath(sDataPath);
    if (!std::filesystem::exists(pDataPath)) {
        std::cerr << "SonicAvg:: error: Data path does not exist" << std::endl;
        return 3;
    }
    for (const auto& fEntry : std::filesystem::directory_iterator(pDataPath)) {
        if (fEntry.is_regular_file()) {
            if (fEntry.path().extension() == ".fsr") {

                // Get full file name and save it to vector
                std::string sFileName = fEntry.path().string();
                svFiles.push_back(sFileName);

                // Get base name, convert it to an epoch time stamp, and store to vector
                std::wstring sBaseName = fEntry.path().filename().wstring();
                static const std::wstring timeStampFormat(L"%Y%m%d.%H");
                std::wstringstream sd( sBaseName );
                std::tm tTimeStamp;
                sd >> std::get_time(&tTimeStamp, timeStampFormat.c_str());
                std::time_t iTimeStamp = std::mktime( &tTimeStamp );
                ivTimeStamp.push_back( iTimeStamp );

            }
        }
    }

    // Main loop: iterate over files
    std::vector<float> rvTimeStamp;
    std::vector<int>   ivTimeIndex;
    std::vector<float> rvU;
    std::vector<float> rvV;
    std::vector<float> rvW;
    std::vector<int>   ivNumData;
    std::vector<float> rvSumU;
    std::vector<float> rvSumV;
    std::vector<float> rvSumW;
    std::vector<float> rvSumUU;
    std::vector<float> rvSumVV;
    std::vector<float> rvSumWW;
    std::vector<float> rvSumUV;
    std::vector<float> rvSumUW;
    std::vector<float> rvSumVW;
    ivNumData.reserve(iNumBlocks);
    rvSumU.reserve(iNumBlocks);
    rvSumV.reserve(iNumBlocks);
    rvSumW.reserve(iNumBlocks);
    rvSumUU.reserve(iNumBlocks);
    rvSumVV.reserve(iNumBlocks);
    rvSumWW.reserve(iNumBlocks);
    rvSumUV.reserve(iNumBlocks);
    rvSumUW.reserve(iNumBlocks);
    rvSumVW.reserve(iNumBlocks);
    auto fOut = std::fstream(sOutFile, std::ios::out);
    fOut << "Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW" << std::endl;
    for (int i = 0; i < iNumBlocks; ++i) {
        ivNumData[i] = 0;
        rvSumU[i]  = 0.f;
        rvSumV[i]  = 0.f;
        rvSumW[i]  = 0.f;
        rvSumUU[i] = 0.f;
        rvSumVV[i] = 0.f;
        rvSumWW[i] = 0.f;
        rvSumUV[i] = 0.f;
        rvSumUV[i] = 0.f;
        rvSumVW[i] = 0.f;
    }
    for (int iFileIdx = 0; iFileIdx < svFiles.size(); ++iFileIdx) {

        // Retrieve the known file name and corresponding base time stamp
        std::string sFileName = svFiles[iFileIdx];
        std::time_t iTimeStamp = ivTimeStamp[iFileIdx];

        // Retrieve this (binary!) file
        std::ifstream fInData(sFileName, std::ios::binary);
        int iNumData;
        short int iNumQuantities;
        float rTemporary;
        double rFill;
        if (fInData.is_open()) {
            rvTimeStamp.clear();
            ivTimeIndex.clear();
            rvU.clear();
            rvV.clear();
            rvW.clear();
            fInData.read((char*)&iNumData, sizeof(iNumData));
            fInData.read((char*)&iNumQuantities, sizeof(iNumQuantities));
            for (int i = 0; i < iNumQuantities; ++i) {
                fInData.read((char*)&rFill, sizeof(rFill));
            }
            for (int i = 0; i < iNumData; ++i) {
                fInData.read((char*)&rTemporary, sizeof(rTemporary));
                rvTimeStamp.push_back(rTemporary);
            }
            for (int i = 0; i < iNumData; ++i) {
                fInData.read((char*)&rTemporary, sizeof(rTemporary));
                rvU.push_back(rTemporary);
            }
            for (int i = 0; i < iNumData; ++i) {
                fInData.read((char*)&rTemporary, sizeof(rTemporary));
                rvV.push_back(rTemporary);
            }
            for (int i = 0; i < iNumData; ++i) {
                fInData.read((char*)&rTemporary, sizeof(rTemporary));
                rvW.push_back(rTemporary);
            }
        }
        fInData.close();

        // Generate the vector of time indices
        for (int i = 0; i < iNumData; ++i) {
            int iTimeIndex = (int)(rvTimeStamp[i] / iAvgTime);
            if (iTimeIndex < 0 || iTimeIndex >= iNumBlocks) iTimeIndex = -1;
            ivTimeIndex.push_back(iTimeIndex);
        }

        // Accumulate data
        for (int i = 0; i < iNumData; ++i) {
            int iIdx = ivTimeIndex[i];
            if (iIdx >= 0) {
                ++ivNumData[iIdx];
                rvSumU[iIdx]  += rvU[i];
                rvSumV[iIdx]  += rvV[i];
                rvSumW[iIdx]  += rvW[i];
                rvSumUU[iIdx] += rvU[i] * rvU[i];
                rvSumVV[iIdx] += rvV[i] * rvV[i];
                rvSumWW[iIdx] += rvW[i] * rvW[i];
                rvSumUV[iIdx] += rvU[i] * rvV[i];
                rvSumUW[iIdx] += rvU[i] * rvW[i];
                rvSumVW[iIdx] += rvV[i] * rvW[i];
            }
        }

        // Render statistics
        std::vector<std::time_t> ivBlockTimeStamp;
        std::vector<std::string> svBlockTimeStamp;
        std::vector<float> rvBlockU;
        std::vector<float> rvBlockV;
        std::vector<float> rvBlockW;
        std::vector<float> rvBlockUU;
        std::vector<float> rvBlockVV;
        std::vector<float> rvBlockWW;
        std::vector<float> rvBlockUV;
        std::vector<float> rvBlockUW;
        std::vector<float> rvBlockVW;
        for (int i = 0; i < iNumBlocks; ++i) {
            ivBlockTimeStamp.push_back(iTimeStamp + i * iAvgTime);
            struct tm * timeinfo;
            char cvBuffer[80];
            time_t iCurTime = ivBlockTimeStamp[i];
            timeinfo = std::gmtime(&iCurTime);
            std::strftime(cvBuffer, sizeof(cvBuffer), "%Y-%m-%d %H:%M:%S", timeinfo);
            svBlockTimeStamp.push_back(cvBuffer);
            if (ivNumData[i] > 0) {
                rvBlockU.push_back(rvSumU[i] / ivNumData[i]);
                rvBlockV.push_back(rvSumV[i] / ivNumData[i]);
                rvBlockW.push_back(rvSumW[i] / ivNumData[i]);
                rvBlockUU.push_back(rvSumUU[i] / ivNumData[i] - rvSumU[i] * rvSumU[i] / ivNumData[i]);
                rvBlockVV.push_back(rvSumVV[i] / ivNumData[i] - rvSumV[i] * rvSumV[i] / ivNumData[i]);
                rvBlockWW.push_back(rvSumWW[i] / ivNumData[i] - rvSumW[i] * rvSumW[i] / ivNumData[i]);
                rvBlockUU.push_back(rvSumUV[i] / ivNumData[i] - rvSumU[i] * rvSumV[i] / ivNumData[i]);
                rvBlockVV.push_back(rvSumUW[i] / ivNumData[i] - rvSumU[i] * rvSumW[i] / ivNumData[i]);
                rvBlockWW.push_back(rvSumVW[i] / ivNumData[i] - rvSumV[i] * rvSumW[i] / ivNumData[i]);
            }
            else {
                rvBlockU.push_back(-9999.9f);
                rvBlockV.push_back(-9999.9f);
                rvBlockW.push_back(-9999.9f);
                rvBlockUU.push_back(-9999.9f);
                rvBlockVV.push_back(-9999.9f);
                rvBlockWW.push_back(-9999.9f);
                rvBlockUU.push_back(-9999.9f);
                rvBlockVV.push_back(-9999.9f);
                rvBlockWW.push_back(-9999.9f);
            }
        }
        for (int i = 0; i < iNumBlocks; ++i) {
            if (rvBlockUU[i] >= 0.0f) {
                fOut << svBlockTimeStamp[i]
                    << ", " << rvBlockU[i]
                    << ", " << rvBlockV[i]
                    << ", " << rvBlockW[i]
                    << ", " << sqrtf(rvBlockUU[i])
                    << ", " << sqrtf(rvBlockVV[i])
                    << ", " << sqrtf(rvBlockWW[i])
                    << ", " << rvBlockUV[i]
                    << ", " << rvBlockUW[i]
                    << ", " << rvBlockVW[i]
                    << std::endl;
            }
        }

        std::cout << "Data: " << iNumData << "    File: " << sFileName << "\n";

    }

    // Leave
    std::cerr << "*** END JOB ***" << std::endl;

}
