#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
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
    std::filesystem::path pDataPath(sDataPath);
    if (!std::filesystem::exists(pDataPath)) {
        std::cerr << "SonicAvg:: error: Data path does not exist" << std::endl;
        return 3;
    }
    for (const auto& fEntry : std::filesystem::directory_iterator(pDataPath)) {
        if (fEntry.is_regular_file()) {
            if (fEntry.path().extension() == ".fsr") {
                std::string sFileName = fEntry.path().string();
                svFiles.push_back(sFileName);
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
    for (int i; i < iNumBlocks; ++i) {
        ivNumData[i] = 0;
        rvU[i] = 0.f;
        rvV[i] = 0.f;
        rvW[i] = 0.f;
        rvUU[i] = 0.f;
        rvVV[i] = 0.f;
        rvWW[i] = 0.f;
        rvUV[i] = 0.f;
        rvUV[i] = 0.f;
        rvVW[i] = 0.f;
    }
    for (const auto& sFileName : svFiles) {

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

        std::cout << "Data: " << iNumData << "    File: " << sFileName << "\n";

    }

    // Leave
    std::cerr << "*** END JOB ***" << std::endl;

}
