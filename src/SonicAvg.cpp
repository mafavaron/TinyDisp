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

    // Iterate over directory and build file list
    std::vector<std::string> svFiles;
    std::filesystem::path pDataPath(sDataPath);
    if (!std::filesystem::exists(pDataPath)) {
        std::cerr << "SonicAvg:: error: Data path does not exist" << std::endl;
        return 2;
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
    std::vector<float> rvU;
    std::vector<float> rvV;
    std::vector<float> rvW;
    for (const auto& sFileName : svFiles) {

        // Retrieve this (binary!) file
        std::ifstream fInData(sFileName, std::ios::binary);
        int iNumData;
        int iNumQuantities;
        float rTemporary;
        if (fInData.is_open()) {
            fInData.read((char*)&iNumData, sizeof(iNumData));
            fInData.read((char*)&iNumQuantities, sizeof(iNumQuantities));
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

        std::cout << "Data: " << iNumData << "    File: " << sFileName << "\n";

    }

    // Leave
    std::cerr << "*** END JOB ***" << std::endl;

}
