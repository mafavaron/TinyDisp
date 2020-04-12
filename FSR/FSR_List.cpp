#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

int main(int argc, char** argv)
{

    // Get parameters
    if (argc != 3) {
        std::cerr << "FSR_List - Procedure for enumerating the '.fsr' files to process in other steps." << std::endl;
        std::cerr << "\nUsage:\n\n";
        std::cerr << "  ./FSR_List <DataPath> <OutFileName>\n" << std::endl;
        return 1;
    }
    std::string sDataPath = argv[1];
    std::string sOutFile = argv[2];

    // Iterate over directory and build file list
    std::vector<std::string> svFiles;
    std::filesystem::path pDataPath(sDataPath);
    if (!std::filesystem::exists(pDataPath)) {
        std::cerr << "FSR_List:: error: Data path does not exist" << std::endl;
        return 2;
    }
    for (const auto& fEntry : std::filesystem::directory_iterator(pDataPath)) {
        if (fEntry.is_regular_file()) {
            if (fEntry.path().extension() == ".fsr") {

                // Get full file name and save it to vector
                std::string sFileName = fEntry.path().string();
                svFiles.push_back(sFileName);

            }
        }
    }

    // Write file names
    auto fOut = std::fstream(sOutFile, std::ios::out);
    for (int iFileIdx = 0; iFileIdx < svFiles.size(); ++iFileIdx) {

        // Retrieve the known file name and corresponding base time stamp
        fOut << svFiles[iFileIdx] << std::endl;

    }

    // Leave
    std::cout << "Total files found: " << svFiles.size() << std::endl;

}
