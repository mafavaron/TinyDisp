#include <iostream>
#include <string>
#include "ini.h"

int main(int argc, const char* argv[]) {

    // Get configuration file and parse it
    if(argc != 2) {
        std::cerr << "tp - The TinyPart dispersion model" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Usage:" << std::endl << std::endl;
        std::cerr << "   ./tp <Config_File_Name>" << std::endl << std::endl;
        std::cerr << "Copyright 2019 by Mauri Favaron" << std::endl << std::endl;
        std::cerr << "This is open-source software, covered by the MIT license" << std::endl << std::endl;
        return 1;
    }
    std::string sFileName = argv[1];
    ini config(sFileName);

    // Test configuration
    std::string sSection = "General";
    std::string sKey     = "run";
    std::string sDefault = "Test_Name_00";
    std::string sName = config.getString(sSection, sKey, sDefault);
    std::cout << "Value = " << sName << std::endl;
    sKey     = "nome";
    sDefault = "Test_Name_01";
    sName = config.getString(sSection, sKey, sDefault);
    std::cout << "Value = " << sName << std::endl;

    // Leave, communicating successful completion
    return 0;

}
