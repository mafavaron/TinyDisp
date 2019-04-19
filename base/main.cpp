#include <iostream>
#include <string>

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

    // Leave, communicating successful completion
    return 0;

}
