#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv)
{

    // Get parameters
    if (argc != 4) {
        std::cerr << "SonicAvg - Procedure for converting raw 3D ultrasonic" << std::endl;
        std::cerr << "           data to a TinyDisp met input file" << std::endl;
        std::cerr << "\nUsage:\n\n";
        std::cerr << "  ./SonicAvg <DataPath> <AvgTime> <OutFileName>\n" << std::cerr;
    }
    std::string sDataPath = argv[1];
    std::string sBuffer   = argv[2];
    stringstream data(sBuffer);
    int iAvgTime = 0;
    data >> iAvgTime;
    std::string sOutFile = argv[3];
}
