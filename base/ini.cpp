//
// Created by Mauri Favaron on 2019-04-20.
//

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "ini.h"

ini::ini(void) {
    this->sIniFileName.empty();
    this->svSection.clear();
    this->svKey.clear();
    this->svValue.clear();
    this->mValues.clear();
}

ini::ini(const std::string sFileName) {

    // Main loop: Iterate through input file, line by line
    std::ifstream iniFile(sFileName);
    std::string sLine;
    std::string sCurrentSection = "";
    while(std::getline(iniFile, sLine)) {

        // Get line, and strip any comment from it
        std::string sCleanLine;
        if(size_t comment_pos = sLine.find('#') != std::string::npos)
            sCleanLine = std::string(sLine.cbegin(), find(sLine.cbegin(), sLine.cend(), '#'));
        else
            sCleanLine = sLine;

        // Remove leading and trailing blanks
        sCleanLine = trim(sCleanLine);

        // Check whether this is a section, or a 'key=value' assignment
        if(sCleanLine[0] == '@') {

        }

        std::cout << sCleanLine << std::endl;
    }
}

ini::~ini() {
    this->sIniFileName.empty();
    this->svSection.clear();
    this->svKey.clear();
    this->svValue.clear();
    this->mValues.clear();
}

std::string ini::trim(const std::string sString) {
    std::string sCleanLine;
    size_t firstpos = sCleanLine.find_first_not_of(" \t");
    if(firstpos != std::string::npos) sCleanLine = sCleanLine.substr(firstpos);
    size_t lastpos = sCleanLine.find_last_not_of(" \t");
    firstpos = sCleanLine.find_first_not_of(" \t");
    if(lastpos != std::string::npos)
    {
        sCleanLine = sCleanLine.substr(0, lastpos+1);
        sCleanLine = sCleanLine.substr(firstpos);
    }
    else
        sCleanLine.erase(std::remove(std::begin(sCleanLine), std::end(sCleanLine), ' '), std::end(sCleanLine));
    return sCleanLine;
}
