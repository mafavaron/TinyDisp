//
// Created by Mauri Favaron on 2019-04-20.
//

#include <iostream>
#include <fstream>
#include <string>
#include "ini.h"

ini::ini(void) {
    this->sIniFileName.empty();
    this->svSection.clear();
    this->svKey.clear();
    this->svValue.clear();
    this->mValues.clear();
}

ini::ini(const std::string sFileName) {
    std::ifstream iniFile(sFileName);
    std::string sLine;
    while(std::getline(iniFile, sLine)) {
        std::cout << sLine;
    }
}

ini::~ini() {
    this->sIniFileName.empty();
    this->svSection.clear();
    this->svKey.clear();
    this->svValue.clear();
    this->mValues.clear();
}
