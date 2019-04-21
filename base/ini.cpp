//
// Created by Maurizio Favaron on 2019-04-20.
//

#include "ini.h"

ini::ini(void) {
    this->sIniFileName.empty();
    this->svSection.clear();
    this->svKey.clear();
    this->svValue.clear();
    this->mValues.clear();
}

ini::ini(const std::string sFileName) {

}

ini::~ini() {

}
