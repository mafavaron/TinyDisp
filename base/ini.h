//
// Created by Mauri Favaron on 2019-04-20.
//

#ifndef TINYPART_INI_H
#define TINYPART_INI_H

#include <string>
#include <vector>
#include <map>

class ini {
private:
    std::string                         sIniFileName;
    std::vector<std::string>            svSection;
    std::map<std::string,std::string>   mValues;
private:
    static std::string trim(const std::string sString);
public:
    ini(void);
    ini(const std::string sFileName);
    virtual ~ini();
    std::string getString(std::string sSection, const std::string sKey, const std::string sDefault);
};


#endif //TINYPART_INI_H
