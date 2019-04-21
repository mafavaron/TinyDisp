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
    std::vector<std::string>            svKey;
    std::vector<std::string>            svValue;
    std::map<std::string,std::string>   mValues;
public:
    ini(void);
    ini(const std::string sFileName);
    virtual ~ini();
};


#endif //TINYPART_INI_H
