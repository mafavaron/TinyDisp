#pragma once

#include <string>
#include <vector>

class FileMgr
{
private:
	bool						lHasData;
	std::string					sBasePath;
	std::string					sSearchMask;
	std::vector<std::string>	svFileName;
public:
	FileMgr(const std::string sPath, const std::string sSearchMask);
	virtual ~FileMgr(void);
};

