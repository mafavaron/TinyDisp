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
	FileMgr(void);
	FileMgr(const std::string sPath, const std::string sSearchMask);
	virtual ~FileMgr(void);
	bool FileMgr::CreateAndCleanPath(void);
};

