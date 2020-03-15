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
	virtual ~FileMgr(void);
	bool FileMgr::MapFiles(const std::string sPath, const std::string sSearchMask);
	bool FileMgr::CreateAndCleanPath(void);
	std::string FileMgr::GetInnermostDirectory(void);
	std::string FileMgr::GetVisItName(void);
	std::string FileMgr::GetFilePath(void);
};

