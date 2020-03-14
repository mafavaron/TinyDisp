#include "FileMgr.h"
#include <filesystem>

FileMgr::FileMgr(const std::string sPath, const std::string sSearchMask) {

	// Clean out file list, first of all
	this->svFileName.clear();

	// Scan the path indicated looking for files satisfying the search mask
	std::filesystem::path tDataPath(sPath);
	for (const auto& entry : std::filesystem::directory_iterator(sPath)) {

	}

	// Store path and search mask
	this->sBasePath   = sPath;
	this->sSearchMask = sSearchMask;

};

FileMgr::~FileMgr(void) {
};
