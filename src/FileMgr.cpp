#include "FileMgr.h"
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

FileMgr::FileMgr(void) {
	this->sBasePath.clear();
	this->sSearchMask.clear();
	this->lHasData = false;
};

FileMgr::~FileMgr(void) {
};

bool FileMgr::MapFiles(const std::string sPath, const std::string sSearchMask) {

	// Create regular expression from the search pattern
	std::regex reMatch(sSearchMask, std::regex_constants::nosubs);
	std::smatch match;

	// Clean out file list, first of all
	this->svFileName.clear();

	// Scan the path indicated looking for files satisfying the search mask
	fs::path tDataPath(sPath);
	for (const auto& entry : fs::directory_iterator(sPath)) {
		const auto sFileName = entry.path().filename().string();
		if (entry.is_directory()) {
			// Ignore it
		}
		else {
			// Check conformance to file search mask
			if (std::regex_search(sFileName, match, reMatch)) {
				this->svFileName.push_back(sFileName);
			}
		}
	}

	// Store path and search mask
	this->sBasePath = sPath;
	this->sSearchMask = sSearchMask;

	// Notify data presence for any further use
	this->lHasData = (this->svFileName.size() > 0);

	// Leave
	return true;

};

bool FileMgr::CreateAndCleanPath(void) {
	bool lResult;
	if (this->lHasData) {
		for (auto i = 0; i < this->svFileName.size(); ++i) {
			std::filesystem::remove(svFileName[i]);
		}
		this->svFileName.clear();
		this->lHasData = false;
		lResult = true;
	}
	else {
		lResult = false;
	}
	if (!std::filesystem::exists(this->sBasePath)) {
		std::filesystem::create_directories(this->sBasePath);
	}
	return lResult;
};

