#include "FileMgr.h"
#include <filesystem>
#include <regex>
#include <vector>
#include <string>

namespace fs = std::filesystem;

FileMgr::FileMgr(void) {
	this->sBasePath.clear();
	this->sSearchMask.clear();
	this->lHasData = false;
};

FileMgr::~FileMgr(void) {
	this->svFileName.clear();
};

bool FileMgr::MapFiles(const std::string sPath, const std::string sSearchMask) {

	// Create regular expression from the search pattern
	std::regex reMatch(sSearchMask, std::regex_constants::nosubs);
	std::smatch match;

	// Clean out file list, first of all
	this->svFileName.clear();

	// Assign path and search names: they always constitute the execution context
	this->sBasePath = sPath;
	this->sSearchMask = sSearchMask;

	// Before to really proceed, check something can be made, that is, the directory exists
	if (!fs::exists(sPath)) {
		return false;
	}

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

std::string FileMgr::GetInnermostDirectory(void) {
	std::string sInnermostDir;
	if (!this->sBasePath.empty()) {
		std::vector<std::string> svParts;
		std::filesystem::path pBasePath = this->sBasePath;
		for (auto& e : pBasePath) {
			std::string sTempPath = e.string();
			svParts.push_back(sTempPath);
		}
		if (svParts.size() > 1) {
			sInnermostDir = svParts[svParts.size()-2];
		}
		else {
			sInnermostDir = "";
		}
	}
	else {
		sInnermostDir = "";
	}
	return sInnermostDir;
};

std::string FileMgr::GetVisItName(void) {
	std::string sVisIt;
	if (!this->sBasePath.empty()) {
		std::vector<std::string> svParts;
		std::filesystem::path pBasePath = this->sBasePath;
		for (auto& e : pBasePath) {
			std::string sTempPath = e.string();
			svParts.push_back(sTempPath);
		}
		if (svParts.size() > 1) {
			std::filesystem::path pVisIt = svParts[0];
			for (int i = 1; i < svParts.size() - 1; ++i) {
				pVisIt /= svParts[i];
			}
			sVisIt = pVisIt.string() + ".visit";
		}
		else {
			sVisIt = "";
		}
	}
	else {
		sVisIt = "";
	}
	return sVisIt;
};

std::string FileMgr::GetFilePath(void) {
	return this->sBasePath;
};

