#include "Config.h"
#include "ini.h"
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <filesystem>


template <class Container> void split(const std::string& str, Container& cont, char delim = ',') {
	std::stringstream ss(str);
	std::string       token;
	while (std::getline(ss, token, delim)) {
		cont.push_back(token);
	}
}


Config::Config(const std::string sConfigFile) {

	// Parse initialization file
	INIReader tCfg(sConfigFile);
	if (tCfg.ParseError() < 0) {
		std::cout << "Cannot load configuration file " << sConfigFile << std::endl;
		this->lIsValid = false;
	}
	else {

		// Get configuration parameters
		this->iTimeStep       = tCfg.GetInteger("General", "TimeStep", -1);
		this->iPartsPerStep   = tCfg.GetInteger("General", "PartsPerStep", -1);
		this->iStepsSurvival  = tCfg.GetInteger("General", "StepsSurvival", -1);
		this->rEdgeLength     = tCfg.GetReal("General", "EdgeLength", -1.0);
		this->iCellsPerEdge   = tCfg.GetInteger("General", "CellsPerEdge", 0);
		this->sMeteoFile      = tCfg.Get("General", "MeteoFile", "");
		this->sOutputFile     = tCfg.Get("General", "OutputFile", "");
		this->sDescriptorFile = tCfg.Get("General", "DescriptorFile", "");
		this->sSnapshotsPath  = tCfg.Get("General", "SnapshotsPath", "");

		// Try reading the meteorological file
		std::ifstream fMeteo;
		fMeteo.open(this->sMeteoFile.c_str());
		if (!fMeteo) {
			std::cout << "Unable to read meteorological file " << this->sMeteoFile << std::endl;
			this->lIsValid = false;
		}
		else {

			// Gather meteo data. By construction, meteo data are sorted
			// ascending with respect to time stamps
			std::string			sBuffer;
			bool				lIsFirst = true;
			std::vector<time_t>	ivTimeStamp;
			std::vector<float>	rvU;
			std::vector<float>	rvV;
			std::vector<float>	rvStdDevU;
			std::vector<float>	rvStdDevV;
			std::vector<float>	rvCovUV;
			while (std::getline(fMeteo, sBuffer)) {
				if (lIsFirst) {
					lIsFirst = false; // And, do nothing with the buffer - a header, in case
				}
				else {
					static const std::string dateTimeFormat{ "%Y-%m-%d %H:%M:%S" };
					std::vector<std::string> svFields;
					split(sBuffer, svFields);
					if (svFields.size() == 6) {
						float rU       = stof(svFields[1]);
						float rV       = stof(svFields[2]);
						float rStdDevU = stof(svFields[3]);
						float rStdDevV = stof(svFields[4]);
						float rCovUV   = stof(svFields[3]);
						if (rU > -9999.0f && rV > -9999.0f && rStdDevU > -9999.0f && rStdDevV > -9999.0f && rCovUV > -9999.0f) {
							std::istringstream ss{svFields[0]};
							struct tm tTimeStamp;
							ss >> std::get_time(&tTimeStamp, dateTimeFormat.c_str());
							ivTimeStamp.push_back(std::mktime(&tTimeStamp));
							rvU.push_back(rU);
							rvV.push_back(rV);
							rvStdDevU.push_back(rStdDevU);
							rvStdDevV.push_back(rStdDevV);
							rvCovUV.push_back(rCovUV);
						}
					}
				}
			}

			// Check some meteo data has been collected (if not, there is
			// nothing to be made
			if (ivTimeStamp.size() <= 0) return;

			// Check configuration values
			if (this->iTimeStep <= 0) return;
			if (this->rEdgeLength <= 0.0f) return;
			if (this->iPartsPerStep <= 0) return;
			if (this->iStepsSurvival <= 0) return;
			if (this->iCellsPerEdge <= 0) return;

			// Interpolate linearly in the time range of meteo data
			std::vector<float> rvInterpDeltaTime;
			int				   iIdx = 0;
			time_t             iTimeStamp = ivTimeStamp[iIdx];
			time_t             iLastTime = ivTimeStamp[ivTimeStamp.size() - 1];
			int                iNumElements = (iLastTime - iTimeStamp) / this->iTimeStep;
			this->ivTimeStamp.reserve(iNumElements);
			this->rvU.reserve(iNumElements);
			this->rvV.reserve(iNumElements);
			this->rvStdDevU.reserve(iNumElements);
			this->rvStdDevV.reserve(iNumElements);
			this->rvCovUV.reserve(iNumElements);
			while (iTimeStamp <= iLastTime) {

				// Exactly the same?
				if (iTimeStamp == ivTimeStamp[iIdx]) {

					// Yes! Just get values
					this->ivTimeStamp.push_back( iTimeStamp);
					this->rvU.push_back(         rvU[iIdx]       );
					this->rvV.push_back(         rvV[iIdx]       );
					this->rvStdDevU.push_back(   rvStdDevU[iIdx] );
					this->rvStdDevV.push_back(   rvStdDevV[iIdx] );
					this->rvCovUV.push_back(     rvCovUV[iIdx]   );

				}
				else {

					// No: Locate iIdx so that ivTimeStamp[iIdx] <= iTimeStamp < ivTimeStamp[iIdx+1]
					while (iTimeStamp < iLastTime && iTimeStamp >= ivTimeStamp[iIdx + 1]) {
						++iIdx;
					}

					// Check whether time is the same or not
					if (iTimeStamp == ivTimeStamp[iIdx]) {
						
						// Same! Just get values
						this->ivTimeStamp.push_back(iTimeStamp);
						this->rvU.push_back(rvU[iIdx]);
						this->rvV.push_back(rvV[iIdx]);
						this->rvStdDevU.push_back(rvStdDevU[iIdx]);
						this->rvStdDevV.push_back(rvStdDevV[iIdx]);
						this->rvCovUV.push_back(rvCovUV[iIdx]);

					}
					else {

						// Somewhere in-between: linear interpolation
						this->ivTimeStamp.push_back(iTimeStamp);
						float rFraction = (float)(iTimeStamp - ivTimeStamp[iIdx]) / (ivTimeStamp[iIdx + 1] - ivTimeStamp[iIdx]);
						this->rvU.push_back(rvU[iIdx] + rFraction * (rvU[iIdx + 1] - rvU[iIdx]));
						this->rvV.push_back(rvV[iIdx] + rFraction * (rvV[iIdx + 1] - rvV[iIdx]));
						this->rvStdDevU.push_back(rvStdDevU[iIdx] + rFraction * (rvStdDevU[iIdx + 1] - rvStdDevU[iIdx]));
						this->rvStdDevV.push_back(rvStdDevV[iIdx] + rFraction * (rvStdDevV[iIdx + 1] - rvStdDevV[iIdx]));
						this->rvCovUV.push_back(rvCovUV[iIdx] + rFraction * (rvCovUV[iIdx + 1] - rvCovUV[iIdx]));

					}

				}

				iTimeStamp += this->iTimeStep;

			}

			// Ensure the snapshots path, if non-empty, is terminated by an os-consistent
			// directory separator
			if (!this->sSnapshotsPath.empty()) {
				char cFinalChar = this->sSnapshotsPath[this->sSnapshotsPath.length() - 1];
				if (cFinalChar != std::filesystem::path::preferred_separator)
					this->sSnapshotsPath += std::filesystem::path::preferred_separator;
			}

		}

	}
};

Config::~Config() {
};

bool Config::GetMeteo(const int i, int& iTimeStamp, float& rU, float& rV, float& rStdDevU, float& rStdDevV, float& rCovUV) {

	// Check something can be made
	int iNumData = this->ivTimeStamp.size();
	bool lGo = (i >= 0 && i < iNumData);
	if (lGo) {

		// Retrieve the information desired
		iTimeStamp = this->ivTimeStamp[i];
		rU = this->rvU[i];
		rV = this->rvV[i];
		rStdDevU = this->rvStdDevU[i];
		rStdDevV = this->rvStdDevV[i];
		rCovUV = this->rvCovUV[i];

	}

	// Leave
	return lGo;

};

int Config::GetNumMeteoData(void) {
	return this->ivTimeStamp.size();
};

int Config::GetParticlePoolSize(void) {
	int iNumPart;
	if (this->lIsValid) {
		iNumPart = this->iPartsPerStep * this->iStepsSurvival;
	}
	else {
		iNumPart = 0;
	}
	return iNumPart;
};

int Config::GetNumNewParticles(void) {
	int iNumPart;
	if (this->lIsValid) {
		iNumPart = this->iPartsPerStep;
	}
	else {
		iNumPart = 0;
	}
	return iNumPart;
};

float Config::GetTimeStep(void) {
	float rTimeStep;
	if (this->lIsValid) {
		rTimeStep = (float)this->iTimeStep;
	}
	else {
		rTimeStep = 0.f;
	}
	return rTimeStep;
};

int Config::GetCellsPerEdge(void) {
	int iNumCells;
	if (this->lIsValid) {
		iNumCells = this->iCellsPerEdge;
	}
	else {
		iNumCells = 0;
	}
	return iNumCells;
};

float Config::GetMinX(void) {
	float rMin;
	if (this->lIsValid) {
		rMin = this->rEdgeLength / 2.f;
	}
	else {
		rMin = 0.f;
	}
	return rMin;
};

float Config::GetMinY(void) {
	float rMin;
	if (this->lIsValid) {
		rMin = this->rEdgeLength / 2.f;
	}
	else {
		rMin = 0.f;
	}
	return rMin;
};

float Config::GetCellSize(void) {
	float rSize;
	if (this->lIsValid) {
		rSize = this->rEdgeLength / this->iCellsPerEdge;
	}
	else {
		rSize = 0.f;
	}
	return rSize;
};

std::string Config::GetOutputFile(void) {
	std::string  sFile;
	if (this->lIsValid) {
		sFile = this->sOutputFile;
	}
	else {
		sFile = "";
	}
	return sFile;
};

std::string Config::GetDescriptorFile(void) {
	std::string  sFile;
	if (this->lIsValid) {
		sFile = this->sDescriptorFile;
	}
	else {
		sFile = "";
	}
	return sFile;
};
