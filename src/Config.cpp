#include "Config.h"
#include "ini.h"
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>

template <class Container> void split(const std::string& str, Container& cont, char delim = ',') {
	std::stringstream ss(str);
	std::string       token;
	while (std::getline(ss, token, delim)) {
		cont.push_back(token);
	}
}


Config::Config(const std::string sConfigFile) {

	// Parse initialization file
	INIReader tCfg(sMeteoFile);
	if (tCfg.ParseError() < 0) {
		std::cout << "Cannot load configuration file " << sConfigFile << std::endl;
		this->lIsValid = false;
	}
	else {

		// Get configuration parameters
		this->iTimeStep      = tCfg.GetInteger("General", "TimeStep", -1);
		this->iPartsPerStep  = tCfg.GetInteger("General", "PartsPerStep", -1);
		this->iStepsSurvival = tCfg.GetInteger("General", "StepsSurvival", -1);
		this->rEdgeLength    = tCfg.GetReal("General", "EdgeLength", -1.0);
		this->sMeteoFile     = tCfg.Get("General", "MeteoFile", "");

		// Try reading the meteorological file
		std::ifstream fMeteo;
		fMeteo.open(this->sMeteoFile.c_str());
		if (!fMeteo) {
			std::cout << "Unable to read meteorological file " << this->sMeteoFile << std::endl;
			this->lIsValid = false;
		}
		else {

			// Gather meteo data
			std::string			sBuffer;
			bool				lIsFirst = true;
			std::vector<time_t>	ivTimeStamp;
			std::vector<float>	rvU;
			std::vector<float>	rvV;
			std::vector<float>	rvStdDevU;
			std::vector<float>	rvStdDevV;
			std::vector<float>	rvCovUV;
			while (fMeteo >> sBuffer) {
				if (lIsFirst) {
					lIsFirst = false; // And, do nothing with the buffer - a header, in case
				}
				else {
					static const std::wstring dateTimeFormat{ L"%Y-%m-%d %H:%M:%S" };
					std::vector<std::string> svFields;
					split(sBuffer, svFields);
					if (svFields.size() == 6) {
						for (int i = 0; i < svFields.size(); i++) {
							float rU       = stof(svFields[1]);
							float rV       = stof(svFields[2]);
							float rStdDevU = stof(svFields[3]);
							float rStdDevV = stof(svFields[4]);
							float rCovUV   = stof(svFields[3]);
							if (rU > -9999.0f && rV > -9999.0f && rStdDevU > -9999.0f && rStdDevV > -9999.0f && rCovUV > -9999.0f) {
								std::istringstream ss{svFields[0]};
								std::tm tTimeStamp;
								std::get_time(&tTimeStamp, dateTimeFormat.c_str());
								ivTimeStamp.push_back(std::mktime(&tTimeStamp));
							}
						}
					}
				}
			}

		}

	}
};

Config::~Config() {
};
