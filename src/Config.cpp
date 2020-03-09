#include "Config.h"
#include "ini.h"
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

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
		std::count << "Cannot load configuration file " << sCfgFile << std::endl;
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
			std::vector<int>	
			std::string sBuffer;
			bool        lIsFirst = true;
			while (fMeteo >> sBuffer) {
				if (lIsFirst) {
					lIsFirst = false; // And, do nothing with the buffer - a header, in case
				}
				else {
					std::vector<std::string> svFields;
					split(sBuffer, svFields);
				}
			}

		}

	}
};

virtual Config::~Config() {
};
