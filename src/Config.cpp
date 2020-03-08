#include "Config.h"
#include "ini.h"

Config::Config(const std::string sMeteoFile) {

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

	}
};

virtual Config::~Config() {
};
