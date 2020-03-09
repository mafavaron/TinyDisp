#pragma once

#include <vector>
#include <string>

class Config
{
private:
	// General part
	bool		lIsValid;
	int			iTimeStep;
	float		rEdgeLength;
	int			iPartsPerStep;
	int			iStepsSurvival;
	std::string	sMeteoFile;
	// Meteorology
	std::vector<int>	ivTimeStamp;
	std::vector<float>	rvU;
	std::vector<float>	rvV;
	std::vector<float>	rvStdDevU;
	std::vector<float>	rvStdDevV;
	std::vector<float>	rvCovUV;
public:
	Config(const std::string sConfigFile);
	virtual ~Config();
};
