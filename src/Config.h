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
	int			iCellsPerEdge;
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
	bool GetMeteo(const int i, int& iTimeStamp, float& rU, float& rV, float& rStdDevU, float& rStdDevV, float& rCovUV);
	int GetNumMeteoData(void);
	int GetParticlePoolSize(void);
	int GetNumNewParticles(void);
	float GetTimeStep(void);
	int GetCellsPerEdge(void);
	float GetMinX();
	float GetMinY();
	float GetCellSize();
};
