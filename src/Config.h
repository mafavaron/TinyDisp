#pragma once

#include <vector>
#include <string>

class Config
{
private:
	// [General]
	bool		lIsValid;
	int			iTimeStep;
	// [Grid]
	float		rEdgeLength;
	// [Particles]
	int			iPartsPerStep;
	int			iStepsSurvival;
	// [Meteo]
	std::string	sMeteoFile;
	// [Output]
	std::string sOutputFile;
	// Actual meteorology (from file)
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
	float GetMinX();
	float GetMinY();
	std::string GetOutputFile(void);
	std::string GetSnapshotsPath(void);
};
