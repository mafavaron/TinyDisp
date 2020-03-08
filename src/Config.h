#pragma once
class Config
{
private:
	bool		lIsValid;
	int			iTimeStep;
	float		rEdgeLength;
	int			iPartsPerStep;
	int			iStepsSurvival;
	std::string	sMeteoFile;
public:
	Config(const std::string sMeteoFile);
	virtual ~Config();
};
