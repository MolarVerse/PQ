#ifndef _TRAJECTORYTOCOM_HPP_

#define _TRAJECTORYTOCOM_HPP_

#include "analysisRunner.hpp"
#include "configurationReader.hpp"
#include "trajOutput.hpp"

#include <memory>

class TrajToCom : public AnalysisRunner
{
private:
    size_t _numberOfAtomsPerMolecule;

    std::vector<std::string> _xyzFiles;
    std::vector<size_t> _atomIndices;

    Frame _frame;

    ConfigurationReader _configReader;

    std::unique_ptr<TrajOutput> _trajOutput = std::make_unique<TrajOutput>("default.xyz");

public:
    using AnalysisRunner::AnalysisRunner;

    void setupMolecules();
    void setup() override;
    void run() override;

    void setNumberOfAtomsPerMolecule(size_t numberOfAtomsPerMolecule) { _numberOfAtomsPerMolecule = numberOfAtomsPerMolecule; }

    void setXyzFiles(const std::vector<std::string> &filenames) { _xyzFiles = filenames; }

    void setXyzOutFile(const std::string &filename) { _trajOutput->setFilename(filename); }

    void setAtomIndices(const std::vector<size_t> &indices) { _atomIndices = indices; }
};

#endif // _TRAJECTORYTOCOM_HPP_