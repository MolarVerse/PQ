#include "trajToComInFileReader.hpp"

#include <iostream>

class AnalysisRunner;   // forward declaration

using namespace std;

AnalysisRunner &TrajToComInFileReader::read()
{
    parseTomlFile();

    const auto xyzFiles = parseXYZFiles();
    _runner.setXyzFiles(xyzFiles);

    const auto atomIndices = parseAtomIndices();
    _runner.setAtomIndices(atomIndices);

    const auto numberOfAtomsPerMolecule = parseNumberOfAtomsPerMolecule();
    _runner.setNumberOfAtomsPerMolecule(numberOfAtomsPerMolecule);

    const auto outputXYZ = parseXYZOutputFile();
    _runner.setXyzOutFile(outputXYZ);

    return _runner;
}