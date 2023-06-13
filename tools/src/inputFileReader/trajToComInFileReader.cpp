#include "trajToComInFileReader.hpp"

#include <iostream>

using namespace std;

AnalysisRunner &TrajToComInFileReader::read()
{
    parseTomlFile();

    auto xyzFiles = parseXYZFiles();
    _runner.setXyzFiles(xyzFiles);

    auto atomIndices = parseAtomIndices();
    _runner.setAtomIndices(atomIndices);

    auto numberOfAtomsPerMolecule = parseNumberOfAtomsPerMolecule();
    _runner.setNumberOfAtomsPerMolecule(numberOfAtomsPerMolecule);

    auto outputXYZ = parseXYZOutputFile();
    _runner.setXyzOutFile(outputXYZ);

    return _runner;
}