#include "guffDatReader.hpp"

#include "defaults.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace StringUtilities;
using namespace simulationBox;

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
void readGuffDat(Engine &engine) {
    GuffDatReader guffDat(engine);
    guffDat.setupGuffMaps();
    guffDat.read();
}

/**
 * @brief constructs the guff dat 4d vectors
 *
 */
void GuffDatReader::setupGuffMaps() {
    const size_t numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size();

    _engine.getSimulationBox().resizeGuff(numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
        _engine.getSimulationBox().resizeGuff(i, numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i) {
        const size_t numberOfAtomTypes = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
            _engine.getSimulationBox().resizeGuff(i, j, numberOfAtomTypes);
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i) {
        const size_t numberOfAtomTypes_i = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j) {
            const size_t numberOfAtomTypes_j = _engine.getSimulationBox().getMoleculeType(j).getNumberOfAtomTypes();

            for (size_t k = 0; k < numberOfAtomTypes_i; ++k)
                _engine.getSimulationBox().resizeGuff(i, j, k, numberOfAtomTypes_j);
        }
    }
}

/**
 * @brief reads the guff.dat file
 *
 * @throws GuffDatException if the file is invalid
 */
void GuffDatReader::read() {
    ifstream fp(_filename);
    string   line;

    while (getline(fp, line)) {
        line = removeComments(line, "#");

        if (line.empty()) {
            ++_lineNumber;
            continue;
        }

        auto lineCommands = getLineCommands(line, _lineNumber);

        if (lineCommands.size() - 1 != 28) throw GuffDatException("Invalid number of commands (" + to_string(lineCommands.size() - 1) + ") in line " + to_string(_lineNumber));

        parseLine(lineCommands);

        ++_lineNumber;
    }
}

/**
 * @brief parses a line from the guff.dat file
 *
 * @param lineCommands
 *
 * @throws GuffDatException if the line is invalid
 */
void GuffDatReader::parseLine(vector<string> &lineCommands) {
    Molecule molecule1;
    Molecule molecule2;

    size_t atomType1 = 0;
    size_t atomType2 = 0;

    try {
        molecule1 = _engine.getSimulationBox().findMoleculeType(stoi(lineCommands[0]));
        molecule2 = _engine.getSimulationBox().findMoleculeType(stoi(lineCommands[2]));
    } catch (const RstFileException &) {
        throw GuffDatException("Invalid molecule type in line " + to_string(_lineNumber));
    }

    try {
        atomType1 = molecule1.getInternalAtomType(stoul(lineCommands[1]));
        atomType2 = molecule2.getInternalAtomType(stoul(lineCommands[3]));
    } catch (const std::exception &) {
        throw GuffDatException("Invalid atom type in line " + to_string(_lineNumber));
    }

    double rncCutOff = stod(lineCommands[4]);

    if (rncCutOff < 0.0) rncCutOff = _engine.getSimulationBox().getRcCutOff();

    const double   coulombCoefficient = stod(lineCommands[5]);
    vector<double> guffCoefficients(22);

    for (size_t i = 0; i < 22; ++i) {
        guffCoefficients[i] = stod(lineCommands[i + 6]);
    }

    const size_t moltype1 = stoul(lineCommands[0]);
    const size_t moltype2 = stoul(lineCommands[2]);

    _engine.getSimulationBox().setGuffCoefficients(moltype1, moltype2, atomType1, atomType2, guffCoefficients);
    _engine.getSimulationBox().setGuffCoefficients(moltype2, moltype1, atomType2, atomType1, guffCoefficients);

    _engine.getSimulationBox().setRncCutOff(moltype1, moltype2, atomType1, atomType2, rncCutOff);
    _engine.getSimulationBox().setRncCutOff(moltype2, moltype1, atomType2, atomType1, rncCutOff);

    _engine.getSimulationBox().setCoulombCoefficient(moltype1, moltype2, atomType1, atomType2, coulombCoefficient);
    _engine.getSimulationBox().setCoulombCoefficient(moltype2, moltype1, atomType2, atomType1, coulombCoefficient);

    double       energy      = 0.0;
    double       force       = 0.0;
    const double dummyCutoff = 1.0;

    _engine._potential->_coulombPotential->calcCoulomb(coulombCoefficient, dummyCutoff, _engine.getSimulationBox().getRcCutOff(), energy, force, 0.0, 0.0);

    _engine.getSimulationBox().setcEnergyCutOff(moltype1, moltype2, atomType1, atomType2, energy);
    _engine.getSimulationBox().setcEnergyCutOff(moltype2, moltype1, atomType2, atomType1, energy);
    _engine.getSimulationBox().setcForceCutOff(moltype1, moltype2, atomType1, atomType2, force);
    _engine.getSimulationBox().setcForceCutOff(moltype2, moltype1, atomType2, atomType1, force);

    energy = 0.0;
    force  = 0.0;

    _engine._potential->_nonCoulombPotential->calcNonCoulomb(guffCoefficients, dummyCutoff, rncCutOff, energy, force, 0.0, 0.0);

    _engine.getSimulationBox().setncEnergyCutOff(moltype1, moltype2, atomType1, atomType2, energy);
    _engine.getSimulationBox().setncEnergyCutOff(moltype2, moltype1, atomType2, atomType1, energy);
    _engine.getSimulationBox().setncForceCutOff(moltype1, moltype2, atomType1, atomType2, force);
    _engine.getSimulationBox().setncForceCutOff(moltype2, moltype1, atomType2, atomType1, force);
}