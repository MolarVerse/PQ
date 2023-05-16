#include "guffDatReader.hpp"
#include "stringUtilities.hpp"
#include "exceptions.hpp"
#include "defaults.hpp"

#include <fstream>
#include <cmath>

using namespace std;
using namespace StringUtilities;

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
void readGuffDat(Engine &engine)
{
    GuffDatReader guffDat(engine);
    guffDat.setupGuffMaps();
    guffDat.read();
}

/**
 * @brief constructs the guff dat 4d vectors
 *
 */
void GuffDatReader::setupGuffMaps()
{
    size_t numberOfMoleculeTypes = _engine.getSimulationBox()._moleculeTypes.size();

    _engine.getSimulationBox()._guffCoefficients.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._rncCutOffs.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._coulombCoefficients.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._cEnergyCutOffs.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._cForceCutOffs.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._ncEnergyCutOffs.resize(numberOfMoleculeTypes);
    _engine.getSimulationBox()._ncForceCutOffs.resize(numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; i++)
    {
        _engine.getSimulationBox()._guffCoefficients[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._rncCutOffs[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._coulombCoefficients[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._cEnergyCutOffs[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._cForceCutOffs[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._ncEnergyCutOffs[i].resize(numberOfMoleculeTypes);
        _engine.getSimulationBox()._ncForceCutOffs[i].resize(numberOfMoleculeTypes);
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; i++)
    {
        size_t numberOfAtomTypes = _engine.getSimulationBox()._moleculeTypes[i].getNumberOfAtomTypes();
        for (size_t j = 0; j < numberOfMoleculeTypes; j++)
        {
            _engine.getSimulationBox()._guffCoefficients[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._rncCutOffs[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._coulombCoefficients[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._cEnergyCutOffs[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._cForceCutOffs[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._ncEnergyCutOffs[i][j].resize(numberOfAtomTypes);
            _engine.getSimulationBox()._ncForceCutOffs[i][j].resize(numberOfAtomTypes);
        }
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; i++)
    {
        size_t numberOfAtomTypes_i = _engine.getSimulationBox()._moleculeTypes[i].getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; j++)
        {
            size_t numberOfAtomTypes_j = _engine.getSimulationBox()._moleculeTypes[j].getNumberOfAtomTypes();

            for (size_t k = 0; k < numberOfAtomTypes_i; k++)
            {
                _engine.getSimulationBox()._guffCoefficients[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._rncCutOffs[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._coulombCoefficients[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._cEnergyCutOffs[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._cForceCutOffs[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._ncEnergyCutOffs[i][j][k].resize(numberOfAtomTypes_j);
                _engine.getSimulationBox()._ncForceCutOffs[i][j][k].resize(numberOfAtomTypes_j);
            }
        }
    }
}

/**
 * @brief reads the guff.dat file
 *
 * @throws GuffDatException if the file is invalid
 */
void GuffDatReader::read()
{
    ifstream fp(_filename);
    string line;

    while (getline(fp, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            _lineNumber++;
            continue;
        }

        auto lineCommands = getLineCommands(line, _lineNumber);

        if (lineCommands.size() - 1 != 28)
            throw GuffDatException("Invalid number of commands (" + to_string(lineCommands.size() - 1) + ") in line " + to_string(_lineNumber));

        parseLine(lineCommands);

        _lineNumber++;
    }
}

/**
 * @brief parses a line from the guff.dat file
 *
 * @param lineCommands
 *
 * @throws GuffDatException if the line is invalid
 */
void GuffDatReader::parseLine(vector<string> &lineCommands)
{
    Molecule molecule1;
    Molecule molecule2;

    int atomType1;
    int atomType2;

    try
    {
        molecule1 = _engine.getSimulationBox().findMoleculeType(stoi(lineCommands[0]));
        molecule2 = _engine.getSimulationBox().findMoleculeType(stoi(lineCommands[2]));
    }
    catch (const RstFileException &e)
    {
        throw GuffDatException("Invalid molecule type in line " + to_string(_lineNumber));
    }

    try
    {
        atomType1 = molecule1.getInternalAtomType(stoi(lineCommands[1]));
        atomType2 = molecule2.getInternalAtomType(stoi(lineCommands[3]));
    }
    catch (const std::exception &e)
    {
        throw GuffDatException("Invalid atom type in line " + to_string(_lineNumber));
    }

    double rncCutOff = stod(lineCommands[4]);

    if (rncCutOff < 0.0)
        rncCutOff = _engine.getSimulationBox().getRcCutOff();

    double coulombCoefficient = stod(lineCommands[5]);
    vector<double> guffCoefficients(22);

    for (size_t i = 0; i < 22; i++)
    {
        guffCoefficients[i] = stod(lineCommands[i + 6]);
    }

    int moltype1 = stoi(lineCommands[0]) - 1;
    int moltype2 = stoi(lineCommands[2]) - 1;

    _engine.getSimulationBox()._guffCoefficients[moltype1][moltype2][atomType1][atomType2] = guffCoefficients;
    _engine.getSimulationBox()._guffCoefficients[moltype2][moltype1][atomType2][atomType1] = guffCoefficients;

    _engine.getSimulationBox()._rncCutOffs[moltype1][moltype2][atomType1][atomType2] = rncCutOff;
    _engine.getSimulationBox()._rncCutOffs[moltype2][moltype1][atomType2][atomType1] = rncCutOff;

    _engine.getSimulationBox()._coulombCoefficients[moltype1][moltype2][atomType1][atomType2] = coulombCoefficient;
    _engine.getSimulationBox()._coulombCoefficients[moltype2][moltype1][atomType2][atomType1] = coulombCoefficient;

    double energy, force;
    double dummyCutoff = 1.0;

    _engine._jobType->calcCoulomb(coulombCoefficient, dummyCutoff, _engine.getSimulationBox().getRcCutOff(), energy, force, 0.0, 0.0);

    _engine.getSimulationBox()._cEnergyCutOffs[moltype1][moltype2][atomType1][atomType2] = energy;
    _engine.getSimulationBox()._cEnergyCutOffs[moltype2][moltype1][atomType2][atomType1] = energy;
    _engine.getSimulationBox()._cForceCutOffs[moltype1][moltype2][atomType1][atomType2] = force;
    _engine.getSimulationBox()._cForceCutOffs[moltype2][moltype1][atomType2][atomType1] = force;

    _engine._jobType->calcNonCoulomb(guffCoefficients, dummyCutoff, rncCutOff, energy, force, 0.0, 0.0);

    _engine.getSimulationBox()._ncEnergyCutOffs[moltype1][moltype2][atomType1][atomType2] = energy;
    _engine.getSimulationBox()._ncEnergyCutOffs[moltype2][moltype1][atomType2][atomType1] = energy;
    _engine.getSimulationBox()._ncForceCutOffs[moltype1][moltype2][atomType1][atomType2] = force;
    _engine.getSimulationBox()._ncForceCutOffs[moltype2][moltype1][atomType2][atomType1] = force;
}