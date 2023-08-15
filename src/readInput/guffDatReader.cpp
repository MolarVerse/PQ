#include "guffDatReader.hpp"

#include "defaults.hpp"
#include "exceptions.hpp"
#include "guffNonCoulomb.hpp"
#include "stringUtilities.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace utilities;
using namespace simulationBox;
using namespace readInput;
using namespace engine;
using namespace customException;
using namespace potential;

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
void readInput::readGuffDat(Engine &engine)
{
    if (engine.getForceFieldPtr()->isNonCoulombicActivated())
        return;

    GuffDatReader guffDat(engine);
    guffDat.setupGuffMaps();
    guffDat.read();
}

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
GuffDatReader::GuffDatReader(Engine &engine) : _engine(engine)
{
    _filename = _engine.getSettings().getGuffDatFilename();
    if (!fileExists(_filename))
    {
        throw InputFileException(format("Could not open guffdat file {}", _filename));
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
    string   line;

    while (getline(fp, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        auto lineCommands = getLineCommands(line, _lineNumber);

        if (lineCommands.size() - 1 != defaults::_NUMBER_OF_GUFF_ENTRIES_)
        {
            const auto message = format("Invalid number of commands ({}) in line {} - {} are allowed",
                                        lineCommands.size() - 1,
                                        _lineNumber,
                                        defaults::_NUMBER_OF_GUFF_ENTRIES_);
            throw GuffDatException(message);
        }

        parseLine(lineCommands);

        ++_lineNumber;
    }
}

/**
 * @brief constructs the guff dat 4d vectors
 *
 */
void GuffDatReader::setupGuffMaps()
{

    const size_t numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size();

    _engine.getPotential().getCoulombPotential().resizeGuff(numberOfMoleculeTypes);
    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());
    guffNonCoulomb.resizeGuff(numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        _engine.getPotential().getCoulombPotential().resizeGuff(i, numberOfMoleculeTypes);
        guffNonCoulomb.resizeGuff(i, numberOfMoleculeTypes);
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        const size_t numberOfAtomTypes = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
        {
            _engine.getPotential().getCoulombPotential().resizeGuff(i, j, numberOfAtomTypes);
            guffNonCoulomb.resizeGuff(i, j, numberOfAtomTypes);
        }
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        const size_t numberOfAtomTypes_i = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
        {
            const size_t numberOfAtomTypes_j = _engine.getSimulationBox().getMoleculeType(j).getNumberOfAtomTypes();

            for (size_t k = 0; k < numberOfAtomTypes_i; ++k)
            {
                _engine.getPotential().getCoulombPotential().resizeGuff(i, j, k, numberOfAtomTypes_j);
                guffNonCoulomb.resizeGuff(i, j, k, numberOfAtomTypes_j);
            }
        }
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

    size_t atomType1 = 0;
    size_t atomType2 = 0;

    try
    {
        molecule1 = _engine.getSimulationBox().findMoleculeType(stoul(lineCommands[0]));
        molecule2 = _engine.getSimulationBox().findMoleculeType(stoul(lineCommands[2]));
    }
    catch (const RstFileException &)
    {
        throw GuffDatException(format("Invalid molecule type in line {}", _lineNumber));
    }

    try
    {
        atomType1 = molecule1.getInternalAtomType(stoul(lineCommands[1]));
        atomType2 = molecule2.getInternalAtomType(stoul(lineCommands[3]));
    }
    catch (const std::exception &)
    {
        throw GuffDatException(format("Invalid atom type in line {}", _lineNumber));
    }

    double rncCutOff = stod(lineCommands[4]);

    if (rncCutOff < 0.0)
        rncCutOff = _engine.getSimulationBox().getCoulombRadiusCutOff();

    const double   coulombCoefficient = stod(lineCommands[5]);
    vector<double> guffCoefficients(22);

    for (size_t i = 0; i < 22; ++i)
        guffCoefficients[i] = stod(lineCommands[i + 6]);

    const size_t moltype1 = stoul(lineCommands[0]);
    const size_t moltype2 = stoul(lineCommands[2]);

    _engine.getPotential().getCoulombPotential().setGuffCoulombCoefficient(
        moltype1, moltype2, atomType1, atomType2, coulombCoefficient);
    _engine.getPotential().getCoulombPotential().setGuffCoulombCoefficient(
        moltype2, moltype1, atomType2, atomType1, coulombCoefficient);

    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    if (_engine.getSettings().getNonCoulombType() == "lj")
    {
        auto lennardJonesPair                  = LennardJonesPair(rncCutOff, guffCoefficients[0], guffCoefficients[2]);
        const auto [energyCutOff, forceCutOff] = lennardJonesPair.calculateEnergyAndForce(rncCutOff);

        guffNonCoulomb.setGuffNonCoulombicPair(
            {moltype1, moltype2, atomType1, atomType2},
            make_shared<LennardJonesPair>(rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[2]));
        guffNonCoulomb.setGuffNonCoulombicPair(
            {moltype2, moltype1, atomType2, atomType1},
            make_shared<LennardJonesPair>(rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[2]));
    }
    else if (_engine.getSettings().getNonCoulombType() == "buck")
    {
        auto buckinghamPair = BuckinghamPair(rncCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]);
        const auto [energyCutOff, forceCutOff] = buckinghamPair.calculateEnergyAndForce(rncCutOff);

        guffNonCoulomb.setGuffNonCoulombicPair(
            {moltype1, moltype2, atomType1, atomType2},
            make_shared<BuckinghamPair>(
                rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]));
        guffNonCoulomb.setGuffNonCoulombicPair(
            {moltype2, moltype1, atomType2, atomType1},
            make_shared<BuckinghamPair>(
                rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]));
    }
    else if (_engine.getSettings().getNonCoulombType() == "morse")
    {
        auto morsePair = MorsePair(rncCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]);
        const auto [energyCutOff, forceCutOff] = morsePair.calculateEnergyAndForce(rncCutOff);

        guffNonCoulomb.setGuffNonCoulombicPair(
            {moltype1, moltype2, atomType1, atomType2},
            make_shared<MorsePair>(
                rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]));
        guffNonCoulomb.setGuffNonCoulombicPair(
            {
                moltype2,
                moltype1,
                atomType2,
                atomType1,
            },
            make_shared<MorsePair>(
                rncCutOff, energyCutOff, forceCutOff, guffCoefficients[0], guffCoefficients[1], guffCoefficients[2]));
    }
    else
    {
        auto guffPair                          = GuffPair(rncCutOff, guffCoefficients);
        const auto [energyCutOff, forceCutOff] = guffPair.calculateEnergyAndForce(rncCutOff);

        guffNonCoulomb.setGuffNonCoulombicPair({moltype1, moltype2, atomType1, atomType2},
                                               make_shared<GuffPair>(rncCutOff, energyCutOff, forceCutOff, guffCoefficients));
        guffNonCoulomb.setGuffNonCoulombicPair({moltype2, moltype1, atomType2, atomType1},
                                               make_shared<GuffPair>(rncCutOff, energyCutOff, forceCutOff, guffCoefficients));
    }
}