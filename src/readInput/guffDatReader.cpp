#include "guffDatReader.hpp"

#include "angleForceField.hpp"       // for potential
#include "buckinghamPair.hpp"        // for BuckinghamPair
#include "connectivityElement.hpp"   // for simulationBox
#include "constants.hpp"             // for _COULOMB_PREFACTOR_
#include "defaults.hpp"              // for _NUMBER_OF_GUFF_ENTRIES_
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for GuffDatException, InputFileException
#include "fileSettings.hpp"          // for FileSettings
#include "forceField.hpp"            // for ForceField
#include "guffNonCoulomb.hpp"        // for GuffNonCoulomb
#include "guffPair.hpp"              // for GuffPair
#include "lennardJonesPair.hpp"      // for LennardJonesPair
#include "mathUtilities.hpp"         // for sign, utilities
#include "molecule.hpp"              // for Molecule
#include "morsePair.hpp"             // for MorsePair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "potential.hpp"             // for Potential
#include "potentialSettings.hpp"     // for PotentialSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "stringUtilities.hpp"       // for fileExists, getLineCommands, removeComments, splitString

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <exception>    // for exception
#include <format>       // for format
#include <fstream>      // for basic_istream, ifstream, std
#include <functional>   // for identity
#include <memory>       // for make_shared

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
    guffDat.postProcessSetup();
}

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
GuffDatReader::GuffDatReader(Engine &engine) : _engine(engine) { _fileName = settings::FileSettings::getGuffDatFileName(); }

/**
 * @brief reads the guff.dat file
 *
 * @throws GuffDatException if the file is invalid
 */
void GuffDatReader::read()
{
    ifstream fp(_fileName);
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

    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());
    guffNonCoulomb.resizeGuff(numberOfMoleculeTypes);

    _guffCoulombCoefficients.resize(numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        guffNonCoulomb.resizeGuff(i, numberOfMoleculeTypes);
        _guffCoulombCoefficients[i].resize(numberOfMoleculeTypes);
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        const size_t numberOfAtomTypes = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
        {
            guffNonCoulomb.resizeGuff(i, j, numberOfAtomTypes);
            _guffCoulombCoefficients[i][j].resize(numberOfAtomTypes);
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
                guffNonCoulomb.resizeGuff(i, j, k, numberOfAtomTypes_j);
                _guffCoulombCoefficients[i][j][k].resize(numberOfAtomTypes_j);
            }
        }
    }
}

/**
 * @brief parses a line from the guff.dat file
 *
 * @param lineCommands
 *
 * æTODO: introduce keyword to ignore coulomb preFactors and use moldescriptor instead
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

    _guffCoulombCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2] = coulombCoefficient;
    _guffCoulombCoefficients[moltype2 - 1][moltype1 - 1][atomType2][atomType1] = coulombCoefficient;

    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    if (settings::PotentialSettings::getNonCoulombType() == "lj")
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
    else if (settings::PotentialSettings::getNonCoulombType() == "buck")
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
    else if (settings::PotentialSettings::getNonCoulombType() == "morse")
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

/**
 * @brief post process guff.dat reading
 *
 * @details sets partial charges of molecule types from guff.dat coulomb Coefficients
 *
 * @TODO: check if all combinations of partial charges are valid
 *
 */
void GuffDatReader::postProcessSetup()
{
    const size_t numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size();

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        const size_t numberOfAtoms = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtoms();

        auto *moleculeType = &(_engine.getSimulationBox().findMoleculeType(i + 1));

        for (size_t j = 0; j < numberOfAtoms; ++j)
        {
            auto atomType           = moleculeType->getAtomType(j);
            auto coulombCoefficient = _guffCoulombCoefficients[i][i][atomType][atomType];

            auto partialCharge =
                ::sqrt(coulombCoefficient / constants::_COULOMB_PREFACTOR_) * sign(moleculeType->getPartialCharge(j));

            moleculeType->setPartialCharge(j, partialCharge);
        }
    }

    auto setPartialCharges = [&engine = _engine](auto &molecule)
    {
        auto moleculeType = engine.getSimulationBox().findMoleculeType(molecule.getMoltype());

        for (size_t atomIndex = 0; atomIndex < molecule.getNumberOfAtoms(); ++atomIndex)
        {
            molecule.setPartialCharge(atomIndex, moleculeType.getPartialCharge(atomIndex));
            molecule.addInternalGlobalVDWType(0);
        }
    };

    ranges::for_each(_engine.getSimulationBox().getMolecules(), setPartialCharges);
}