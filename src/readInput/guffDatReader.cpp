#include "guffDatReader.hpp"

#include "buckinghamPair.hpp"        // for BuckinghamPair
#include "constants.hpp"             // for _COULOMB_PREFACTOR_
#include "defaults.hpp"              // for _NUMBER_OF_GUFF_ENTRIES_
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for GuffDatException, InputFileException
#include "fileSettings.hpp"          // for FileSettings
#include "forceFieldClass.hpp"       // for ForceField
#include "guffNonCoulomb.hpp"        // for GuffNonCoulomb
#include "guffPair.hpp"              // for GuffPair
#include "lennardJonesPair.hpp"      // for LennardJonesPair
#include "mathUtilities.hpp"         // for sign, utilities
#include "molecule.hpp"              // for Molecule
#include "morsePair.hpp"             // for MorsePair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "potential.hpp"             // for Potential
#include "potentialSettings.hpp"     // for PotentialSettings
#include "settings.hpp"              // for settings
#include "simulationBox.hpp"         // for SimulationBox
#include "stringUtilities.hpp"       // for fileExists, getLineCommands, removeComments, splitString

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <exception>    // for exception
#include <format>       // for format
#include <fstream>      // for basic_istream, std::ifstream, std
#include <functional>   // for idestd::ntity
#include <memory>       // for make_shared
#include <ranges>       // for views::drop, for_each, ranges

using namespace readInput::guffdat;

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @details Following steps are performed:
 * 1. setupGuffMaps()
 * 2. read()
 * 3. postProcessSetup()
 *
 * @param engine
 */
void readInput::guffdat::readGuffDat(engine::Engine &engine)
{
    GuffDatReader guffDat(engine);

    if (!guffDat.isNeeded())
        return;

    guffDat.setupGuffMaps();
    guffDat.read();
    guffDat.postProcessSetup();
}

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
GuffDatReader::GuffDatReader(engine::Engine &engine) : _engine(engine)
{
    _fileName = settings::FileSettings::getGuffDatFileName();
}

/**
 * @brief checks wether reading the guff.dat is necessary or not
 *
 * @details not necessary if:
 * - mm is not active
 * - force field non coulombics is active
 *
 * @return true
 * @return false
 */
bool GuffDatReader::isNeeded()
{
    if (!settings::Settings::isMMActivated())
        return false;
    else if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
        return false;
    else
        return true;
}

/**
 * @brief reads the guff.dat file
 *
 * @details the guff.dat file is read line by line. Each line is parsed and the guffNonCoulombicPair is constructed. For further
 * details about the entries of the line see  the documentation of the guff.dat file.
 *
 * @throws customException::GuffDatException command line has not 28 entries
 */
void GuffDatReader::read()
{
    std::ifstream fp(_fileName);
    std::string   line;

    while (getline(fp, line))
    {
        line = utilities::removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        auto lineCommands = utilities::getLineCommands(line, _lineNumber);

        if (lineCommands.size() != defaults::_NUMBER_OF_GUFF_ENTRIES_)
        {
            const auto message = std::format("Invalid number of commands ({}) in line {} - {} are allowed",
                                             lineCommands.size(),
                                             _lineNumber,
                                             defaults::_NUMBER_OF_GUFF_ENTRIES_);
            throw customException::GuffDatException(message);
        }

        parseLine(lineCommands);

        ++_lineNumber;
    }
}

/**
 * @brief constructs the guff dat 4d vectors
 *
 * @details resizes the 4d vectors of guffNonCoulomb and _guffCoulombCoefficients in order to access elements with molTypes and
 * internal atomTypes
 *
 */
void GuffDatReader::setupGuffMaps()
{

    const size_t numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size();

    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    guffNonCoulomb.resizeGuff(numberOfMoleculeTypes);
    _guffCoulombCoefficients.resize(numberOfMoleculeTypes);
    _isGuffPairSet.resize(numberOfMoleculeTypes);

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
    {
        guffNonCoulomb.resizeGuff(i, numberOfMoleculeTypes);
        _guffCoulombCoefficients[i].resize(numberOfMoleculeTypes);
        _isGuffPairSet[i].resize(numberOfMoleculeTypes);
    }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
        {
            const auto numberOfAtomTypes = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();

            guffNonCoulomb.resizeGuff(i, j, numberOfAtomTypes);
            _guffCoulombCoefficients[i][j].resize(numberOfAtomTypes);
            _isGuffPairSet[i][j].resize(numberOfAtomTypes);
        }

    for (size_t i = 0; i < numberOfMoleculeTypes; ++i)
        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
            for (size_t k = 0, numberOfAtomTypes1 = _engine.getSimulationBox().getMoleculeType(i).getNumberOfAtomTypes();
                 k < numberOfAtomTypes1;
                 ++k)
            {
                const size_t numberOfAtomTypes2 = _engine.getSimulationBox().getMoleculeType(j).getNumberOfAtomTypes();

                guffNonCoulomb.resizeGuff(i, j, k, numberOfAtomTypes2);
                _guffCoulombCoefficients[i][j][k].resize(numberOfAtomTypes2);
                _isGuffPairSet[i][j][k].resize(numberOfAtomTypes2);

                for (size_t l = 0; l < numberOfAtomTypes2; ++l)
                    _isGuffPairSet[i][j][k][l] = false;
            }
}

/**
 * @brief parses a line from the guff.dat file
 *
 * @details the line is parsed and the guffNonCoulombicPair is constructed. For further details about the entries of the line see
 * the documentation of the guff.dat file
 *
 * @param lineCommands
 *
 * æTODO: introduce keyword to ignore coulomb preFactors and use moldescriptor instead
 *
 * @throws customException::GuffDatException if molecule or atom type is invalid
 */
void GuffDatReader::parseLine(const std::vector<std::string> &lineCommands)
{
    simulationBox::MoleculeType molecule1;
    simulationBox::MoleculeType molecule2;

    try
    {
        molecule1 = _engine.getSimulationBox().findMoleculeType(stoul(lineCommands[0]));
        molecule2 = _engine.getSimulationBox().findMoleculeType(stoul(lineCommands[2]));
    }
    catch (const std::exception &)
    {
        throw customException::GuffDatException(std::format("Invalid molecule type in line {}", _lineNumber));
    }

    size_t atomType1 = 0;
    size_t atomType2 = 0;

    try
    {
        atomType1 = molecule1.getInternalAtomType(stoul(lineCommands[1]));
        atomType2 = molecule2.getInternalAtomType(stoul(lineCommands[3]));
    }
    catch (const std::exception &)
    {
        throw customException::GuffDatException(std::format("Invalid atom type in line {}", _lineNumber));
    }

    double rncCutOff = stod(lineCommands[4]);

    if (rncCutOff < 0.0)
        rncCutOff = _engine.getSimulationBox().getCoulombRadiusCutOff();

    const double        coulombCoefficient = stod(lineCommands[5]);
    std::vector<double> guffCoefficients;

    std::ranges::for_each(lineCommands | std::views::drop(6),
                          [&guffCoefficients](const auto &entry) { guffCoefficients.push_back(stod(entry)); });

    const size_t moltype1 = stoul(lineCommands[0]);
    const size_t moltype2 = stoul(lineCommands[2]);

    _guffCoulombCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2] = coulombCoefficient;
    _guffCoulombCoefficients[moltype2 - 1][moltype1 - 1][atomType2][atomType1] = coulombCoefficient;
    _isGuffPairSet[moltype1 - 1][moltype2 - 1][atomType1][atomType2]           = true;
    _isGuffPairSet[moltype2 - 1][moltype1 - 1][atomType2][atomType1]           = true;

    addNonCoulombPair(moltype1, moltype2, atomType1, atomType2, guffCoefficients, rncCutOff);
}

/**
 * @brief checks which nonCoulombic type is given and adds the corresponding nonCoulombic pair
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 *
 * @throws customException::UserInputException if nonCoulombic type is invalid
 */
void GuffDatReader::addNonCoulombPair(const size_t               molType1,
                                      const size_t               molType2,
                                      const size_t               atomType1,
                                      const size_t               atomType2,
                                      const std::vector<double> &coefficients,
                                      const double               rncCutOff)
{
    switch (settings::PotentialSettings::getNonCoulombType())
    {
    case settings::NonCoulombType::LJ:
    {
        addLennardJonesPair(molType1, molType2, atomType1, atomType2, coefficients, rncCutOff);
        break;
    }
    case settings::NonCoulombType::BUCKINGHAM:
    {
        addBuckinghamPair(molType1, molType2, atomType1, atomType2, coefficients, rncCutOff);
        break;
    }
    case settings::NonCoulombType::MORSE:
    {
        addMorsePair(molType1, molType2, atomType1, atomType2, coefficients, rncCutOff);
        break;
    }
    case settings::NonCoulombType::GUFF:
    {
        addGuffPair(molType1, molType2, atomType1, atomType2, coefficients, rncCutOff);
        break;
    }
    default:
    {
        throw customException::UserInputException(std::format(
            "Invalid nonCoulombic type {} given", settings::string(settings::PotentialSettings::getNonCoulombType())));
    }
    }
}

/**
 * @brief adds a lennard jones pair to the guffNonCoulombic potential
 *
 * @details first guff coefficient is c6, third is c12
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 */
void GuffDatReader::addLennardJonesPair(const size_t               molType1,
                                        const size_t               molType2,
                                        const size_t               atomType1,
                                        const size_t               atomType2,
                                        const std::vector<double> &coefficients,
                                        const double               rncCutOff)
{
    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    auto lennardJonesPair                  = potential::LennardJonesPair(rncCutOff, coefficients[0], coefficients[2]);
    const auto [energyCutOff, forceCutOff] = lennardJonesPair.calculateEnergyAndForce(rncCutOff);

    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<potential::LennardJonesPair>(rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[2]));
    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<potential::LennardJonesPair>(rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[2]));
}

/**
 * @brief adds a buckingham pair to the guffNonCoulombic potential
 *
 * @details first guff coefficient is a, second is dRho, third is c6
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 */
void GuffDatReader::addBuckinghamPair(const size_t               molType1,
                                      const size_t               molType2,
                                      const size_t               atomType1,
                                      const size_t               atomType2,
                                      const std::vector<double> &coefficients,
                                      const double               rncCutOff)
{
    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    auto buckinghamPair = potential::BuckinghamPair(rncCutOff, coefficients[0], coefficients[1], coefficients[2]);
    const auto [energyCutOff, forceCutOff] = buckinghamPair.calculateEnergyAndForce(rncCutOff);

    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<potential::BuckinghamPair>(
            rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[1], coefficients[2]));
    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<potential::BuckinghamPair>(
            rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[1], coefficients[2]));
}

/**
 * @brief adds a morse pair to the guffNonCoulombic potential
 *
 * @details first guff coefficient is the dissociationEnergy , second is the wellWidth, third is the equilibriumDistance
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 */
void GuffDatReader::addMorsePair(const size_t               molType1,
                                 const size_t               molType2,
                                 const size_t               atomType1,
                                 const size_t               atomType2,
                                 const std::vector<double> &coefficients,
                                 const double               rncCutOff)
{
    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    auto morsePair                         = potential::MorsePair(rncCutOff, coefficients[0], coefficients[1], coefficients[2]);
    const auto [energyCutOff, forceCutOff] = morsePair.calculateEnergyAndForce(rncCutOff);

    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<potential::MorsePair>(
            rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[1], coefficients[2]));
    guffNonCoulomb.setGuffNonCoulombicPair(
        {
            molType2,
            molType1,
            atomType2,
            atomType1,
        },
        std::make_shared<potential::MorsePair>(
            rncCutOff, energyCutOff, forceCutOff, coefficients[0], coefficients[1], coefficients[2]));
}

/**
 * @brief adds a guff pair to the guffNonCoulombic potential
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 */
void GuffDatReader::addGuffPair(const size_t               molType1,
                                const size_t               molType2,
                                const size_t               atomType1,
                                const size_t               atomType2,
                                const std::vector<double> &coefficients,
                                const double               rncCutOff)
{
    auto &guffNonCoulomb = dynamic_cast<potential::GuffNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());
    auto  guffPair       = potential::GuffPair(rncCutOff, coefficients);
    const auto [energyCutOff, forceCutOff] = guffPair.calculateEnergyAndForce(rncCutOff);

    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<potential::GuffPair>(rncCutOff, energyCutOff, forceCutOff, coefficients));
    guffNonCoulomb.setGuffNonCoulombicPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<potential::GuffPair>(rncCutOff, energyCutOff, forceCutOff, coefficients));
}

/**
 * @brief post process guff.dat reading
 *
 * @details following steps are performed:
 * 1) check if all necessary guff pairs are set
 * 2) calculate the partial charges of the molecule types from the guff.dat coulomb coefficients
 * 3) resize the internal global vdw types // just for compatibility with force field
 * 4) check if the partial charges are in accordance with all guff.dat entries
 *
 */
void GuffDatReader::postProcessSetup()
{
    checkNecessaryGuffPairs();

    calculatePartialCharges();
    _engine.getSimulationBox().setPartialChargesOfMoleculesFromMoleculeTypes();

    _engine.getSimulationBox().resizeInternalGlobalVDWTypes();

    checkPartialCharges();
}

/**
 * @brief calculates the partial charges of the molecule types from the guff.dat coulomb coefficients
 *
 * @details the partial charges are calculated via sqrt(coulombCoefficient / constants::_COULOMB_PREFACTOR_) of the self
 * interactions and then set to the molecule type
 *
 * @note the correct sign of the partial charge has already to be set in the moldescriptor file - otherwise it is impossible to
 * determine the sign of the partial charge.
 *
 */
void GuffDatReader::calculatePartialCharges()
{
    for (size_t i = 0, numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size(); i < numberOfMoleculeTypes;
         ++i)
    {
        auto *moleculeType = &(_engine.getSimulationBox().findMoleculeType(i + 1));

        for (size_t j = 0, numberOfAtoms = moleculeType->getNumberOfAtoms(); j < numberOfAtoms; ++j)
        {
            auto atomType           = moleculeType->getAtomType(j);
            auto coulombCoefficient = _guffCoulombCoefficients[i][i][atomType][atomType];

            auto partialCharge =
                ::sqrt(coulombCoefficient / constants::_COULOMB_PREFACTOR_) * utilities::sign(moleculeType->getPartialCharge(j));

            moleculeType->setPartialCharge(j, partialCharge);
        }
    }
}

/**
 * @brief checks if the partial charges are in accordance with all guff.dat entries.
 *
 * @details The checks includes only moleculeTypes which are also used in the simulationBox, meaning that only the used
 * coulombCoefficient are checked. All coulombCoefficient combinations have to be of the form q1 * q2 * 1 / (4 * pi * epsilon_0)
 *
 * @throws customException::GuffDatException if the partial charges are invalid
 *
 */
void GuffDatReader::checkPartialCharges()
{
    for (size_t i = 0, numberOfMoleculeTypes = _engine.getSimulationBox().getMoleculeTypes().size(); i < numberOfMoleculeTypes;
         ++i)
    {
        const auto moleculeType1Optional = _engine.getSimulationBox().findMolecule(i + 1);

        simulationBox::Molecule moleculeType1;
        if (moleculeType1Optional == std::nullopt)
            continue;
        else
            moleculeType1 = moleculeType1Optional.value();

        for (size_t j = 0; j < numberOfMoleculeTypes; ++j)
        {
            const auto moleculeType2Optional = _engine.getSimulationBox().findMolecule(j + 1);

            simulationBox::Molecule moleculeType2;
            if (moleculeType2Optional == std::nullopt)
                continue;
            else
                moleculeType2 = moleculeType2Optional.value();

            for (size_t k = 0, numberOfAtomTypes1 = moleculeType1.getNumberOfAtomTypes(); k < numberOfAtomTypes1; ++k)
                for (size_t l = 0, numberOfAtomTypes2 = moleculeType2.getNumberOfAtomTypes(); l < numberOfAtomTypes2; ++l)
                {
                    auto partialCharge1 = moleculeType1.getPartialCharge(k);
                    auto partialCharge2 = moleculeType2.getPartialCharge(l);

                    if (!utilities::compare(partialCharge1 * partialCharge2 * constants::_COULOMB_PREFACTOR_,
                                            _guffCoulombCoefficients[i][j][k][l],
                                            1e-6))
                        throw customException::GuffDatException(
                            std::format("Invalid coulomb coefficient guff file for molecule "
                                        "types {} and {} and the {}. and the {}. atom type. The coulomb coefficient should "
                                        "be {} but is {}",
                                        i + 1,
                                        j + 1,
                                        k + 1,
                                        l + 1,
                                        partialCharge1 * partialCharge2 * constants::_COULOMB_PREFACTOR_,
                                        _guffCoulombCoefficients[i][j][k][l]));
                }
        }
    }
}

/**
 * @brief check if all necessary guff pairs are set
 *
 * @details the necessary guff pairs are determined by the molecule types which are used in the simulation box in _molecules
 * (defined by the restart file and not in the moldescriptor file) - the moldescriptor can have more molecule types than the
 * actual simulation box (e.g. if the moldescriptor is used for multiple simulations).
 *
 * @throws customException::GuffDatException if a necessary guff pair is not set
 */
void GuffDatReader::checkNecessaryGuffPairs()
{
    const auto necessaryMoleculeTypes = _engine.getSimulationBox().findNecessaryMoleculeTypes();

    for (const auto &moleculeType1 : necessaryMoleculeTypes)
        for (const auto &moleculeType2 : necessaryMoleculeTypes)
            for (size_t atomIndex1 = 0, numberOfAtoms1 = moleculeType1.getNumberOfAtoms(); atomIndex1 < numberOfAtoms1;
                 ++atomIndex1)
                for (size_t atomIndex2 = 0, numberOfAtoms2 = moleculeType2.getNumberOfAtoms(); atomIndex2 < numberOfAtoms2;
                     ++atomIndex2)
                    if (!_isGuffPairSet[moleculeType1.getMoltype() - 1][moleculeType2.getMoltype() - 1]
                                       [moleculeType1.getAtomType(atomIndex1)][moleculeType2.getAtomType(atomIndex2)])
                        throw customException::GuffDatException(
                            std::format("No guff pair set for molecule types {} and {} and atom types {} and the {}",
                                        moleculeType1.getMoltype(),
                                        moleculeType2.getMoltype(),
                                        moleculeType1.getExternalAtomType(atomIndex1),
                                        moleculeType2.getExternalAtomType(atomIndex2)));
}