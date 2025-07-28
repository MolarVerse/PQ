/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "guffDatReader.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <exception>    // for exception
#include <format>       // for format
#include <fstream>      // for basic_istream, std::ifstream, std
#include <functional>   // for idestd::ntity
#include <memory>       // for make_shared
#include <ranges>       // for views::drop, for_each, ranges

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
#include "stringUtilities.hpp"   // for fileExists, getLineCommands, removeComments, splitString

using namespace input::guffdat;
using namespace settings;
using namespace utilities;
using namespace defaults;
using namespace customException;
using namespace simulationBox;
using namespace potential;
using namespace constants;

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
void input::guffdat::readGuffDat(engine::Engine &engine)
{
    if (!isNeeded(engine))
        return;

    engine.getStdoutOutput().writeRead(
        "Guffdat File",
        FileSettings::getGuffDatFileName()
    );
    engine.getLogOutput().writeRead(
        "Guffdat File",
        FileSettings::getGuffDatFileName()
    );

    GuffDatReader guffDat(engine);
    guffDat.setupGuffMaps();
    guffDat.read();
    guffDat.postProcessSetup();
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
bool input::guffdat::isNeeded(engine::Engine &engine)
{
    if (!Settings::isMMActivated())
        return false;
    else if (engine.getForceFieldPtr()->isNonCoulombicActivated())
        return false;
    else
        return true;
}

/**
 * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
 *
 * @param engine
 */
GuffDatReader::GuffDatReader(engine::Engine &engine) : _engine(engine)
{
    _fileName = FileSettings::getGuffDatFileName();
}

/**
 * @brief reads the guff.dat file
 *
 * @details the guff.dat file is read line by line. Each line is parsed and the
 * guffNonCoulombicPair is constructed. For further details about the entries of
 * the line see  the documentation of the guff.dat file.
 *
 * @throws GuffDatException command line has not 28 entries
 */
void GuffDatReader::read()
{
    std::ifstream fp(_fileName);
    std::string   line;

    while (getline(fp, line))
    {
        line = removeComments(line, "#");

        if (line.empty())
        {
            ++_lineNumber;
            continue;
        }

        auto lineCommands = getLineCommands(line, _lineNumber);

        if (lineCommands.size() != _NUMBER_OF_GUFF_ENTRIES_)
        {
            const auto message = std::format(
                "Invalid number of commands ({}) in line {} - {} are allowed",
                lineCommands.size(),
                _lineNumber,
                _NUMBER_OF_GUFF_ENTRIES_
            );
            throw GuffDatException(message);
        }

        parseLine(lineCommands);

        ++_lineNumber;
    }
}

/**
 * @brief constructs the guff dat 4d vectors
 *
 * @details resizes the 4d vectors of guffNonCoulomb and
 * _guffCoulombCoeffs in order to access elements with molTypes and
 * internal atomTypes
 *
 */
void GuffDatReader::setupGuffMaps()
{
    auto        &simBox    = _engine.getSimulationBox();
    const size_t nMolTypes = simBox.getMoleculeTypes().size();

    auto &guffNonCoulomb = dynamic_cast<GuffNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    guffNonCoulomb.resizeGuff(nMolTypes);
    _guffCoulombCoeffs.resize(nMolTypes);
    _isGuffPairSet.resize(nMolTypes);

    for (size_t i = 0; i < nMolTypes; ++i)
    {
        guffNonCoulomb.resizeGuff(i, nMolTypes);
        _guffCoulombCoeffs[i].resize(nMolTypes);
        _isGuffPairSet[i].resize(nMolTypes);
    }

    for (size_t i = 0; i < nMolTypes; ++i)
        for (size_t j = 0; j < nMolTypes; ++j)
        {
            auto      &molType    = simBox.getMoleculeType(i);
            const auto nAtomTypes = molType.getNumberOfAtomTypes();

            guffNonCoulomb.resizeGuff(i, j, nAtomTypes);
            _guffCoulombCoeffs[i][j].resize(nAtomTypes);
            _isGuffPairSet[i][j].resize(nAtomTypes);
        }

    for (size_t i = 0; i < nMolTypes; ++i)
        for (size_t j = 0; j < nMolTypes; ++j)
        {
            auto      &molType1    = simBox.getMoleculeType(i);
            const auto nAtomTypes1 = molType1.getNumberOfAtomTypes();

            for (size_t k = 0; k < nAtomTypes1; ++k)
            {
                auto        &molType2    = simBox.getMoleculeType(j);
                const size_t nAtomTypes2 = molType2.getNumberOfAtomTypes();

                guffNonCoulomb.resizeGuff(i, j, k, nAtomTypes2);
                _guffCoulombCoeffs[i][j][k].resize(nAtomTypes2);
                _isGuffPairSet[i][j][k].resize(nAtomTypes2);

                for (size_t l = 0; l < nAtomTypes2; ++l)
                    _isGuffPairSet[i][j][k][l] = false;
            }
        }
}

/**
 * @brief parses a line from the guff.dat file
 *
 * @details the line is parsed and the guffNonCoulombicPair is constructed. For
 * further details about the entries of the line see the documentation of the
 * guff.dat file
 *
 * @param lineCommands
 *
 * æTODO: introduce keyword to ignore coulomb preFactors and use moldescriptor
 * instead
 *
 * @throws GuffDatException if molecule or atom type is invalid
 */
void GuffDatReader::parseLine(const std::vector<std::string> &lineCommands)
{
    MoleculeType molecule1;
    MoleculeType molecule2;

    auto &simBox = _engine.getSimulationBox();

    try
    {
        molecule1 = simBox.findMoleculeType(stoul(lineCommands[0]));
        molecule2 = simBox.findMoleculeType(stoul(lineCommands[2]));
    }
    catch (const std::exception &)
    {
        throw GuffDatException(
            std::format("Invalid molecule type in line {}", _lineNumber)
        );
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
        throw GuffDatException(
            std::format("Invalid atom type in line {}", _lineNumber)
        );
    }

    double rncCutOff = stod(lineCommands[4]);

    if (rncCutOff < 0.0)
        rncCutOff = PotentialSettings::getCoulombRadiusCutOff();

    const double        coulombCoeff = stod(lineCommands[5]);
    std::vector<double> guffCoefficients;

    std::ranges::for_each(
        lineCommands | std::views::drop(6),
        [&guffCoefficients](const auto &entry)
        { guffCoefficients.push_back(stod(entry)); }
    );

    const size_t moltype1 = stoul(lineCommands[0]);
    const size_t moltype2 = stoul(lineCommands[2]);

    // clang-format off
    _guffCoulombCoeffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2] = coulombCoeff;
    _guffCoulombCoeffs[moltype2 - 1][moltype1 - 1][atomType2][atomType1] = coulombCoeff;
    _isGuffPairSet[moltype1 - 1][moltype2 - 1][atomType1][atomType2]     = true;
    _isGuffPairSet[moltype2 - 1][moltype1 - 1][atomType2][atomType1]     = true;
    // clang-format on

    addNonCoulombPair(
        moltype1,
        moltype2,
        atomType1,
        atomType2,
        guffCoefficients,
        rncCutOff
    );
}

/**
 * @brief checks which nonCoulombic type is given and adds the corresponding
 * nonCoulombic pair
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 *
 * @throws UserInputException if nonCoulombic type is invalid
 */
void GuffDatReader::addNonCoulombPair(
    const size_t               molType1,
    const size_t               molType2,
    const size_t               atomType1,
    const size_t               atomType2,
    const std::vector<double> &coefficients,
    const double               rncCutOff
)
{
    switch (PotentialSettings::getNonCoulombType())
    {
        using enum NonCoulombType;

        case LJ:
        {
            addLennardJonesPair(
                molType1,
                molType2,
                atomType1,
                atomType2,
                coefficients,
                rncCutOff
            );
            break;
        }
        case BUCKINGHAM:
        {
            addBuckinghamPair(
                molType1,
                molType2,
                atomType1,
                atomType2,
                coefficients,
                rncCutOff
            );
            break;
        }
        case MORSE:
        {
            addMorsePair(
                molType1,
                molType2,
                atomType1,
                atomType2,
                coefficients,
                rncCutOff
            );
            break;
        }
        case GUFF:
        {
            addGuffPair(
                molType1,
                molType2,
                atomType1,
                atomType2,
                coefficients,
                rncCutOff
            );
            break;
        }
        default:
        {
            throw UserInputException(
                std::format(
                    "Invalid nonCoulombic type {} given",
                    string(PotentialSettings::getNonCoulombType())
                )
            );
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
void GuffDatReader::addLennardJonesPair(
    const size_t               molType1,
    const size_t               molType2,
    const size_t               atomType1,
    const size_t               atomType2,
    const std::vector<double> &coefficients,
    const double               rncCutOff
)
{
    auto &guffNonCoulomb = dynamic_cast<GuffNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    const auto LJPair =
        LennardJonesPair(rncCutOff, coefficients[0], coefficients[2]);

    const auto [eCutOff, fCutOff] = LJPair.calculate(rncCutOff);

    guffNonCoulomb.setGuffNonCoulPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<LennardJonesPair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coefficients[0],
            coefficients[2]
        )
    );

    guffNonCoulomb.setGuffNonCoulPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<LennardJonesPair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coefficients[0],
            coefficients[2]
        )
    );
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
void GuffDatReader::addBuckinghamPair(
    const size_t               molType1,
    const size_t               molType2,
    const size_t               atomType1,
    const size_t               atomType2,
    const std::vector<double> &coefficients,
    const double               rncCutOff
)
{
    auto &guffNonCoulomb = dynamic_cast<GuffNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    const auto buckPair = BuckinghamPair(
        rncCutOff,
        coefficients[0],
        coefficients[1],
        coefficients[2]
    );
    const auto [eCutOff, fCutOff] = buckPair.calculate(rncCutOff);

    guffNonCoulomb.setGuffNonCoulPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<BuckinghamPair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coefficients[0],
            coefficients[1],
            coefficients[2]
        )
    );

    guffNonCoulomb.setGuffNonCoulPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<BuckinghamPair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coefficients[0],
            coefficients[1],
            coefficients[2]
        )
    );
}

/**
 * @brief adds a morse pair to the guffNonCoulombic potential
 *
 * @details first guff coefficient is the dissociationEnergy , second is the
 * wellWidth, third is the equilibriumDistance
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficients
 * @param rncCutOff
 */
void GuffDatReader::addMorsePair(
    const size_t               molType1,
    const size_t               molType2,
    const size_t               atomType1,
    const size_t               atomType2,
    const std::vector<double> &coeffs,
    const double               rncCutOff
)
{
    auto &guffNonCoulomb = dynamic_cast<GuffNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    // clang-format off
    const auto morsePair          = MorsePair(rncCutOff, coeffs[0], coeffs[1], coeffs[2]);
    const auto [eCutOff, fCutOff] = morsePair.calculate(rncCutOff);
    // clang-format on

    guffNonCoulomb.setGuffNonCoulPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<MorsePair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coeffs[0],
            coeffs[1],
            coeffs[2]
        )
    );

    guffNonCoulomb.setGuffNonCoulPair(
        {
            molType2,
            molType1,
            atomType2,
            atomType1,
        },
        std::make_shared<MorsePair>(
            rncCutOff,
            eCutOff,
            fCutOff,
            coeffs[0],
            coeffs[1],
            coeffs[2]
        )
    );
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
void GuffDatReader::addGuffPair(
    const size_t               molType1,
    const size_t               molType2,
    const size_t               atomType1,
    const size_t               atomType2,
    const std::vector<double> &coefficients,
    const double               rncCutOff
)
{
    auto &guffNonCoulomb = dynamic_cast<GuffNonCoulomb &>(
        _engine.getPotential().getNonCoulombPotential()
    );

    const auto guffPair           = GuffPair(rncCutOff, coefficients);
    const auto [eCutOff, fCutOff] = guffPair.calculate(rncCutOff);

    guffNonCoulomb.setGuffNonCoulPair(
        {molType1, molType2, atomType1, atomType2},
        std::make_shared<GuffPair>(rncCutOff, eCutOff, fCutOff, coefficients)
    );

    guffNonCoulomb.setGuffNonCoulPair(
        {molType2, molType1, atomType2, atomType1},
        std::make_shared<GuffPair>(rncCutOff, eCutOff, fCutOff, coefficients)
    );
}

/**
 * @brief post process guff.dat reading
 *
 * @details following steps are performed:
 * 1) check if all necessary guff pairs are set
 * 2) calculate the partial charges of the molecule types from the guff.dat
 * coulomb coefficients 3) check if the partial charges are in accordance with
 * all guff.dat entries
 *
 */
void GuffDatReader::postProcessSetup()
{
    checkNecessaryGuffPairs();
    calculatePartialCharges();
    _engine.getSimulationBox().setPartialChargesOfMoleculesFromMoleculeTypes();

    checkPartialCharges();
}

/**
 * @brief calculates the partial charges of the molecule types from the guff.dat
 * coulomb coefficients
 *
 * @details the partial charges are calculated via sqrt(coulombCoefficient /
 * _COULOMB_PREFACTOR_) of the self interactions and then set to the
 * molecule type
 *
 * @note the correct sign of the partial charge has already to be set in the
 * moldescriptor file - otherwise it is impossible to determine the sign of the
 * partial charge.
 *
 */
void GuffDatReader::calculatePartialCharges()
{
    auto      &simBox    = _engine.getSimulationBox();
    const auto nMolTypes = simBox.getMoleculeTypes().size();

    for (size_t i = 0; i < nMolTypes; ++i)
    {
        auto      *moleculeType = &(simBox.findMoleculeType(i + 1));
        const auto nAtoms       = moleculeType->getNumberOfAtoms();

        for (size_t j = 0; j < nAtoms; ++j)
        {
            // clang-format off
            const auto atomType     = moleculeType->getAtomType(j);
            const auto coulombCoeff = _guffCoulombCoeffs[i][i][atomType][atomType];
            // clang-format on

            const auto prefactor     = coulombCoeff / _COULOMB_PREFACTOR_;
            const auto prefactorSqrt = ::sqrt(prefactor);
            const auto prefactorSign = sign(moleculeType->getPartialCharge(j));
            const auto partialCharge = prefactorSqrt * prefactorSign;

            moleculeType->setPartialCharge(j, partialCharge);
        }
    }
}

/**
 * @brief checks if the partial charges are in accordance with all guff.dat
 * entries.
 *
 * @details The checks includes only moleculeTypes which are also used in the
 * simulationBox, meaning that only the used coulombCoefficient are checked. All
 * coulombCoefficient combinations have to be of the form q1 * q2 * 1 / (4 * pi
 * * epsilon_0)
 *
 * @throws GuffDatException if the partial charges are invalid
 *
 */
void GuffDatReader::checkPartialCharges()
{
    auto      &simBox    = _engine.getSimulationBox();
    const auto nMolTypes = simBox.getMoleculeTypes().size();

    for (size_t i = 0; i < nMolTypes; ++i)
    {
        const auto moleculeType1Optional = simBox.findMolecule(i + 1);

        Molecule moleculeType1;

        if (moleculeType1Optional == std::nullopt)
            continue;

        else
            moleculeType1 = moleculeType1Optional.value();

        for (size_t j = 0; j < nMolTypes; ++j)
        {
            const auto moleculeType2Optional = simBox.findMolecule(j + 1);

            Molecule moleculeType2;

            if (moleculeType2Optional == std::nullopt)
                continue;

            else
                moleculeType2 = moleculeType2Optional.value();

            const auto nAtomTypes1 = moleculeType1.getNumberOfAtomTypes();

            for (size_t k = 0; k < nAtomTypes1; ++k)
            {
                const auto nAtomTypes2 = moleculeType2.getNumberOfAtomTypes();
                for (size_t l = 0; l < nAtomTypes2; ++l)
                {
                    auto partialCharge1 = moleculeType1.getPartialCharge(k);
                    auto partialCharge2 = moleculeType2.getPartialCharge(l);

                    const auto coeff         = _guffCoulombCoeffs[i][j][k][l];
                    const auto chargeSquared = partialCharge1 * partialCharge2;
                    const auto prefactor = chargeSquared * _COULOMB_PREFACTOR_;

                    if (!compare(prefactor, coeff, 1e-6))
                        throw GuffDatException(
                            std::format(
                                "Invalid coulomb coefficient guff file for "
                                "molecule "
                                "types {} and {} and the {}. and the {}. atom "
                                "type. The coulomb "
                                "coefficient should "
                                "be {} but is {}",
                                i + 1,
                                j + 1,
                                k + 1,
                                l + 1,
                                prefactor,
                                coeff
                            )
                        );
                }
            }
        }
    }
}

/**
 * @brief check if all necessary guff pairs are set
 *
 * @details the necessary guff pairs are determined by the molecule types which
 * are used in the simulation box in _molecules (defined by the restart file and
 * not in the moldescriptor file) - the moldescriptor can have more molecule
 * types than the actual simulation box (e.g. if the moldescriptor is used for
 * multiple simulations).
 *
 * @throws GuffDatException if a necessary guff pair is not set
 */
void GuffDatReader::checkNecessaryGuffPairs()
{
    auto      &simBox                 = _engine.getSimulationBox();
    const auto necessaryMoleculeTypes = simBox.findNecessaryMoleculeTypes();

    for (const auto &moleculeType1 : necessaryMoleculeTypes)
        for (const auto &moleculeType2 : necessaryMoleculeTypes)
        {
            const auto nAtoms1 = moleculeType1.getNumberOfAtoms();
            for (size_t atomIndex1 = 0; atomIndex1 < nAtoms1; ++atomIndex1)
            {
                const auto nAtoms2 = moleculeType2.getNumberOfAtoms();
                for (size_t atomIndex2 = 0; atomIndex2 < nAtoms2; ++atomIndex2)
                    if (!_isGuffPairSet[moleculeType1.getMoltype() - 1]
                                       [moleculeType2.getMoltype() - 1]
                                       [moleculeType1.getAtomType(atomIndex1)]
                                       [moleculeType2.getAtomType(atomIndex2)])

                        throw GuffDatException(
                            std::format(
                                "No guff pair set for molecule types {} and {} "
                                "and "
                                "atom types {} and "
                                "the {}",
                                moleculeType1.getMoltype(),
                                moleculeType2.getMoltype(),
                                moleculeType1.getExternalAtomType(atomIndex1),
                                moleculeType2.getExternalAtomType(atomIndex2)
                            )
                        );
            }
        }
}

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief set the filename of the guff.dat file
 *
 * @param filename
 */
void GuffDatReader::setFilename(const std::string_view &filename)
{
    _fileName = filename;
}

/**
 * @brief set the guffCoulombCoefficients
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param coefficient
 */
void GuffDatReader::setGuffCoulombCoefficients(
    const size_t molType1,
    const size_t molType2,
    const size_t atomType1,
    const size_t atomType2,
    const double coefficient
)
{
    _guffCoulombCoeffs[molType1][molType2][atomType1][atomType2] = coefficient;
}

/**
 * @brief set if a guff pair is set
 *
 * @param molType1
 * @param molType2
 * @param atomType1
 * @param atomType2
 * @param isSet
 */
void GuffDatReader::setIsGuffPairSet(
    const size_t molType1,
    const size_t molType2,
    const size_t atomType1,
    const size_t atomType2,
    const bool   isSet
)
{
    _isGuffPairSet[molType1][molType2][atomType1][atomType2] = isSet;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief Get the Guff Coulomb Coefficients object
 *
 * @return pq::stlVector4d&
 */
pq::stlVector4d &GuffDatReader::getGuffCoulombCoefficients()
{
    return _guffCoulombCoeffs;
}

/**
 * @brief Get the Is Guff Pair Set object
 *
 * @return pq::stlVector4dBool&
 */
pq::stlVector4dBool &GuffDatReader::getIsGuffPairSet()
{
    return _isGuffPairSet;
}