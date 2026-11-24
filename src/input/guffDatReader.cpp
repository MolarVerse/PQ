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

#include <algorithm>
#include <cmath>
#include <exception>
#include <format>
#include <fstream>
#include <memory>
#include <ranges>
#include <tuple>

#include "buckinghamPair.hpp"
#include "constants.hpp"
#include "defaults.hpp"
#include "engine.hpp"
#include "enums/potential.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "guffNonCoulomb.hpp"
#include "guffPair.hpp"
#include "lennardJonesPair.hpp"
#include "mathUtilities.hpp"
#include "molecule.hpp"
#include "morsePair.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "stringUtilities.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "waterModelSettings.hpp"

namespace input::guffdat
{

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
    void readGuffDat(engine::Engine &engine)
    {
        if (!isNeeded(engine.getSettings()))
            return;

        const auto &guffdatFileName =
            engine.getSettings().fileSettings.getGuffDatFileName();

        out::StdoutOutput::writeRead("Guffdat File", guffdatFileName);
        engine.getLogOutput().writeRead("Guffdat File", guffdatFileName);

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
    bool isNeeded(const Settings &settings)
    {
        if (!isMMActivated(settings.generalSettings.getJobtype()))
            return false;

        if (settings.forceFieldSettings.isNonCoulombicActive())
            return false;

        return true;
    }

    /**
     * @brief Construct a new Guff Dat Reader:: Guff Dat Reader object
     *
     * @param engine
     */
    GuffDatReader::GuffDatReader(engine::Engine &engine) : _engine(engine)
    {
        _fileName = _engine.getSettings().fileSettings.getGuffDatFileName();
    }

    /**
     * @brief reads the guff.dat file
     *
     * @details the guff.dat file is read line by line. Each line is parsed and
     * the guffNonCoulombicPair is constructed. For further details about the
     * entries of the line see  the documentation of the guff.dat file.
     *
     * @throws exc::GuffDatException command line has not 28 entries
     */
    void GuffDatReader::read()
    {
        std::ifstream file(_fileName);
        std::string   line;

        while (getline(file, line))
        {
            line = utilities::removeComments(line, "#");

            if (line.empty())
            {
                ++_lineNumber;
                continue;
            }

            auto lineCommands = utilities::getLineCommands(line, _lineNumber);

            if (lineCommands.size() != defaults::NUMBER_OF_GUFF_ENTRIES)
            {
                const auto message = std::format(
                    "Invalid number of commands ({}) in line {} - {} are "
                    "allowed",
                    lineCommands.size(),
                    _lineNumber,
                    defaults::NUMBER_OF_GUFF_ENTRIES
                );
                throw exc::GuffDatException(message);
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
        auto        &simBox    = _engine.get_simulationBox();
        const size_t nMolTypes = simBox.getMoleculeTypes().size();

        auto &guffNonCoulomb = dynamic_cast<pot::GuffNonCoulomb &>(
            _engine.getPotential()->getNonCoulombPotential()
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
        {
            for (size_t j = 0; j < nMolTypes; ++j)
            {
                auto      &molType    = simBox.getMoleculeType(i);
                const auto nAtomTypes = molType.getNumberOfAtomTypes();

                guffNonCoulomb.resizeGuff(i, j, nAtomTypes);
                _guffCoulombCoeffs[i][j].resize(nAtomTypes);
                _isGuffPairSet[i][j].resize(nAtomTypes);
            }
        }

        for (size_t i = 0; i < nMolTypes; ++i)
        {
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
    }

    /**
     * @brief parses a line from the guff.dat file
     *
     * @details the line is parsed and the guffNonCoulombicPair is constructed.
     * For further details about the entries of the line see the documentation
     * of the guff.dat file
     *
     * @param lineCommands
     *
     * æTODO: introduce keyword to ignore coulomb preFactors and use
     * moldescriptor instead
     *
     * @throws exc::GuffDatException if molecule or atom type is invalid
     */
    void GuffDatReader::parseLine(const std::vector<std::string> &lineCommands)
    {
        const MolType moltype1{stoul(lineCommands[0])};
        const MolType moltype2{stoul(lineCommands[2])};

        auto &simBox = _engine.get_simulationBox();

        if (bothMoltypesAreWaterType(moltype1, moltype2))
            return;

        molsys::MoleculeType molecule1;
        molsys::MoleculeType molecule2;

        try
        {
            molecule1 = simBox.findMoleculeType(moltype1);
            molecule2 = simBox.findMoleculeType(moltype2);
        }
        catch (const std::exception &)
        {
            throw exc::GuffDatException(
                std::format("Invalid molecule type in line {}", _lineNumber)
            );
        }

        AtomType atomType1{0};
        AtomType atomType2{0};

        try
        {
            atomType1 = molecule1.getInternalAtomType(
                ExtAtomType{stoul(lineCommands[1])}
            );
            atomType2 = molecule2.getInternalAtomType(
                ExtAtomType{stoul(lineCommands[3])}
            );
        }
        catch (const std::exception &)
        {
            throw exc::GuffDatException(
                std::format("Invalid atom type in line {}", _lineNumber)
            );
        }

        Distance rncCutoff = stod(lineCommands[4]) * angstrom;

        if (rncCutoff < 0.0)
            rncCutoff =
                _engine.getSettings().potentialSettings.getCoulombRadiusCutoff(
                );

        constexpr auto coulombCoeffIndex = 5;
        const auto     coulombCoeff =
            stod(lineCommands[coulombCoeffIndex]) * kcal_per_mol * angstrom;
        std::vector<double> guffCoefficients;

        std::ranges::for_each(
            lineCommands | std::views::drop(coulombCoeffIndex + 1),
            [&guffCoefficients](const auto &entry)
            { guffCoefficients.push_back(stod(entry)); }
        );

        _guffCoulombCoeffs[moltype1.get() - 1][moltype2.get() - 1]
                          [atomType1.get()][atomType2.get()] = coulombCoeff;
        _guffCoulombCoeffs[moltype2.get() - 1][moltype1.get() - 1]
                          [atomType2.get()][atomType1.get()] = coulombCoeff;

        _isGuffPairSet[moltype1.get() - 1][moltype2.get() - 1][atomType1.get()]
                      [atomType2.get()] = true;
        _isGuffPairSet[moltype2.get() - 1][moltype1.get() - 1][atomType2.get()]
                      [atomType1.get()] = true;

        addNonCoulombPair(
            moltype1,
            moltype2,
            atomType1,
            atomType2,
            guffCoefficients,
            rncCutoff
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
     * @param rncCutoff
     *
     * @throws exc::UserInputException if nonCoulombic type is invalid
     */
    void GuffDatReader::addNonCoulombPair(
        MolType                    molType1,
        MolType                    molType2,
        AtomType                   atomType1,
        AtomType                   atomType2,
        const std::vector<double> &coefficients,
        const Distance            &rncCutoff
    )
    {
        switch (_engine.getSettings().potentialSettings.getNonCoulombType())
        {
            case NonCoulombType::LJ:
            {
                addLennardJonesPair(
                    molType1,
                    molType2,
                    atomType1,
                    atomType2,
                    coefficients,
                    rncCutoff
                );
                break;
            }
            case NonCoulombType::BUCKINGHAM:
            {
                addBuckinghamPair(
                    molType1,
                    molType2,
                    atomType1,
                    atomType2,
                    coefficients,
                    rncCutoff
                );
                break;
            }
            case NonCoulombType::MORSE:
            {
                addMorsePair(
                    molType1,
                    molType2,
                    atomType1,
                    atomType2,
                    coefficients,
                    rncCutoff
                );
                break;
            }
            case NonCoulombType::GUFF:
            {
                std::array<double, defaults::NUM_GUFF_COEFFICIENTS> coeffs;
                for (size_t i = 0; i < coefficients.size(); ++i)
                    coeffs.at(i) = coefficients[i];

                addGuffPair(
                    molType1,
                    molType2,
                    atomType1,
                    atomType2,
                    coeffs,
                    rncCutoff
                );
                break;
            }
            case NonCoulombType::LJ_9_12:
            case NonCoulombType::NONE:
            {
                throw exc::UserInputException(
                    std::format(
                        "Invalid nonCoulombic type {} given",
                        NonCoulombTypeMeta::toString(
                            _engine.getSettings()
                                .potentialSettings.getNonCoulombType()
                        )
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
     * @param rncCutoff
     */
    void GuffDatReader::addLennardJonesPair(
        MolType                    molType1,
        MolType                    molType2,
        AtomType                   atomType1,
        AtomType                   atomType2,
        const std::vector<double> &coefficients,
        const Distance            &rncCutoff
    )
    {
        auto &guffNonCoulomb = dynamic_cast<pot::GuffNonCoulomb &>(
            _engine.getPotential()->getNonCoulombPotential()
        );

        const auto params = LJParams{
            .c6  = coefficients[0] * kcal_per_mol * angstrom6,
            .c12 = coefficients[2] * kcal_per_mol * angstrom12
        };
        const auto LJPair = pot::LennardJonesPair(rncCutoff, params);

        const auto [eCutoff, fCutoff] = LJPair.calculate(rncCutoff);

        guffNonCoulomb.setGuffNonCoulPair(
            {molType1, molType2, atomType1, atomType2},
            std::make_shared<pot::LennardJonesPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
            )
        );

        guffNonCoulomb.setGuffNonCoulPair(
            {molType2, molType1, atomType2, atomType1},
            std::make_shared<pot::LennardJonesPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
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
     * @param rncCutoff
     */
    void GuffDatReader::addBuckinghamPair(
        MolType                    molType1,
        MolType                    molType2,
        AtomType                   atomType1,
        AtomType                   atomType2,
        const std::vector<double> &coefficients,
        const Distance            &rncCutoff
    )
    {
        auto &guffNonCoulomb = dynamic_cast<pot::GuffNonCoulomb &>(
            _engine.getPotential()->getNonCoulombPotential()
        );

        const auto params = BuckinghamParams{
            .scaling = coefficients[0] * kcal_per_mol,
            .dRho    = coefficients[1] / angstrom,
            .c6      = coefficients[2] * kcal_per_mol * angstrom6
        };
        const auto buckPair           = pot::BuckinghamPair(rncCutoff, params);
        const auto [eCutoff, fCutoff] = buckPair.calculate(rncCutoff);

        guffNonCoulomb.setGuffNonCoulPair(
            {molType1, molType2, atomType1, atomType2},
            std::make_shared<pot::BuckinghamPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
            )
        );

        guffNonCoulomb.setGuffNonCoulPair(
            {molType2, molType1, atomType2, atomType1},
            std::make_shared<pot::BuckinghamPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
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
     * @param rncCutoff
     */
    void GuffDatReader::addMorsePair(
        MolType                    molType1,
        MolType                    molType2,
        AtomType                   atomType1,
        AtomType                   atomType2,
        const std::vector<double> &coefficients,
        const Distance            &rncCutoff
    )
    {
        auto &guffNonCoulomb = dynamic_cast<pot::GuffNonCoulomb &>(
            _engine.getPotential()->getNonCoulombPotential()
        );

        const auto params = MorseParams{
            .dissociationEnergy  = coefficients[0] * kcal_per_mol,
            .wellWidth           = coefficients[1] / angstrom,
            .equilibriumDistance = coefficients[2] * angstrom
        };

        const auto morsePair          = pot::MorsePair(rncCutoff, params);
        const auto [eCutoff, fCutoff] = morsePair.calculate(rncCutoff);

        guffNonCoulomb.setGuffNonCoulPair(
            {molType1, molType2, atomType1, atomType2},
            std::make_shared<pot::MorsePair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
            )
        );

        guffNonCoulomb.setGuffNonCoulPair(
            {
                molType2,
                molType1,
                atomType2,
                atomType1,
            },
            std::make_shared<pot::MorsePair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                params
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
     * @param rncCutoff
     */
    void GuffDatReader::addGuffPair(
        MolType                                                    molType1,
        MolType                                                    molType2,
        AtomType                                                   atomType1,
        AtomType                                                   atomType2,
        const std::array<double, defaults::NUM_GUFF_COEFFICIENTS> &coefficients,
        const Distance                                            &rncCutoff
    )
    {
        auto &guffNonCoulomb = dynamic_cast<pot::GuffNonCoulomb &>(
            _engine.getPotential()->getNonCoulombPotential()
        );

        const auto guffPair           = pot::GuffPair(rncCutoff, coefficients);
        const auto [eCutoff, fCutoff] = guffPair.calculate(rncCutoff);

        guffNonCoulomb.setGuffNonCoulPair(
            {molType1, molType2, atomType1, atomType2},
            std::make_shared<pot::GuffPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                coefficients
            )
        );

        guffNonCoulomb.setGuffNonCoulPair(
            {molType2, molType1, atomType2, atomType1},
            std::make_shared<pot::GuffPair>(
                rncCutoff,
                eCutoff,
                fCutoff,
                coefficients
            )
        );
    }

    /**
     * @brief post process guff.dat reading
     *
     * @details following steps are performed:
     * 1) check if all necessary guff pairs are set
     * 2) calculate the partial charges of the molecule types from the guff.dat
     * coulomb coefficients 3) check if the partial charges are in accordance
     * with all guff.dat entries
     *
     */
    void GuffDatReader::postProcessSetup()
    {
        checkNecessaryGuffPairs();
        calculatePartialCharges();
        _engine.get_simulationBox()
            .setPartialChargesOfMoleculesFromMoleculeTypes();

        checkPartialCharges();
    }

    /**
     * @brief calculates the partial charges of the molecule types from the
     * guff.dat coulomb coefficients
     *
     * @details the partial charges are calculated via sqrt(coulombCoefficient /
     * _COULOMB_PREFACTOR_) of the self interactions and then set to the
     * molecule type
     *
     * @note the correct sign of the partial charge has already to be set in the
     * moldescriptor file - otherwise it is impossible to determine the sign of
     * the partial charge.
     *
     */
    void GuffDatReader::calculatePartialCharges()
    {
        auto      &simBox    = _engine.get_simulationBox();
        const auto nMolTypes = simBox.getMoleculeTypes().size();
        const auto waterType = simBox.getWaterType();

        for (size_t i = 0; i < nMolTypes; ++i)
        {
            const MolType molType{i + 1};
            // Skip water type molecules - their charges come from moldescriptor
            if (waterType.has_value() && molType == waterType.value())
                continue;

            auto      *moleculeType = &(simBox.findMoleculeType(molType));
            const auto nAtoms       = moleculeType->getNumberOfAtoms();

            for (AtomIndex j{0}; j.get() < nAtoms; ++j)
            {
                const auto atomType = moleculeType->getAtomType(j);
                const auto coulombCoeff =
                    _guffCoulombCoeffs[i][i][atomType.get()][atomType.get()];

                const auto prefactor     = coulombCoeff / coulomb_prefactor;
                const auto prefactorSqrt = mstd::sqrt(prefactor);
                const auto prefactorSign =
                    utilities::sign(moleculeType->getPartialCharge(j));
                const auto partialCharge = prefactorSqrt * prefactorSign;

                moleculeType->setPartialCharge(j, partialCharge);
            }
        }
    }

    /**
     * @brief checks if the partial charges are in accordance with all guff.dat
     * entries.
     *
     * @details The checks includes only moleculeTypes which are also used in
     * the simulationBox, meaning that only the used coulombCoefficient are
     * checked. All coulombCoefficient combinations have to be of the form q1 *
     * q2 * 1 / (4 * pi
     * * epsilon_0)
     *
     * @throws exc::GuffDatException if the partial charges are invalid
     *
     */
    void GuffDatReader::checkPartialCharges()
    {
        auto      &simBox    = _engine.get_simulationBox();
        const auto nMolTypes = simBox.getMoleculeTypes().size();

        for (size_t i = 0; i < nMolTypes; ++i)
        {
            const auto moleculeType1Optional =
                simBox.findMolecule(MolType{i + 1});

            molsys::Molecule moleculeType1;

            if (moleculeType1Optional == std::nullopt)
                continue;

            moleculeType1 = moleculeType1Optional.value();

            for (size_t j = 0; j < nMolTypes; ++j)
            {
                const auto moleculeType2Optional =
                    simBox.findMolecule(MolType{j + 1});

                molsys::Molecule moleculeType2;

                if (moleculeType2Optional == std::nullopt)
                    continue;

                if (bothMoltypesAreWaterType(
                        moleculeType1.getMoltype(),
                        moleculeType2Optional.value().getMoltype()
                    ))
                    continue;

                moleculeType2 = moleculeType2Optional.value();

                const auto nAtomTypes1 = moleculeType1.getNumberOfAtomTypes();

                for (AtomIndex k{0}; k.get() < nAtomTypes1; ++k)
                {
                    const auto nAtomTypes2 =
                        moleculeType2.getNumberOfAtomTypes();
                    for (AtomIndex l{0}; l.get() < nAtomTypes2; ++l)
                    {
                        auto partialCharge1 = moleculeType1.getPartialCharge(k);
                        auto partialCharge2 = moleculeType2.getPartialCharge(l);

                        const auto coeff =
                            _guffCoulombCoeffs[i][j][k.get()][l.get()];
                        const auto chargeSquared =
                            partialCharge1 * partialCharge2;
                        const auto prefactor =
                            chargeSquared * coulomb_prefactor;

                        if (!utilities::compare(
                                prefactor,
                                coeff,
                                GUFF_DAT_COULOMB_PREFACTOR_THRESHOLD
                            ))
                        {
                            throw exc::GuffDatException(
                                std::format(
                                    "Invalid coulomb coefficient guff file for "
                                    "molecule types {} and {} and the {}. and "
                                    "the {}. atom type. The coulomb "
                                    "coefficient should be {} but is {}",
                                    i + 1,
                                    j + 1,
                                    k.get() + 1,
                                    l.get() + 1,
                                    prefactor.in(kcal_per_mol * angstrom),
                                    coeff
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief check if all necessary guff pairs are set
     *
     * @details the necessary guff pairs are determined by the molecule types
     * which are used in the simulation box in _molecules (defined by the
     * restart file and not in the moldescriptor file) - the moldescriptor can
     * have more molecule types than the actual simulation box (e.g. if the
     * moldescriptor is used for multiple simulations).
     *
     * @throws exc::GuffDatException if a necessary guff pair is not set
     */
    void GuffDatReader::checkNecessaryGuffPairs()
    {
        auto      &simBox                 = _engine.get_simulationBox();
        const auto necessaryMoleculeTypes = simBox.findNecessaryMoleculeTypes();

        for (const auto &moleculeType1 : necessaryMoleculeTypes)
        {
            for (const auto &moleculeType2 : necessaryMoleculeTypes)
            {
                if (bothMoltypesAreWaterType(
                        moleculeType1.getMoltype(),
                        moleculeType2.getMoltype()
                    ))
                    continue;

                const auto nAtoms1 = moleculeType1.getNumberOfAtoms();
                for (AtomIndex atomIndex1{0}; atomIndex1.get() < nAtoms1;
                     ++atomIndex1)
                {
                    const auto nAtoms2 = moleculeType2.getNumberOfAtoms();
                    for (AtomIndex atomIndex2{0}; atomIndex2.get() < nAtoms2;
                         ++atomIndex2)
                    {
                        if (!_isGuffPairSet
                                [moleculeType1.getMoltype().get() - 1]
                                [moleculeType2.getMoltype().get() - 1]
                                [moleculeType1.getAtomType(atomIndex1).get()]
                                [moleculeType2.getAtomType(atomIndex2).get()])
                        {
                            throw exc::GuffDatException(
                                std::format(
                                    "No guff pair set for molecule types {} "
                                    "and {} "
                                    "and "
                                    "atom types {} and "
                                    "the {}",
                                    moleculeType1.getMoltype().toString(),
                                    moleculeType2.getMoltype().toString(),
                                    moleculeType1
                                        .getExternalAtomType(atomIndex1)
                                        .toString(),
                                    moleculeType2
                                        .getExternalAtomType(atomIndex2)
                                        .toString()
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Determine if both provided molecule types are of the
     *        water molecule type.
     *
     * This helper checks whether the inter-water model is enabled and whether
     * both molecule type indices match the simulation box's configured water
     * type. Note that molecule type indices are expected to be 1-based in the
     * surrounding code.
     *
     * @param molType1 1-based index of the first molecule type
     * @param molType2 1-based index of the second molecule type
     * @return true if the inter-water model is set and both indices equal the
     *         configured water type; false otherwise
     */
    bool GuffDatReader::bothMoltypesAreWaterType(
        MolType molType1,
        MolType molType2
    )
    {
        auto      &simBox    = _engine.get_simulationBox();
        const auto waterType = simBox.getWaterType();

        const auto isInterWaterModelSet =
            _engine.getSettings()
                .waterModelSettings.isInterWaterModelSetAndNotNone();

        return isInterWaterModelSet && waterType.has_value() &&
               molType1 == waterType.value() && molType2 == waterType.value();
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
        MolType                   molType1,
        MolType                   molType2,
        AtomType                  atomType1,
        AtomType                  atomType2,
        const EnergyLengthPerMol &coefficient
    )
    {
        _guffCoulombCoeffs[molType1.get()][molType2.get()][atomType1.get()]
                          [atomType2.get()] = coefficient;
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
        MolType  molType1,
        MolType  molType2,
        AtomType atomType1,
        AtomType atomType2,
        bool     isSet
    )
    {
        _isGuffPairSet[molType1.get()][molType2.get()][atomType1.get()]
                      [atomType2.get()] = isSet;
    }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    /**
     * @brief Get the Guff Coulomb Coefficients object
     *
     * @return
     * std::vector<std::vector<std::vector<std::vector<EnergyLengthPerMol>>>>&
     */
    std::vector<std::vector<std::vector<std::vector<EnergyLengthPerMol>>>> &GuffDatReader::
        getGuffCoulombCoefficients()
    {
        return _guffCoulombCoeffs;
    }

    /**
     * @brief Get the Is Guff Pair Set object
     *
     * @return std::vector<std::vector<std::vector<std::vector<bool>>>>&
     */
    std::vector<std::vector<std::vector<std::vector<bool>>>> &GuffDatReader::
        getIsGuffPairSet()
    {
        return _isGuffPairSet;
    }

}   // namespace input::guffdat
