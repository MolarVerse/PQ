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

#include "simulationBoxSetup.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "atom.hpp"
#include "atomNumberMap.hpp"
#include "constants/conversionFactors.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "logOutput.hpp"
#include "maxwellBoltzmann.hpp"
#include "molecule.hpp"
#include "outputMessages.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "simulationBoxSettings.hpp"
#include "stdoutOutput.hpp"
#include "stringUtilities.hpp"
#include "strongTypes.hpp"

namespace setup
{

    /**
     * @brief wrapper to create SetupSimulationBox object and call setup
     *
     * @param engine
     */
    void setupSimulationBox(engine::Engine &engine)
    {
        out::StdoutOutput::writeSetup("simulation box");
        engine.getLogOutput().writeSetup("simulation box");

        SimulationBoxSetup simulationBoxSetup(engine);
        simulationBoxSetup.setup();
    }

    /**
     * @brief Construct a new Simulation Box Setup object
     *
     * @param engine
     */
    SimulationBoxSetup::SimulationBoxSetup(engine::Engine &engine)
        : _engine(engine)
    {
    }

    /**
     * @brief setup simulation box
     *
     */
    void SimulationBoxSetup::setup()
    {
        setAtomNames();
        setAtomTypes();
        if (_engine.getForceField()->isNonCoulombicActivated())
            setExternalVDWTypes();
        setPartialCharges();

        setAtomMasses();
        setAtomicNumbers();
        calculateTotalCharge();

        auto &simBox = _engine.getSimulationBox();

        simBox.calculateTotalMass();

        checkBoxSettings();
        checkRcCutoff();

        simBox.calculateDegreesOfFreedom();
        simBox.calculateCenterOfMassMolecules();

        checkZeroVelocities();
        initVelocities();

        writeSetupInfo();
    }

    /**
     * @brief set all atomNames in atoms from moleculeTypes
     *
     */
    void SimulationBoxSetup::setAtomNames()
    {
        auto &simBox = _engine.getSimulationBox();

        auto setAtomNamesOfMolecule = [&simBox](auto &molecule)
        {
            const auto &molType = molecule.getMoltype();
            if (molType == MolType{0})
                return;

            const auto moleculeType  = simBox.findMoleculeType(molType);
            const auto numberOfAtoms = molecule.getNumberOfAtoms();

            for (AtomIndex i{0}; i.get() < numberOfAtoms; ++i)
                molecule.getAtom(i).setName(moleculeType.getAtomName(i));
        };

        std::ranges::for_each(simBox.getMolecules(), setAtomNamesOfMolecule);

        std::ranges::for_each(
            simBox.getAtoms(),
            [](auto &atom)
            {
                atom->setName(
                    utilities::firstLetterToUpperCaseCopy(atom->getName())
                );
            }
        );
    }

    /**
     * @brief set all external and internal atom types for _atoms from
     * _moleculeTypes
     *
     */
    void SimulationBoxSetup::setAtomTypes()
    {
        auto &simBox = _engine.getSimulationBox();

        auto setAtomTypesOfMolecule = [&simBox](auto &molecule)
        {
            const auto &molType = molecule.getMoltype();

            if (molType == MolType{0})
                return;

            auto       moleculeType = simBox.findMoleculeType(molType);
            const auto nAtoms       = molecule.getNumberOfAtoms();

            for (AtomIndex i{0}; i.get() < nAtoms; ++i)
            {
                const auto externalAtomType =
                    moleculeType.getExternalAtomType(i);
                molecule.getAtom(i).setAtomType(moleculeType.getAtomType(i));
                molecule.getAtom(i).setExternalAtomType(externalAtomType);
            }
        };

        std::ranges::for_each(simBox.getMolecules(), setAtomTypesOfMolecule);
    }

    /**
     * @brief set all external van der Waals types in atoms from moleculeTypes
     *
     */
    void SimulationBoxSetup::setExternalVDWTypes()
    {
        auto &simBox = _engine.getSimulationBox();

        auto setExternalVDWTypesOfMolecule = [&simBox](auto &molecule)
        {
            const auto &molType = molecule.getMoltype();

            if (molType == MolType{0})
                return;

            auto       moleculeType = simBox.findMoleculeType(molType);
            const auto nAtoms       = molecule.getNumberOfAtoms();

            if (moleculeType.getExternalGlobalVDWTypes().size() != nAtoms)
            {
                throw exc::UserInputException(
                    std::format(
                        "Ooooops! Something went wrong. Please check again "
                        "your "
                        "input files and please report this issue to the "
                        "developers via "
                        "https://github.com/Molarverse/PQ/issues.\n"
                        "in simulationBoxSetup.cpp: setExternalVDWTypes()\n"
                        "Number of external VDW types ({}) does not match "
                        "number "
                        "of atoms ({}) in molecule type {}",
                        moleculeType.getExternalGlobalVDWTypes().size(),
                        nAtoms,
                        molType.toString()
                    )
                );
            }

            for (AtomIndex i{0}; i.get() < nAtoms; ++i)
            {
                const auto extVDWType =
                    moleculeType.getExternalGlobalVDWTypes()[i.get()];
                molecule.getAtom(i).setExternalGlobalVDWType(extVDWType);
            }
        };

        std::ranges::for_each(
            simBox.getMolecules(),
            setExternalVDWTypesOfMolecule
        );
    }

    /**
     * @brief set all partial charges in atoms from _moleculeTypes
     *
     */
    void SimulationBoxSetup::setPartialCharges()
    {
        auto &simBox = _engine.getSimulationBox();

        auto setPartialChargesOfMolecule = [&simBox](auto &molecule)
        {
            const auto &molType = molecule.getMoltype();

            if (molType == MolType{0})
                return;

            auto        moleculeType = simBox.findMoleculeType(molType);
            const auto &nAtoms       = molecule.getNumberOfAtoms();

            for (AtomIndex i{0}; i.get() < nAtoms; ++i)
            {
                const auto partialCharge =
                    moleculeType.getPartialCharges()[i.get()];
                molecule.getAtom(i).setPartialCharge(partialCharge);
            }
        };

        std::ranges::for_each(
            simBox.getMolecules(),
            setPartialChargesOfMolecule
        );
    }

    /**
     * @brief Sets the mass of each atom in the simulation box
     *
     * @throw MolDescriptorException if atom name is invalid
     */
    void SimulationBoxSetup::setAtomMasses()
    {
        auto setAtomMasses = [](::molsys::Molecule &molecule)
        {
            for (auto &atom : molecule.getAtoms()) atom->initMass();
        };

        auto &molecules = _engine.getSimulationBox().getMolecules();
        std::ranges::for_each(molecules, setAtomMasses);
    }

    /**
     * @brief Sets the atomic number of each atom in the simulation box
     *
     * @throw MolDescriptorException if atom name is invalid
     */
    void SimulationBoxSetup::setAtomicNumbers()
    {
        auto setAtomicNumbers = [](::molsys::Molecule &molecule)
        {
            const auto nAtoms = molecule.getNumberOfAtoms();
            for (AtomIndex i{0}; i.get() < nAtoms; ++i)
            {
                const auto keyword =
                    utilities::toLowerCopy(molecule.getAtomName(i));

                if (!atomNumberMap.contains(keyword))
                {
                    throw exc::MolDescriptorException(
                        "Invalid atom name \"" + keyword + "\""
                    );
                }

                molecule.getAtom(i).setAtomicNumber(atomNumberMap.at(keyword));
            }
        };

        auto &molecules = _engine.getSimulationBox().getMolecules();
        std::ranges::for_each(molecules, setAtomicNumbers);
    }

    /**
     * @brief Calculates the total charge of the simulation box
     */
    void SimulationBoxSetup::calculateTotalCharge()
    {
        double totalCharge = 0.0;

        auto calcMolCharge = [&totalCharge](const ::molsys::Molecule &mol)
        {
            const auto &charges = mol.getPartialCharges();
            totalCharge += std::accumulate(charges.begin(), charges.end(), 0.0);
        };

        auto &molecules = _engine.getSimulationBox().getMolecules();
        std::ranges::for_each(molecules, calcMolCharge);

        _engine.getSimulationBox().setTotalCharge(totalCharge);
    }

    /**
     * @brief Checks if the box dimensions and density are set and calculates
     * the missing values
     *
     * @throw exc::UserInputException if box dimensions and density are not set
     * @throw UserInputExceptionWarning if density and box dimensions are set.
     * Density will be ignored.
     *
     * @note If density is set, box dimensions will be calculated from density.
     * @note If box dimensions are set, density will be calculated from box
     * dimensions.
     */
    void SimulationBoxSetup::checkBoxSettings()
    {
        auto &simBox = _engine.getSimulationBox();

        const auto isDensitySet =
            settings::SimulationBoxSettings::getDensitySet();
        const auto isBoxSet = settings::SimulationBoxSettings::getBoxSet();

        if (!isDensitySet && !isBoxSet)
            throw exc::UserInputException("Box dimensions and density not set");

        if (!isBoxSet)
        {
            const auto boxDimensions = simBox.calcBoxDimFromDensity();

            simBox.setBoxDimensions(boxDimensions);
            simBox.setVolume(simBox.calculateVolume());
        }
        else if (!settings::SimulationBoxSettings::getDensitySet())
        {
            const auto volume = simBox.calculateVolume();
            const auto density =
                simBox.getTotalMass() / volume * AMU_PER_ANGSTROM3_TO_KG_PER_L;

            simBox.setVolume(volume);
            simBox.setDensity(density);
        }
        else
        {
            const auto volume     = simBox.calculateVolume();
            const auto convFactor = AMU_PER_ANGSTROM3_TO_KG_PER_L;
            const auto density    = simBox.getTotalMass() / volume * convFactor;

            simBox.setVolume(volume);
            simBox.setDensity(density);

            _engine.getLogOutput().writeDensityWarning();
            out::StdoutOutput::writeDensityWarning();
        }

        _engine.getPhysicalData().setVolume(simBox.getVolume());
        _engine.getPhysicalData().setDensity(simBox.getDensity());
    }

    /**
     * @brief Checks if the cutoff radius is larger than half of the minimal box
     * dimension
     *
     * @throw InputFileException if cutoff radius is larger than half of the
     * minimal box dimension
     */
    void SimulationBoxSetup::checkRcCutoff()
    {
        const auto &simBox = _engine.getSimulationBox();
        const auto  coulombCutoff =
            settings::PotentialSettings::getCoulombRadiusCutOff();
        const auto minDim = simBox.getMinimalBoxDimension();

        if (coulombCutoff > minDim / 2.0)
        {
            throw exc::InputFileException(
                std::format(
                    "Rc cutoff is larger than half of the minimal box "
                    "dimension of "
                    "{} Angstrom.",
                    minDim
                )
            );
        }
    }

    /**
     * @brief Check if all velocities of the simulation box are 0
     *
     * @details If all velocities of the simulation box are 0, _zeroVelocities
     * is set to true
     */
    void SimulationBoxSetup::checkZeroVelocities()
    {
        constexpr double epsilon    = 1e-10;
        auto            &simBox     = _engine.getSimulationBox();
        const auto       velocities = simBox.getVelocities();

        if ((std::ranges::any_of(
                velocities,
                [epsilon](const auto &vel)
                {
                    return std::abs(vel[0]) > epsilon ||
                           std::abs(vel[1]) > epsilon ||
                           std::abs(vel[2]) > epsilon;
                }
            )))
            setZeroVelocities(false);
        else
            setZeroVelocities(true);
    }

    /**
     * @brief Initialize the velocities of the simulation box
     *
     * @details If initializeVelocities is set, the velocities are initialized
     * with a Maxwell-Boltzmann distribution.
     */
    void SimulationBoxSetup::initVelocities()
    {
        if (settings::SimulationBoxSettings::getInitializeVelocities() ==
                InitVelocities::FALSE ||
            (settings::SimulationBoxSettings::getInitializeVelocities() ==
                 InitVelocities::TRUE &&
             !getZeroVelocities()))
            return;

        maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
        maxwellBoltzmann.initializeVelocities(_engine.getSimulationBox());
    }

    /**
     * @brief write setup info to log file
     *
     */
    void SimulationBoxSetup::writeSetupInfo() const
    {
        auto &log    = _engine.getLogOutput();
        auto &simBox = _engine.getSimulationBox();

        const auto nAtoms = simBox.getNumberOfAtoms();
        const auto mass   = simBox.getTotalMass();
        const auto charge = simBox.getTotalCharge();
        const auto dof    = simBox.getDegreesOfFreedom();

        log.writeSetupInfo(std::format("number of atoms:   {:8d}", nAtoms));
        log.writeSetupInfo(
            std::format("total mass:        {:14.5f} g/mol", mass)
        );
        log.writeSetupInfo(std::format("total charge:      {:14.5f}", charge));
        log.writeSetupInfo(std::format("unconstrained DOF: {:8d}", dof));
        log.writeEmptyLine();

        const auto density = simBox.getDensity();
        const auto volume  = simBox.getVolume();
        const auto volumeStr =
            std::format("{:14.5f} {}³", volume, out::ANGSTROM);

        log.writeSetupInfo(
            std::format("density:         {:14.5f} kg/L", density)
        );
        log.writeSetupInfo(std::format("volume:          {}", volumeStr));
        log.writeEmptyLine();

        const auto boxA = simBox.getBoxDimensions()[0];
        const auto boxB = simBox.getBoxDimensions()[1];
        const auto boxC = simBox.getBoxDimensions()[2];

        const auto boxAStr = std::format("{:14.5f} {}", boxA, out::ANGSTROM);
        const auto boxBStr = std::format("{:14.5f} {}", boxB, out::ANGSTROM);
        const auto boxCstr = std::format("{:14.5f} {}", boxC, out::ANGSTROM);

        const auto alpha = simBox.getBoxAngles()[0];
        const auto beta  = simBox.getBoxAngles()[1];
        const auto gamma = simBox.getBoxAngles()[2];

        const auto alphaStr = std::format("{:14.5f}°", alpha);
        const auto betaStr  = std::format("{:14.5f}°", beta);
        const auto gammaStr = std::format("{:14.5f}°", gamma);

        // clang-format off
    log.writeSetupInfo(std::format("box dimensions:  {} {} {}", boxAStr, boxBStr, boxCstr));
    log.writeSetupInfo(std::format("box angles:      {}  {}  {}", alphaStr, betaStr, gammaStr));
    log.writeEmptyLine();
        // clang-format on

        const auto coulombCutoff =
            settings::PotentialSettings::getCoulombRadiusCutOff();
        const auto coulombCutoffStr =
            std::format("{:14.5f} {}", coulombCutoff, out::ANGSTROM);

        log.writeSetupInfo(
            std::format("coulomb cutoff:  {}", coulombCutoffStr)
        );
        log.writeEmptyLine();

        if (settings::SimulationBoxSettings::getInitializeVelocities() ==
                InitVelocities::TRUE &&
            !getZeroVelocities())
        {
            log.writeSetupWarning(
                std::format(
                    "Ignoring 'init_velocities' because non-zero velocities in "
                    "\"{}\"",
                    settings::FileSettings::getStartFileName()
                )
            );
            out::StdoutOutput::writeSetupWarning(
                std::format(
                    "Ignoring 'init_velocities' because non-zero velocities in "
                    "\"{}\"",
                    settings::FileSettings::getStartFileName()
                )
            );
        }

        if ((settings::SimulationBoxSettings::getInitializeVelocities() ==
                 InitVelocities::TRUE &&
             getZeroVelocities()) ||
            settings::SimulationBoxSettings::getInitializeVelocities() ==
                InitVelocities::FORCE)
        {
            log.writeSetupInfo(
                "velocities initialized with Maxwell-Boltzmann distribution"
            );
        }
        else
        {
            log.writeSetupInfo(
                std::format(
                    "velocities taken from start file \"{}\"",
                    settings::FileSettings::getStartFileName()
                )
            );
        }
        log.writeEmptyLine();
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief sets if the velocities in the start file are zero
     *
     */
    void SimulationBoxSetup::setZeroVelocities(bool zeroVelocities)
    {
        _zeroVelocities = zeroVelocities;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief returns if the velocities in the start file are zero
     *
     * @return bool
     */
    bool SimulationBoxSetup::getZeroVelocities() { return _zeroVelocities; }

}   // namespace setup
