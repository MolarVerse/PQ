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

#include "mdEngine.hpp"

#include "constraintSettings.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "hybridConfigurator.hpp"
#include "interrupt.hpp"
#include "manostatFactory.hpp"
#include "outputFileSettings.hpp"
#include "terminalProgressBar.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"
#include "velocityVerlet.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

namespace engine
{

    /**
     * @brief Constructor for MDEngine
     *
     * @details This constructor initializes the MDEngine with default settings.
     */
    MDEngine::MDEngine(const Settings &settings)
        : Engine(settings),
          _resetKinetics(settings.resetKinetics),
          _kinetics(settings.kinetics),
          _configurator(settings.hybridSettings),
          _integrator(std::make_unique<integrator::VelocityVerlet>()),
          _thermostat(std::make_unique<thermostat::Thermostat>()),
          _manostat(
              manostat::ManostatFactory::createManostat(
                  settings.manostatSettings
              )
          )
    {
    }

    /**
     * @brief Run the simulation for numberOfSteps steps.
     *
     */
    void MDEngine::run()
    {
        get_physicalData().calculateKinetics(get_simulationBox());

        getEngineOutput().getLogOutput().writeInitialMomentum(
            norm(get_physicalData().get_momentum())
        );

        const auto nSteps = getSettings().timings.getNumberOfSteps();
        utilities::TerminalProgressBar bar(static_cast<int>(nSteps));
        const auto &outputFileSettings = getSettings().outputFiles;
        const auto  jobType = getSettings().generalSettings.getJobtype();

        for (; getStep() <= nSteps; _setStep(getStep() + 1))
        {
            utilities::throwIfInterrupted();

            bar.update();
            takeStep();

            writeOutput(jobType, outputFileSettings);
            deleteTmpFiles();
        }

        getEngineOutput().waitForFrames();

        timings::GlobalTimer::get().stopSimulationTimer();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime();

#ifdef WITH_MPI
        if (mpi::MPI::isRoot())
#endif
        {
            getEngineOutput().writeReferencesFile();

            getEngineOutput().writeTimingsFile();
        }

        getEngineOutput().getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    /**
     * @brief MD Loop before force calculation.
     *
     */
    void MDEngine::takeStepBeforeForces()
    {
        const auto &timeStep = getSettings().timings.getTimeStep();
        _thermostat->applyThermostatHalfStep(
            get_simulationBox(),
            get_physicalData(),
            _kinetics.getCurrentTargetTemperature(),
            timeStep
        );

        if (getSettings().constraintSettings.isMShakeActivated())
            get_simulationBox().updateOldPositions();

        getIntegrator().firstStep(get_simulationBox(), timeStep);

        get_constraints().applyShake(get_simulationBox(), timeStep);
    }

    /**
     * @brief MD Loop after force calculation.
     *
     */
    void MDEngine::takeStepAfterForces()
    {
        _resetKinetics.resetForces(getStep(), get_simulationBox());

        const auto &timingsSettings = getSettings().timings;
        const auto &timeStep        = timingsSettings.getTimeStep();

        get_constraints().applyDistanceConstraints(
            get_simulationBox(),
            get_physicalData(),
            calculateTotalSimulationTime(
                timeStep,
                timingsSettings.getStartingTime()
            )
        );

        get_constraints().calculateConstraintBondRefs(get_simulationBox());

        const auto jobType = getSettings().generalSettings.getJobtype();
        if (!isHybridJobtype(jobType))
        {
            const auto virial =
                virial::intraMolecularVirialCorrection(get_simulationBox());
            get_physicalData().addVirial(virial);
        }

        _thermostat->applyThermostatOnForces(
            get_simulationBox(),
            _kinetics.getCurrentTargetTemperature()
        );

        _integrator->secondStep(get_simulationBox(), timeStep);

        get_constraints().applyRattle(get_simulationBox());

        _thermostat->applyThermostat(
            get_simulationBox(),
            get_physicalData(),
            _kinetics.getCurrentTargetTemperature(),
            timeStep
        );

        get_physicalData().calculateKinetics(get_simulationBox());

        const auto &currentTargetTemperature =
            _kinetics.getCurrentTargetTemperature();

        _manostat->applyManostat(
            get_simulationBox(),
            get_physicalData(),
            currentTargetTemperature,
            timeStep,
            getSettings().potentialSettings.getCoulombRadiusCutoff()
        );

        const auto driftWarning = _resetKinetics.checkMomentumDrift(
            getStep(),
            get_physicalData(),
            get_simulationBox()
        );

        if (driftWarning)
        {
            getEngineOutput().getLogOutput().writeSetupWarning(*driftWarning);
            out::StdoutOutput::writeSetupWarning(*driftWarning);
        }

        _resetKinetics.reset(
            getStep(),
            get_physicalData(),
            get_simulationBox(),
            currentTargetTemperature
        );

        _kinetics.applyTemperatureRamping();

        if (isQMOnlyJobtype(jobType))
        {
            const auto nQMAtoms = get_simulationBox().getNumberOfActiveAtoms();
            get_physicalData().set_nQMAtoms(static_cast<double>(nQMAtoms));
        }
    }

    void MDEngine::calculateForcesWrapper()
    {
        get_simulationBox().resetAllForces();
        calculateForces();
    }

    /**
     * @brief Takes one step in the simulation.
     *
     */
    void MDEngine::takeStep()
    {
        takeStepBeforeForces();

        calculateForcesWrapper();

        takeStepAfterForces();
    }

    /**
     * @brief Writes output files using the default time step.
     *
     */
    void MDEngine::writeOutput(
        JobType                   jobType,
        const OutputFileSettings &outputSettings
    )
    {
        writeOutput(
            jobType,
            outputSettings,
            getSettings().timings.getTimeStep(),
            getSettings().timings.getStartingStepCount()
        );
    }

    /**
     * @brief Writes output files.
     *
     * @details output files are written if the step is a multiple of the output
     * frequency.
     *
     */
    void MDEngine::writeOutput(
        JobType                   jobType,
        const OutputFileSettings &outputSettings,
        const Time               &timeStep,
        size_t                    startingStepCount
    )
    {
        const auto outputFreq = outputSettings.getOutputFrequency();
        const auto effStep    = getStep() + startingStepCount;

        if (0 == getStep() % outputFreq)
        {
            getEngineOutput().writeFrame(
                get_simulationBox(),
                *_thermostat,
                effStep,
                get_physicalData().getPotentialEnergy().numerical_value_in(
                    kcal_per_mol
                ),
                calculateTotalSimulationTime(
                    timeStep,
                    getSettings().timings.getStartingTime()
                )
            );

            getEngineOutput().writeVirialFile(
                effStep,
                get_physicalData()
            );   // use physicalData instead of averagePhysicalData

            getEngineOutput().writeStressFile(
                effStep,
                get_physicalData()
            );   // use physicalData instead of averagePhysicalData

            getEngineOutput().writeBoxFile(
                effStep,
                get_simulationBox().getBox()
            );

            if (isHybridJobtype(jobType))
            {
                getEngineOutput().writeHybridCenterXyzFile(
                    _configurator,
                    effStep
                );
            }
        }

        // NOTE:
        // stop and restart immediately time manager - maximum lost time is en
        // file writing in last step of simulation but on the other hand setup
        // is now included in total simulation time Unfortunately, setup is
        // therefore included in the first looptime output but this is not a big
        // problem - could also be a feature and not a bug
        timings::GlobalTimer::get().stopAndRestartSimulationTimer();

        get_physicalData().set_loopTime(
            timings::GlobalTimer::get().calculateLoopTime()
        );
        get_averagePhysicalData().updateAverages(get_physicalData());

        if (0 == getStep() % outputFreq)
        {
            get_averagePhysicalData().makeAverages(
                static_cast<double>(outputFreq)
            );

            const auto simTime = calculateTotalSimulationTime(
                timeStep,
                getSettings().timings.getStartingTime()
            );

            getEngineOutput().writeEnergyFile(
                effStep,
                get_averagePhysicalData(),
                getSettings()
            );
            getEngineOutput().writeInstantEnergyFile(
                effStep,
                get_physicalData(),
                getSettings()
            );
            getEngineOutput().getInfoOutput().write(
                simTime,
                get_averagePhysicalData(),
                getSettings()
            );
            getEngineOutput().writeMomentumFile(
                effStep,
                get_averagePhysicalData()
            );

            get_averagePhysicalData() = physicalData::PhysicalData();
        }

        get_physicalData().reset();
    }

    /**
     * @brief get the reference to the reset kinetics
     *
     * @return kinetics::ResetKinetics&
     */
    kinetics::ResetKinetics &MDEngine::getResetKinetics()
    {
        return _resetKinetics;
    }

    /**
     * @brief get the reference to the integrator
     *
     * @return integrator::Integrator&
     */
    integrator::Integrator &MDEngine::getIntegrator() { return *_integrator; }

    /**
     * @brief get the reference to the thermostat
     *
     * @return thermostat::Thermostat&
     */
    thermostat::Thermostat &MDEngine::getThermostat() { return *_thermostat; }

    /**
     * @brief get the reference to the manostat
     *
     * @return manostat::Manostat&
     */
    manostat::Manostat &MDEngine::getManostat() { return *_manostat; }

    /**
     * @brief get the reference to the instant energy output
     *
     * @return out::EnergyOutput&
     */
    out::EnergyOutput &MDEngine::getInstantEnergyOutput()
    {
        return getEngineOutput().getInstantEnergyOutput();
    }

    /**
     * @brief get the reference to the momentum output
     *
     * @return out::MomentumOutput&
     */
    out::MomentumOutput &MDEngine::getMomentumOutput()
    {
        return getEngineOutput().getMomentumOutput();
    }

    /**
     * @brief get the reference to the xyz hybrid center output
     *
     * @return out::TrajectoryOutput&
     */
    out::TrajectoryOutput &MDEngine::getXyzHybridCenterOutput()
    {
        return getEngineOutput().getXyzHybridCenterOutput();
    }

    /**
     * @brief get the reference to the vel output
     *
     * @return out::TrajectoryOutput&
     */
    out::TrajectoryOutput &MDEngine::getVelOutput()
    {
        return getEngineOutput().getVelOutput();
    }

    /**
     * @brief get the reference to the charge output
     *
     * @return out::TrajectoryOutput&
     */
    out::TrajectoryOutput &MDEngine::getChargeOutput()
    {
        return getEngineOutput().getChargeOutput();
    }

    /**
     * @brief get the reference to the virial output
     *
     * @return out::VirialOutput&
     */
    out::VirialOutput &MDEngine::getVirialOutput()
    {
        return getEngineOutput().getVirialOutput();
    }

    /**
     * @brief get the reference to the stress output
     *
     * @return out::StressOutput&
     */
    out::StressOutput &MDEngine::getStressOutput()
    {
        return getEngineOutput().getStressOutput();
    }

    /**
     * @brief get the reference to the box file output
     *
     * @return out::BoxFileOutput&
     */
    out::BoxFileOutput &MDEngine::getBoxFileOutput()
    {
        return getEngineOutput().getBoxFileOutput();
    }

    /**
     * @brief get the reference to the ring polymer rst file output
     *
     * @return out::RingPolymerRestartFileOutput&
     */
    out::RingPolymerRestartFileOutput &MDEngine::getRingPolymerRstFileOutput()
    {
        return getEngineOutput().getRingPolymerRstFileOutput();
    }

    /**
     * @brief get the reference to the ring polymer xyz output
     *
     * @return out::RingPolymerTrajectoryOutput&
     */
    out::RingPolymerTrajectoryOutput &MDEngine::getRingPolymerXyzOutput()
    {
        return getEngineOutput().getRingPolymerXyzOutput();
    }

    /**
     * @brief get the reference to the ring polymer vel output
     *
     * @return out::RingPolymerTrajectoryOutput&
     */
    out::RingPolymerTrajectoryOutput &MDEngine::getRingPolymerVelOutput()
    {
        return getEngineOutput().getRingPolymerVelOutput();
    }

    /**
     * @brief get the reference to the ring polymer force output
     *
     * @return out::RingPolymerTrajectoryOutput&
     */
    out::RingPolymerTrajectoryOutput &MDEngine::getRingPolymerForceOutput()
    {
        return getEngineOutput().getRingPolymerForceOutput();
    }

    /**
     * @brief get the reference to the ring polymer charge output
     *
     * @return out::RingPolymerTrajectoryOutput&
     */
    out::RingPolymerTrajectoryOutput &MDEngine::getRingPolymerChargeOutput()
    {
        return getEngineOutput().getRingPolymerChargeOutput();
    }

    /**
     * @brief get the reference to the ring polymer energy output
     *
     * @return out::RingPolymerEnergyOutput&
     */
    out::RingPolymerEnergyOutput &MDEngine::getRingPolymerEnergyOutput()
    {
        return getEngineOutput().getRingPolymerEnergyOutput();
    }

    /**
     * @brief get the reference to the hybrid configurator
     *
     * @return configurator::HybridConfigurator&
     */
    configurator::HybridConfigurator &MDEngine::_getHybridConfigurator()
    {
        return _configurator;
    }

    /**
     * @brief get the reference to the hybrid configurator (const)
     *
     * @return const configurator::HybridConfigurator&
     */
    const configurator::HybridConfigurator &MDEngine::_getHybridConfigurator(
    ) const
    {
        return _configurator;
    }

    /**
     * @brief get the reference to the kinetics (const)
     *
     * @return const kinetics::Kinetics&
     */
    const kinetics::Kinetics &MDEngine::_getKinetics() const
    {
        return _kinetics;
    }

    /**
     * @brief get the reference to the kinetics
     *
     * @return kinetics::Kinetics&
     */
    kinetics::Kinetics &MDEngine::_getKinetics() { return _kinetics; }
}   // namespace engine
