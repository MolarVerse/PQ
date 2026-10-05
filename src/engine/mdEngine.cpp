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

#include "constants/conversionFactors.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "outputFileSettings.hpp"
#include "progressbar.hpp"
#include "referencesOutput.hpp"
#include "timingsSettings.hpp"
#include "velocityVerlet.hpp"

namespace engine
{

    /**
     * @brief Constructor for MDEngine
     *
     * @details This constructor initializes the MDEngine with default settings.
     */
    MDEngine::MDEngine()
        : _integrator(std::make_unique<integrator::VelocityVerlet>()),
          _thermostat(std::make_unique<thermostat::Thermostat>()),
          _manostat(std::make_unique<manostat::Manostat>())
    {
    }

    /**
     * @brief Run the simulation for numberOfSteps steps.
     *
     */
    void MDEngine::run()
    {
        getPhysicalDataPtr()->calculateKinetics(getSimulationBox());

        getEngineOutput().getLogOutput().writeInitialMomentum(
            norm(getPhysicalDataPtr()->getMomentum())
        );

        _setNSteps(settings::TimingsSettings::getNumberOfSteps());
        progressbar bar(static_cast<int>(_getNSteps()), true, std::cout);

        for (; getStep() <= _getNSteps(); _setStep(getStep() + 1))
        {
            bar.update();
            takeStep();

            writeOutput();
            deleteTmpFiles();
        }

        timings::GlobalTimer::get().stopSimulationTimer();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime() * MS_TO_S;

        references::ReferencesOutput::writeReferencesFile();

        getEngineOutput().writeTimingsFile();

        getEngineOutput().getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    /**
     * @brief MD Loop before force calculation.
     *
     */
    void MDEngine::takeStepBeforeForces()
    {
        _thermostat->applyThermostatHalfStep(
            getSimulationBox(),
            getPhysicalData()
        );

        if (getConstraints()->isMShakeActive())
            getSimulationBox().updateOldPositions();

        _integrator->firstStep(getSimulationBox());

        getConstraints()->applyShake(getSimulationBox());
    }

    /**
     * @brief MD Loop after force calculation.
     *
     */
    void MDEngine::takeStepAfterForces()
    {
        _resetKinetics.resetForces(getStep(), getSimulationBox());

        getConstraints()->applyDistanceConstraints(
            getSimulationBox(),
            getPhysicalData(),
            calculateTotalSimulationTime()
        );

        getConstraints()->calculateConstraintBondRefs(getSimulationBox());

        if (!settings::GeneralSettings::isHybridJobtype())
        {
            const auto virial =
                virial::intraMolecularVirialCorrection(getSimulationBox());
            getPhysicalDataPtr()->addVirial(virial);
        }

        _thermostat->applyThermostatOnForces(getSimulationBox());

        _integrator->secondStep(getSimulationBox());

        getConstraints()->applyRattle(getSimulationBox());

        _thermostat->applyThermostat(getSimulationBox(), getPhysicalData());

        getPhysicalDataPtr()->calculateKinetics(getSimulationBox());

        _manostat->applyManostat(getSimulationBox(), getPhysicalData());

        _resetKinetics.reset(getStep(), getPhysicalData(), getSimulationBox());

        _thermostat->applyTemperatureRamping();

        if (settings::GeneralSettings::isQMOnlyJobtype())
        {
            const auto nQMAtoms = getSimulationBox().getNumberOfQMAtoms();
            getPhysicalDataPtr()->setNumberOfQMAtoms(
                static_cast<double>(nQMAtoms)
            );
        }
    }

    void MDEngine::calculateForcesWrapper()
    {
        getSimulationBox().resetAllForces();
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
     * @brief Writes output files.
     *
     * @details output files are written if the step is a multiple of the output
     * frequency.
     *
     */
    void MDEngine::writeOutput()
    {
        const auto outputFreq =
            settings::OutputFileSettings::getOutputFrequency();
        const auto step0   = settings::TimingsSettings::getStepCount();
        const auto effStep = getStep() + step0;

        if (0 == getStep() % outputFreq)
        {
            getEngineOutput().writeXyzFile(getSimulationBox(), effStep);
            getEngineOutput().writeVelFile(getSimulationBox(), effStep);
            getEngineOutput().writeForceFile(getSimulationBox(), effStep);
            getEngineOutput().writeChargeFile(getSimulationBox(), effStep);
            getEngineOutput()
                .writeRstFile(getSimulationBox(), *_thermostat, effStep);

            getEngineOutput().writeVirialFile(
                effStep,
                getPhysicalData()
            );   // use physicalData instead of averagePhysicalData

            getEngineOutput().writeStressFile(
                effStep,
                getPhysicalData()
            );   // use physicalData instead of averagePhysicalData

            getEngineOutput().writeBoxFile(
                effStep,
                getSimulationBox().getBox()
            );

            if (settings::GeneralSettings::isHybridJobtype())
                getEngineOutput().writeHybridCenterXyzFile(
                    _configurator,
                    effStep
                );
        }

        // NOTE:
        // stop and restart immediately time manager - maximum lost time is en
        // file writing in last step of simulation but on the other hand setup
        // is now included in total simulation time Unfortunately, setup is
        // therefore included in the first looptime output but this is not a big
        // problem - could also be a feature and not a bug
        timings::GlobalTimer::get().stopAndRestartSimulationTimer();

        getPhysicalDataPtr()->setLoopTime(
            timings::GlobalTimer::get().calculateLoopTime()
        );
        getAveragePhysicalData().updateAverages(getPhysicalData());

        if (0 == getStep() % outputFreq)
        {
            getAveragePhysicalData().makeAverages(
                static_cast<double>(outputFreq)
            );

            const auto timeStep      = settings::TimingsSettings::getTimeStep();
            const auto effStepDouble = static_cast<double>(effStep);
            const auto simTime       = effStepDouble * timeStep * FS_TO_PS;

            getEngineOutput().writeEnergyFile(
                effStep,
                getAveragePhysicalData()
            );
            getEngineOutput().writeInstantEnergyFile(
                effStep,
                getPhysicalData()
            );
            getEngineOutput().writeInfoFile(simTime, getAveragePhysicalData());
            getEngineOutput().writeMomentumFile(
                effStep,
                getAveragePhysicalData()
            );

            getAveragePhysicalData() = physicalData::PhysicalData();
        }

        getPhysicalDataPtr()->reset();
    }

    /**
     * @brief get the reference to the reset kinetics
     *
     * @return resetKinetics::ResetKinetics&
     */
    resetKinetics::ResetKinetics &MDEngine::getResetKinetics()
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

}   // namespace engine
