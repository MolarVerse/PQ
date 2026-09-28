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

#include <gtest/gtest.h>

#include <limits>   // for numeric_limits
#include <memory>   // for make_unique, unique_ptr
#include <string>   // for string

#include "defaults.hpp"                // for default settings
#include "exceptions.hpp"              // for exc::InputFileException
#include "hessianSettings.hpp"         // for settings::HessianSettings
#include "inputFileReader.hpp"         // for InputFileReader
#include "manostatSettings.hpp"        // for settings::ManostatSettings
#include "optEngine.hpp"               // for OptEngine
#include "optimizerSettings.hpp"       // for settings::OptimizerSettings
#include "potentialSettings.hpp"       // for settings::PotentialSettings
#include "qmSettings.hpp"              // for settings::QMSettings
#include "settings.hpp"                // for Settings
#include "simulationBoxSettings.hpp"   // for settings::SimulationBoxSettings
#include "thermostatSettings.hpp"      // for settings::ThermostatSettings
#include "throwWithMessage.hpp"        // for ASSERT_THROW_MSG
#include "timingsSettings.hpp"         // for settings::TimingsSettings

class TestInputValidation : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        settings::Settings::setJobtype(settings::JobType::NONE);
        settings::HessianSettings::setOptimizeBeforeHessian(false);

        settings::OptimizerSettings::setLearningRateStrategy(
            LearningRate::CONSTANT
        );
        settings::OptimizerSettings::setMinLearningRate(1.0e-15);
        settings::OptimizerSettings::setMaxLearningRate(1.0);

        settings::ManostatSettings::setManostatType(ManostatType::NONE);
        settings::ManostatSettings::setTauManostat(
            defaults::BERENDSEN_MANOSTAT_RELAX_TIME
        );

        settings::ThermostatSettings::setThermostatType(
            settings::ThermostatType::NONE
        );
        settings::ThermostatSettings::setTargetTemperature(0.0);
        settings::ThermostatSettings::setStartTemperature(0.0);
        settings::ThermostatSettings::setEndTemperature(0.0);
        settings::ThermostatSettings::setTemperatureSet(false);
        settings::ThermostatSettings::setStartTemperatureSet(false);
        settings::ThermostatSettings::setEndTemperatureSet(false);
        settings::ThermostatSettings::setTemperatureRampSteps(0);
        settings::ThermostatSettings::setTemperatureRampFrequency(1);
        settings::ThermostatSettings::setRelaxationTime(
            defaults::BERENDSEN_THERMOSTAT_RELAX_TIME
        );
        settings::ThermostatSettings::setFriction(
            defaults::LANGEVIN_THERMOSTAT_FRICTION
        );
        settings::SimulationBoxSettings::setInitializeVelocities(
            settings::InitVelocities::FALSE
        );

        settings::PotentialSettings::setCoulombLongRangeType(
            CoulombLongRangeType::SHIFTED
        );
        settings::PotentialSettings::setCoulombRadiusCutOff(
            defaults::COULOMB_CUT_OFF_DEFAULT
        );
        settings::TimingsSettings::setTimeStep(0.5);

        settings::QMSettings::setQMMethod(settings::QMMethod::NONE);
        settings::QMSettings::setMaceModel(settings::MaceModel::MEDIUM);
        settings::QMSettings::setMaceModelType(
            settings::MaceModelType::MACE_MP
        );
        settings::QMSettings::setMaceModelPath("");
        settings::QMSettings::setSlakosType(settings::SlakosType::NONE);
        settings::QMSettings::setUseThirdOrderDftb(false);
        settings::QMSettings::setIsThirdOrderDftbSet(false);
        settings::QMSettings::setIsHubbardDerivsSet(false);
        settings::QMSettings::setFennolModelPath("");

        _engine = std::make_unique<engine::OptEngine>();
        _reader =
            std::make_unique<input::InputFileReader>("input.in", *_engine);
    }

    void setKeyword(const std::string &keyword)
    {
        _reader->setKeywordCount(keyword, 1);
    }

    void configureMDJob(const settings::JobType jobType)
    {
        settings::Settings::setJobtype(jobType);
        settings::TimingsSettings::setNumberOfSteps(100);
        setKeyword("nstep");
        setKeyword("timestep");
        if (jobType == settings::JobType::QM_MD ||
            jobType == settings::JobType::RING_POLYMER_QM_MD)
            setKeyword("qm_prog");
    }

    void TearDown() override { settings::Settings::deactivateCellList(); }

    std::unique_ptr<engine::OptEngine>      _engine;
    std::unique_ptr<input::InputFileReader> _reader;
};

TEST_F(TestInputValidation, requiresNumberOfStepsForMD)
{
    settings::Settings::setJobtype(settings::JobType::MM_MD);
    setKeyword("timestep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "Job type MM_MD selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, requiresNumberOfStepsForOptimization)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "Job type MM_OPT selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, requiresNumberOfStepsForPreoptimizedHessian)
{
    settings::Settings::setJobtype(settings::JobType::MM_HESSIAN);
    settings::HessianSettings::setOptimizeBeforeHessian(true);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "Job type MM_HESSIAN selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, hessianWithoutOptimizationNeedsNoTimings)
{
    settings::Settings::setJobtype(settings::JobType::MM_HESSIAN);

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresTimeStepForMD)
{
    settings::Settings::setJobtype(settings::JobType::MM_MD);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "Molecular Dynamics job type MM_MD selected. Please set the time step "
        "in the input file."
    );
}

TEST_F(TestInputValidation, requiresPressureForManostat)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Pressure not set for BERENDSEN manostat"
    );
}

TEST_F(TestInputValidation, rejectsUnstableManostatRelaxationTime)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setTauManostat(0.0001);
    setKeyword("pressure");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "The timestep must not exceed the manostat relaxation time"
    );
}

TEST_F(TestInputValidation, requiresQMProgramForQMJob)
{
    settings::Settings::setJobtype(settings::JobType::QM_MD);
    setKeyword("nstep");
    setKeyword("timestep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "QM job selected but the \"qm_prog\" keyword has not been set"
    );
}

TEST_F(TestInputValidation, requiresTemperatureForThermostat)
{
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Target or end temperature not set for berendsen thermostat"
    );
}

TEST_F(TestInputValidation, rejectsBothThermostatTemperatures)
{
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    setKeyword("temp");
    setKeyword("end_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Both target and end temperature set for berendsen thermostat. They "
        "are mutually exclusive as they are treated as synonyms"
    );
}

TEST_F(TestInputValidation, acceptsEndTemperatureForThermostat)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    settings::ThermostatSettings::setEndTemperature(300.0);
    setKeyword("end_temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
    EXPECT_DOUBLE_EQ(settings::ThermostatSettings::getTargetTemperature(), 0.0);
    EXPECT_DOUBLE_EQ(
        settings::ThermostatSettings::getActualTargetTemperature(),
        0.0
    );
}

TEST_F(TestInputValidation, requiresTemperatureForVelocityInitialization)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::SimulationBoxSettings::setInitializeVelocities(
        settings::InitVelocities::FORCE
    );

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Initializing velocities requires temp, start_temp, or end_temp"
    );
}

TEST_F(TestInputValidation, rejectsUnstableThermostatRelaxationTime)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::VELOCITY_RESCALING
    );
    settings::ThermostatSettings::setTargetTemperature(300.0);
    settings::ThermostatSettings::setRelaxationTime(0.0001);
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "The timestep must not exceed the thermostat relaxation time"
    );
}

TEST_F(TestInputValidation, rejectsNonFiniteLangevinScale)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::LANGEVIN
    );
    settings::ThermostatSettings::setTargetTemperature(300.0);
    settings::ThermostatSettings::setFriction(
        std::numeric_limits<double>::max()
    );
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Langevin thermostat parameters produce a non-finite random-force "
        "scale"
    );
}

TEST_F(TestInputValidation, rejectsNonFiniteLangevinRampScale)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::LANGEVIN
    );
    settings::ThermostatSettings::setTargetTemperature(300.0);
    settings::ThermostatSettings::setStartTemperature(
        std::numeric_limits<double>::max()
    );
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Langevin thermostat parameters produce a non-finite random-force "
        "scale"
    );
}

TEST_F(TestInputValidation, rejectsCellListWithoutCoulombCutoff)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::Settings::activateCellList();
    settings::PotentialSettings::setCoulombRadiusCutOff(0.0);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "An active cell list requires rcoulomb to be greater than zero"
    );
}

TEST_F(TestInputValidation, rejectsCellListForPureQM)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    settings::Settings::activateCellList();

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Cell lists are not available for pure QM simulations"
    );
}

TEST_F(TestInputValidation, rejectsZeroTemperatureForNoseHoover)
{
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::NOSE_HOOVER
    );
    settings::ThermostatSettings::setTargetTemperature(0.0);
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Nose-Hoover target temperature must be greater than zero"
    );
}

TEST_F(TestInputValidation, acceptsZeroTemperatureForBerendsen)
{
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    settings::ThermostatSettings::setTargetTemperature(0.0);
    setKeyword("temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsTemperatureRampLongerThanSimulation)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    settings::ThermostatSettings::setTemperatureRampSteps(200);
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Number of total simulation steps 100 is smaller than the number of "
        "temperature ramping steps 200"
    );
}

TEST_F(TestInputValidation, rejectsTemperatureRampFrequencyAboveRampSteps)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    settings::ThermostatSettings::setTemperatureRampSteps(2);
    settings::ThermostatSettings::setTemperatureRampFrequency(4);
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Temperature ramp frequency 4 is larger than the number of ramping "
        "steps 2"
    );
}

TEST_F(TestInputValidation, acceptsDefaultTemperatureRampLength)
{
    configureMDJob(settings::JobType::MM_MD);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    settings::ThermostatSettings::setTemperatureRampFrequency(100);
    setKeyword("temp");
    setKeyword("start_temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresReplicaCountForRingPolymer)
{
    configureMDJob(settings::JobType::RING_POLYMER_QM_MD);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Number of beads not set for ring polymer simulation"
    );
}

TEST_F(TestInputValidation, acceptsReplicaCountForRingPolymer)
{
    configureMDJob(settings::JobType::RING_POLYMER_QM_MD);
    setKeyword("rpmd_n_replica");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresSlaterKosterSetForAseDftbPlus)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::NONE);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "ASE-DFTB+ requires slakos to be 3ob, matsci, or custom"
    );
}

TEST_F(TestInputValidation, requiresPathForCustomSlaterKosterParameters)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::CUSTOM);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "Custom Slater-Koster parameters require the \"slakos_path\" keyword"
    );
}

TEST_F(TestInputValidation, rejectsHubbardDerivativesWithoutThirdOrder)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::CUSTOM);
    settings::QMSettings::setUseThirdOrderDftb(false);
    setKeyword("slakos_path");
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "You have set custom Hubbard derivatives but disabled 3rd order DFTB. "
        "This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsHubbardDerivativesWithThirdOrder)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::CUSTOM);
    settings::QMSettings::setUseThirdOrderDftb(true);
    setKeyword("slakos_path");
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

#ifdef WITH_ASE
TEST_F(TestInputValidation, rejectsExplicitlyDisabledThreeObThirdOrder)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::THREEOB);
    settings::QMSettings::setUseThirdOrderDftb(false);
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "You have set custom Hubbard derivatives but disabled 3rd order DFTB. "
        "This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsThreeObDefaultThirdOrder)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::ASEDFTBPLUS);
    settings::QMSettings::setSlakosType(settings::SlakosType::THREEOB);
    settings::QMSettings::setUseThirdOrderDftb(false);
    setKeyword("hubbard_derivs");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}
#endif

TEST_F(TestInputValidation, requiresFennolModelPath)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::FENNOL);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "The FeNNol QM runner has been selected but the "
        "\"fennol_model_path\" keyword has not been set. This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsFennolModelPath)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::FENNOL);
    setKeyword("fennol_model_path");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsMaceModelForWrongModelType)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    settings::QMSettings::setMaceModelType(settings::MaceModelType::MACE_OFF);
    settings::QMSettings::setMaceModel(settings::MaceModel::MEDIUMOMAT0);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "The 'medium-omat-0' model size is only compatible with the 'mace_mp' "
        "model type."
    );
}

TEST_F(TestInputValidation, acceptsStandardMaceModelForNonMpType)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    settings::QMSettings::setMaceModelType(settings::MaceModelType::MACE_OFF);
    settings::QMSettings::setMaceModel(settings::MaceModel::SMALL);

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresPathForCustomMaceModel)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    settings::QMSettings::setMaceModel(settings::MaceModel::CUSTOM);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "You have requested a custom MACE model but haven't provided a MACE "
        "model path.This setup is invalid."
    );
}

TEST_F(TestInputValidation, rejectsPathForBundledMaceModel)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    settings::QMSettings::setMaceModel(settings::MaceModel::MEDIUMOMAT0);
    setKeyword("mace_model_path");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::InputFileException,
        "You have set a custom MACE model path without requesting a custom "
        "mace model size.This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsValidConditionalKeywords)
{
    configureMDJob(settings::JobType::QM_MD);
    settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    settings::QMSettings::setMaceModel(settings::MaceModel::CUSTOM);
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ThermostatSettings::setThermostatType(
        settings::ThermostatType::BERENDSEN
    );
    setKeyword("mace_model_path");
    setKeyword("pressure");
    setKeyword("temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresDecayForConstantDecayOptimization)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    settings::OptimizerSettings::setLearningRateStrategy(
        LearningRate::CONSTANT_DECAY
    );
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "The constant-decay learning rate strategy requires "
        "learning-rate-decay."
    );
}

TEST_F(TestInputValidation, requiresDecayForExponentialDecayOptimization)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    settings::OptimizerSettings::setLearningRateStrategy(
        LearningRate::EXPONENTIAL_DECAY
    );
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "The exponential-decay learning rate strategy requires "
        "learning-rate-decay."
    );
}

TEST_F(TestInputValidation, acceptsConstantOptimizationWithoutDecay)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    settings::OptimizerSettings::setLearningRateStrategy(
        LearningRate::CONSTANT
    );
    setKeyword("nstep");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsUnimplementedLineSearchOptimization)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    settings::OptimizerSettings::setLearningRateStrategy(
        LearningRate::LINESEARCH_WOLFE
    );
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "The Wolfe line search learning rate strategy is not yet implemented"
    );
}

TEST_F(TestInputValidation, rejectsMissingLearningRateStrategy)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "In order to run the optimizer, you need to specify a learning rate "
        "strategy."
    );
}

TEST_F(TestInputValidation, rejectsOverlappingLearningRateBounds)
{
    settings::Settings::setJobtype(settings::JobType::MM_OPT);
    settings::TimingsSettings::setNumberOfSteps(100);
    settings::OptimizerSettings::setLearningRateStrategy(
        LearningRate::CONSTANT
    );
    settings::OptimizerSettings::setMinLearningRate(0.5);
    settings::OptimizerSettings::setMaxLearningRate(0.5);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        exc::UserInputException,
        "The minimum learning rate 0.5 is greater or equal to the maximum "
        "learning rate 0.5, which is not allowed."
    );
}
