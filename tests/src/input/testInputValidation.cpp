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

#include "celllist.hpp"              // for CellList
#include "defaults.hpp"              // for default settings
#include "exceptions.hpp"            // for InputFileException
#include "hessianSettings.hpp"       // for HessianSettings
#include "inputFileReader.hpp"       // for InputFileReader
#include "manostatSettings.hpp"      // for ManostatSettings
#include "optEngine.hpp"             // for OptEngine
#include "optimizerSettings.hpp"     // for OptimizerSettings
#include "potentialSettings.hpp"     // for PotentialSettings
#include "qmSettings.hpp"            // for QMSettings
#include "settings.hpp"              // for Settings
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "thermostatSettings.hpp"    // for ThermostatSettings
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG
#include "timingsSettings.hpp"       // for TimingsSettings

using namespace customException;
using namespace input;
using namespace settings;

class TestInputValidation : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        Settings::setJobtype(JobType::NONE);
        HessianSettings::setOptimizeBeforeHessian(false);

        OptimizerSettings::setLearningRateStrategy(LREnum::CONSTANT);
        OptimizerSettings::setMinLearningRate(1.0e-15);
        OptimizerSettings::setMaxLearningRate(1.0);

        ManostatSettings::setManostatType(ManostatType::NONE);
        ManostatSettings::setTauManostat(
            defaults::BERENDSEN_MANOSTAT_RELAX_TIME
        );

        ThermostatSettings::setThermostatType(ThermostatType::NONE);
        ThermostatSettings::setTargetTemperature(0.0);
        ThermostatSettings::setStartTemperature(0.0);
        ThermostatSettings::setEndTemperature(0.0);
        ThermostatSettings::setTemperatureSet(false);
        ThermostatSettings::setStartTemperatureSet(false);
        ThermostatSettings::setEndTemperatureSet(false);
        ThermostatSettings::setTemperatureRampSteps(0);
        ThermostatSettings::setTemperatureRampFrequency(1);
        ThermostatSettings::setRelaxationTime(
            defaults::BERENDSEN_THERMOSTAT_RELAX_TIME
        );
        ThermostatSettings::setFriction(defaults::LANGEVIN_THERMOSTAT_FRICTION);
        SimulationBoxSettings::setInitializeVelocities(InitVelocities::FALSE);

        PotentialSettings::setCoulombLongRangeType(
            CoulombLongRangeType::SHIFTED
        );
        PotentialSettings::setCoulombRadiusCutOff(
            defaults::COULOMB_CUT_OFF_DEFAULT
        );
        TimingsSettings::setTimeStep(0.5);

        QMSettings::setQMMethod(QMMethod::NONE);
        QMSettings::setMaceModel(MaceModel::MEDIUM);
        QMSettings::setMaceModelType(MaceModelType::MACE_MP);
        QMSettings::setMaceModelPath("");
        QMSettings::setSlakosType(SlakosType::NONE);
        QMSettings::setUseThirdOrderDftb(false);
        QMSettings::setIsThirdOrderDftbSet(false);
        QMSettings::setIsHubbardDerivsSet(false);
        QMSettings::setFennolModelPath("");

        _engine = std::make_unique<engine::OptEngine>();
        _reader = std::make_unique<InputFileReader>("input.in", *_engine);
    }

    void setKeyword(const std::string &keyword)
    {
        _reader->setKeywordCount(keyword, 1);
    }

    void configureMDJob(const JobType jobType)
    {
        Settings::setJobtype(jobType);
        TimingsSettings::setNumberOfSteps(100);
        setKeyword("nstep");
        setKeyword("timestep");
        if (jobType == JobType::QM_MD || jobType == JobType::RING_POLYMER_QM_MD)
            setKeyword("qm_prog");
    }

    std::unique_ptr<engine::OptEngine> _engine;
    std::unique_ptr<InputFileReader>   _reader;
};

TEST_F(TestInputValidation, requiresNumberOfStepsForMD)
{
    Settings::setJobtype(JobType::MM_MD);
    setKeyword("timestep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "Job type MM_MD selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, requiresNumberOfStepsForOptimization)
{
    Settings::setJobtype(JobType::MM_OPT);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "Job type MM_OPT selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, requiresNumberOfStepsForPreoptimizedHessian)
{
    Settings::setJobtype(JobType::MM_HESSIAN);
    HessianSettings::setOptimizeBeforeHessian(true);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "Job type MM_HESSIAN selected. Please set nstep in the input file."
    );
}

TEST_F(TestInputValidation, hessianWithoutOptimizationNeedsNoTimings)
{
    Settings::setJobtype(JobType::MM_HESSIAN);

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresTimeStepForMD)
{
    Settings::setJobtype(JobType::MM_MD);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "Molecular Dynamics job type MM_MD selected. Please set the time step "
        "in the input file."
    );
}

TEST_F(TestInputValidation, requiresPressureForManostat)
{
    ManostatSettings::setManostatType(ManostatType::BERENDSEN);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Pressure not set for berendsen manostat"
    );
}

TEST_F(TestInputValidation, rejectsUnstableManostatRelaxationTime)
{
    configureMDJob(JobType::MM_MD);
    ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    ManostatSettings::setTauManostat(0.0001);
    setKeyword("pressure");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "The timestep must not exceed the manostat relaxation time"
    );
}

TEST_F(TestInputValidation, requiresQMProgramForQMJob)
{
    Settings::setJobtype(JobType::QM_MD);
    setKeyword("nstep");
    setKeyword("timestep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "QM job selected but the \"qm_prog\" keyword has not been set"
    );
}

TEST_F(TestInputValidation, requiresTemperatureForThermostat)
{
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Target or end temperature not set for berendsen thermostat"
    );
}

TEST_F(TestInputValidation, rejectsBothThermostatTemperatures)
{
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    setKeyword("temp");
    setKeyword("end_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Both target and end temperature set for berendsen thermostat. They "
        "are mutually exclusive as they are treated as synonyms"
    );
}

TEST_F(TestInputValidation, acceptsEndTemperatureForThermostat)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    ThermostatSettings::setEndTemperature(300.0);
    setKeyword("end_temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
    EXPECT_DOUBLE_EQ(ThermostatSettings::getTargetTemperature(), 0.0);
    EXPECT_DOUBLE_EQ(ThermostatSettings::getActualTargetTemperature(), 0.0);
}

TEST_F(TestInputValidation, requiresTemperatureForVelocityInitialization)
{
    configureMDJob(JobType::MM_MD);
    SimulationBoxSettings::setInitializeVelocities(InitVelocities::FORCE);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Initializing velocities requires temp, start_temp, or end_temp"
    );
}

TEST_F(TestInputValidation, rejectsUnstableThermostatRelaxationTime)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::VELOCITY_RESCALING);
    ThermostatSettings::setTargetTemperature(300.0);
    ThermostatSettings::setRelaxationTime(0.0001);
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "The timestep must not exceed the thermostat relaxation time"
    );
}

TEST_F(TestInputValidation, rejectsNonFiniteLangevinScale)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::LANGEVIN);
    ThermostatSettings::setTargetTemperature(300.0);
    ThermostatSettings::setFriction(std::numeric_limits<double>::max());
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Langevin thermostat parameters produce a non-finite random-force "
        "scale"
    );
}

TEST_F(TestInputValidation, rejectsNonFiniteLangevinRampScale)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::LANGEVIN);
    ThermostatSettings::setTargetTemperature(300.0);
    ThermostatSettings::setStartTemperature(std::numeric_limits<double>::max());
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Langevin thermostat parameters produce a non-finite random-force "
        "scale"
    );
}

TEST_F(TestInputValidation, rejectsCellListWithoutCoulombCutoff)
{
    configureMDJob(JobType::MM_MD);
    _engine->getCellList().activate();
    PotentialSettings::setCoulombRadiusCutOff(0.0);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "An active cell list requires rcoulomb to be greater than zero"
    );
}

TEST_F(TestInputValidation, rejectsCellListForPureQM)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    _engine->getCellList().activate();

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Cell lists are not available for pure QM simulations"
    );
}

TEST_F(TestInputValidation, rejectsZeroTemperatureForNoseHoover)
{
    ThermostatSettings::setThermostatType(ThermostatType::NOSE_HOOVER);
    ThermostatSettings::setTargetTemperature(0.0);
    setKeyword("temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Nose-Hoover target temperature must be greater than zero"
    );
}

TEST_F(TestInputValidation, acceptsZeroTemperatureForBerendsen)
{
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    ThermostatSettings::setTargetTemperature(0.0);
    setKeyword("temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsTemperatureRampLongerThanSimulation)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    ThermostatSettings::setTemperatureRampSteps(200);
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Number of total simulation steps 100 is smaller than the number of "
        "temperature ramping steps 200"
    );
}

TEST_F(TestInputValidation, rejectsTemperatureRampFrequencyAboveRampSteps)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    ThermostatSettings::setTemperatureRampSteps(2);
    ThermostatSettings::setTemperatureRampFrequency(4);
    setKeyword("temp");
    setKeyword("start_temp");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Temperature ramp frequency 4 is larger than the number of ramping "
        "steps 2"
    );
}

TEST_F(TestInputValidation, acceptsDefaultTemperatureRampLength)
{
    configureMDJob(JobType::MM_MD);
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    ThermostatSettings::setTemperatureRampFrequency(100);
    setKeyword("temp");
    setKeyword("start_temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresReplicaCountForRingPolymer)
{
    configureMDJob(JobType::RING_POLYMER_QM_MD);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Number of beads not set for ring polymer simulation"
    );
}

TEST_F(TestInputValidation, acceptsReplicaCountForRingPolymer)
{
    configureMDJob(JobType::RING_POLYMER_QM_MD);
    setKeyword("rpmd_n_replica");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresSlaterKosterSetForAseDftbPlus)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::NONE);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "ASE-DFTB+ requires slakos to be 3ob, matsci, or custom"
    );
}

TEST_F(TestInputValidation, requiresPathForCustomSlaterKosterParameters)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::CUSTOM);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "Custom Slater-Koster parameters require the \"slakos_path\" keyword"
    );
}

TEST_F(TestInputValidation, rejectsHubbardDerivativesWithoutThirdOrder)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::CUSTOM);
    QMSettings::setUseThirdOrderDftb(false);
    setKeyword("slakos_path");
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "You have set custom Hubbard derivatives but disabled 3rd order DFTB. "
        "This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsHubbardDerivativesWithThirdOrder)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::CUSTOM);
    QMSettings::setUseThirdOrderDftb(true);
    setKeyword("slakos_path");
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

#ifdef WITH_ASE
TEST_F(TestInputValidation, rejectsExplicitlyDisabledThreeObThirdOrder)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::THREEOB);
    QMSettings::setUseThirdOrderDftb(false);
    setKeyword("third_order");
    setKeyword("hubbard_derivs");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "You have set custom Hubbard derivatives but disabled 3rd order DFTB. "
        "This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsThreeObDefaultThirdOrder)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType(SlakosType::THREEOB);
    QMSettings::setUseThirdOrderDftb(false);
    setKeyword("hubbard_derivs");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}
#endif

TEST_F(TestInputValidation, requiresFennolModelPath)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::FENNOL);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "The FeNNol QM runner has been selected but the "
        "\"fennol_model_path\" keyword has not been set. This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsFennolModelPath)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::FENNOL);
    setKeyword("fennol_model_path");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsMaceModelForWrongModelType)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModelType(MaceModelType::MACE_OFF);
    QMSettings::setMaceModel(MaceModel::MEDIUMOMAT0);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "The 'medium-omat-0' model size is only compatible with the 'mace_mp' "
        "model type."
    );
}

TEST_F(TestInputValidation, acceptsStandardMaceModelForNonMpType)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModelType(MaceModelType::MACE_OFF);
    QMSettings::setMaceModel(MaceModel::SMALL);

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresPathForCustomMaceModel)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModel(MaceModel::CUSTOM);

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "You have requested a custom MACE model but haven't provided a MACE "
        "model path.This setup is invalid."
    );
}

TEST_F(TestInputValidation, rejectsPathForBundledMaceModel)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModel(MaceModel::MEDIUMOMAT0);
    setKeyword("mace_model_path");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        InputFileException,
        "You have set a custom MACE model path without requesting a custom "
        "mace model size.This setup is invalid."
    );
}

TEST_F(TestInputValidation, acceptsValidConditionalKeywords)
{
    configureMDJob(JobType::QM_MD);
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModel(MaceModel::CUSTOM);
    ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    ThermostatSettings::setThermostatType(ThermostatType::BERENDSEN);
    setKeyword("mace_model_path");
    setKeyword("pressure");
    setKeyword("temp");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, requiresDecayForConstantDecayOptimization)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::CONSTANT_DECAY);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "The constant-decay learning rate strategy requires "
        "learning-rate-decay."
    );
}

TEST_F(TestInputValidation, requiresDecayForExponentialDecayOptimization)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::EXPONENTIAL_DECAY);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "The exponential-decay learning rate strategy requires "
        "learning-rate-decay."
    );
}

TEST_F(TestInputValidation, acceptsConstantOptimizationWithoutDecay)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::CONSTANT);
    setKeyword("nstep");

    EXPECT_NO_THROW(_reader->validateInputConfiguration());
}

TEST_F(TestInputValidation, rejectsUnimplementedLineSearchOptimization)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::LINESEARCH_WOLFE);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "The Wolfe line search learning rate strategy is not yet implemented"
    );
}

TEST_F(TestInputValidation, rejectsMissingLearningRateStrategy)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::NONE);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "In order to run the optimizer, you need to specify a learning rate "
        "strategy."
    );
}

TEST_F(TestInputValidation, rejectsOverlappingLearningRateBounds)
{
    Settings::setJobtype(JobType::MM_OPT);
    TimingsSettings::setNumberOfSteps(100);
    OptimizerSettings::setLearningRateStrategy(LREnum::CONSTANT);
    OptimizerSettings::setMinLearningRate(0.5);
    OptimizerSettings::setMaxLearningRate(0.5);
    setKeyword("nstep");

    ASSERT_THROW_MSG(
        _reader->validateInputConfiguration(),
        UserInputException,
        "The minimum learning rate 0.5 is greater or equal to the maximum "
        "learning rate 0.5, which is not allowed."
    );
}
