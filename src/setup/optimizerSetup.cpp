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

#include "optimizerSetup.hpp"

#include <memory>

#include "constant.hpp"
#include "constantDecay.hpp"
#include "convergenceSettings.hpp"
#include "defaults.hpp"
#include "expDecay.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizerSettings.hpp"
#include "settings.hpp"
#include "steepestDescent.hpp"
#include "timingsSettings.hpp"

using setup::OptimizerSetup;
using namespace settings;
using namespace customException;
using namespace defaults;

using SharedCellList       = std::shared_ptr<simulationBox::CellList>;
using SharedSimBox         = std::shared_ptr<simulationBox::SimulationBox>;
using SharedForceField     = std::shared_ptr<forceField::ForceField>;
using SharedPotential      = std::shared_ptr<potential::Potential>;
using SharedPhysicalData   = std::shared_ptr<physicalData::PhysicalData>;
using SharedConstraints    = std::shared_ptr<constraints::Constraints>;
using SharedIntraNonBonded = std::shared_ptr<intraNonBonded::IntraNonBonded>;
using SharedVirial         = std::shared_ptr<virial::Virial>;

/**
 * @brief Wrapper for the optimizer setup
 *
 * @param engine
 */
void setup::setupOptimizer(engine::Engine &engine)
{
    if (!Settings::isOptJobType())
        return;

    engine.getStdoutOutput().writeSetup("optimizer");
    engine.getLogOutput().writeSetup("optimizer");

    OptimizerSetup optimizerSetup(dynamic_cast<engine::OptEngine &>(engine));
    optimizerSetup.setup();
}

/**
 * @brief Construct a new OptimizerSetup object
 *
 * @param optEngine
 */
OptimizerSetup::OptimizerSetup(engine::OptEngine &optEngine)
    : _optEngine(optEngine)
{
}

/**
 * @brief Setup the optimizer
 *
 */
void OptimizerSetup::setup()
{
    auto       learningRateStrategy = setupLearningRateStrategy();
    const auto optimizer            = setupEmptyOptimizer();
    const auto evaluator            = setupEvaluator();

    setupMinMaxLR(learningRateStrategy);

    _optEngine.setLearningRateStrategy(learningRateStrategy);
    _optEngine.setOptimizer(optimizer);
    _optEngine.setEvaluator(evaluator);
}

/**
 * @brief Setup an empty optimizer
 *
 */
std::shared_ptr<opt::Optimizer> OptimizerSetup::setupEmptyOptimizer()
{
    const auto nEpochs = TimingsSettings::getNumberOfSteps();

    std::shared_ptr<opt::Optimizer> optimizer;

    switch (OptimizerSettings::getOptimizer())
    {
        using enum Optimizer;

        case STEEPEST_DESCENT:
        {
            optimizer = std::make_shared<opt::SteepestDescent>(nEpochs);
            break;
        }
        default:
        {
            throw UserInputException(std::format(
                "Unknown optimizer type {}",
                string(OptimizerSettings::getOptimizer())
            ));
        }
    }

    optimizer->setSimulationBox(_optEngine.getSharedSimulationBox());
    optimizer->setPhysicalData(_optEngine.getSharedPhysicalData());
    optimizer->setPhysicalDataOld(_optEngine.getSharedPhysicalDataOld());

    return optimizer;
}

/**
 * @brief Setup the learning rate strategy
 *
 */
std::shared_ptr<opt::LearningRateStrategy> OptimizerSetup::
    setupLearningRateStrategy()
{
    const auto alpha_0 = OptimizerSettings::getInitialLearningRate();

    switch (OptimizerSettings::getLearningRateStrategy())
    {
        using enum LREnum;

        case CONSTANT:
        {
            return std::make_shared<opt::ConstantLRStrategy>(alpha_0);
        }
        case CONSTANT_DECAY:
        {
            const auto alphaDecay = OptimizerSettings::getLearningRateDecay();

            if (!alphaDecay.has_value())
                throw UserInputException(
                    "You need to specify a learning rate decay factor for the "
                    "constant decay learning rate strategy"
                );

            const auto alphaDecayValue = alphaDecay.value();
            const auto alphaFreq = OptimizerSettings::getLRUpdateFrequency();

            return std::make_shared<opt::ConstantDecayLRStrategy>(
                alpha_0,
                alphaDecayValue,
                alphaFreq
            );
        }

        case EXPONENTIAL_DECAY:
        {
            const auto alphaDecay = OptimizerSettings::getLearningRateDecay();

            if (!alphaDecay.has_value())
                throw UserInputException(
                    "You need to specify a learning rate decay factor for the "
                    "constant decay learning rate strategy"
                );

            const auto alphaDecayValue = alphaDecay.value();
            const auto alphaFreq = OptimizerSettings::getLRUpdateFrequency();

            return std::make_shared<opt::ExpDecayLR>(
                alpha_0,
                alphaDecayValue,
                alphaFreq
            );
        }

        default:
        {
            throw UserInputException(
                std::format("In order to run the optimizer, you need to "
                            "specify a learning rate strategy.")
            );
        }
    }
}

/**
 * @brief setup min max learning rate
 *
 * @param learningRateStrategy as shared pointer reference
 */
void OptimizerSetup::setupMinMaxLR(
    std::shared_ptr<opt::LearningRateStrategy> &learningRateStrategy
)
{
    const auto minLearningRate = OptimizerSettings::getMinLearningRate();
    const auto maxLearningRate = OptimizerSettings::getMaxLearningRate();

    if (maxLearningRate.has_value())
    {
        const auto maxLearningRateValue = maxLearningRate.value();

        if (minLearningRate >= maxLearningRateValue)
        {
            throw UserInputException(std::format(
                "The minimum learning rate {} is greater or equal to the "
                "maximum learning rate {}, which is not allowed.",
                minLearningRate,
                maxLearningRateValue
            ));
        }
    }

    learningRateStrategy->setMinLearningRate(minLearningRate);
    learningRateStrategy->setMaxLearningRate(maxLearningRate);
}

/**
 * @brief Setup the evaluator
 *
 */
std::shared_ptr<opt::Evaluator> OptimizerSetup::setupEvaluator()
{
    std::shared_ptr<opt::Evaluator> evaluator;

    if (Settings::getJobtype() == JobType::MM_OPT)
        evaluator = std::make_shared<opt::MMEvaluator>();

    else
        throw UserInputException(
            "Unknown job type for the optimizer in order to setup up the "
            "evaluator"
        );

    evaluator->setCellList(_optEngine.getSharedCellList());
    evaluator->setSimulationBox(_optEngine.getSharedSimulationBox());
    evaluator->setPotential(_optEngine.getSharedPotential());
    evaluator->setForceField(_optEngine.getSharedForceField());
    evaluator->setConstraints(_optEngine.getSharedConstraints());
    evaluator->setIntraNonBonded(_optEngine.getSharedIntraNonBonded());
    evaluator->setVirial(_optEngine.getSharedVirial());
    evaluator->setSimulationBox(_optEngine.getSharedSimulationBox());
    evaluator->setPhysicalData(_optEngine.getSharedPhysicalData());
    evaluator->setPhysicalDataOld(_optEngine.getSharedPhysicalDataOld());

    return evaluator;
}

/**
 * @brief setup convergence
 *
 * @param optimizer as shared pointer reference
 */
void OptimizerSetup::setupConvergence(
    const std::shared_ptr<opt::Optimizer> &optimizer
)
{
    const auto energyConvStrategy = ConvSettings::getEnConvStrategy();
    const auto defEConvStrategy = ConvSettings::getDefaultEnergyConvStrategy();
    const auto energyStrategy   = energyConvStrategy.value_or(defEConvStrategy);

    const auto useEnergyConv   = ConvSettings::getUseEnergyConv();
    const auto useMaxForceConv = ConvSettings::getUseMaxForceConv();
    const auto useRMSForceConv = ConvSettings::getUseRMSForceConv();

    const auto energyConv    = ConvSettings::getEnergyConv();
    const auto absEnergyConv = ConvSettings::getAbsEnergyConv();
    const auto relEnergyConv = ConvSettings::getRelEnergyConv();

    const auto defRelEnergyConv = _REL_ENERGY_CONV_DEFAULT_;
    const auto defAbsEnergyConv = _ABS_ENERGY_CONV_DEFAULT_;

    auto relEnergyConvValue = energyConv.value_or(defRelEnergyConv);
    auto absEnergyConvValue = energyConv.value_or(defAbsEnergyConv);

    relEnergyConvValue = relEnergyConv.value_or(relEnergyConvValue);
    absEnergyConvValue = absEnergyConv.value_or(absEnergyConvValue);

    const auto forceConv    = ConvSettings::getForceConv();
    const auto maxForceConv = ConvSettings::getMaxForceConv();
    const auto rmsForceConv = ConvSettings::getRMSForceConv();

    const auto defMaxForceConv = _MAX_FORCE_CONV_DEFAULT_;
    const auto defRMSForceConv = _RMS_FORCE_CONV_DEFAULT_;

    auto maxForceConvValue = forceConv.value_or(defMaxForceConv);
    auto rmsForceConvValue = forceConv.value_or(defRMSForceConv);

    maxForceConvValue = maxForceConv.value_or(maxForceConvValue);
    rmsForceConvValue = rmsForceConv.value_or(rmsForceConvValue);

    const opt::Convergence convergence(
        useEnergyConv,
        useMaxForceConv,
        useRMSForceConv,
        relEnergyConvValue,
        absEnergyConvValue,
        maxForceConvValue,
        rmsForceConvValue,
        energyStrategy
    );

    optimizer->setConvergence(convergence);
}