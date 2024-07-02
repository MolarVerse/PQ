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

#include "adam.hpp"
#include "constant.hpp"
#include "constantDecay.hpp"
#include "convergenceSettings.hpp"
#include "defaults.hpp"
#include "engine.hpp"
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
using namespace engine;
using namespace opt;

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
void setup::setupOptimizer(Engine &engine)
{
    if (!Settings::isOptJobType())
        return;

    engine.getStdoutOutput().writeSetup("Optimizer");
    engine.getLogOutput().writeSetup("Optimizer");

    OptimizerSetup optimizerSetup(dynamic_cast<OptEngine &>(engine));
    optimizerSetup.setup();
}

/**
 * @brief Construct a new OptimizerSetup object
 *
 * @param optEngine
 */
OptimizerSetup::OptimizerSetup(OptEngine &optEngine) : _optEngine(optEngine) {}

/**
 * @brief Setup the optimizer
 *
 */
void OptimizerSetup::setup()
{
    auto       learningRateStrategy = setupLearningRateStrategy();
    auto       optimizer            = setupEmptyOptimizer();
    const auto evaluator            = setupEvaluator();

    setupConvergence(optimizer);
    setupMinMaxLR(learningRateStrategy);

    learningRateStrategy->setEvaluator(evaluator);
    learningRateStrategy->setOptimizer(optimizer);

    _optEngine.setLearningRateStrategy(learningRateStrategy);
    _optEngine.setOptimizer(optimizer);
    _optEngine.setEvaluator(evaluator);

    writeSetupInfo();
}

/**
 * @brief Setup an empty optimizer
 *
 */
pq::SharedOptimizer OptimizerSetup::setupEmptyOptimizer()
{
    const auto nEpochs       = TimingsSettings::getNumberOfSteps();
    const auto simBox        = _optEngine.getSimulationBox();
    const auto optimizerType = OptimizerSettings::getOptimizer();

    pq::SharedOptimizer optimizer;

    switch (optimizerType)
    {
        using enum OptimizerType;

        case STEEPEST_DESCENT:
        {
            optimizer = std::make_shared<SteepestDescent>(nEpochs);
            break;
        }

        case ADAM:
        {
            const auto nAtoms = simBox.getNumberOfAtoms();
            optimizer         = std::make_shared<Adam>(nEpochs, nAtoms);
            break;
        }

        default:
            throw UserInputException(
                std::format("Unknown optimizer type {}", string(optimizerType))
            );
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
pq::SharedLearningRate OptimizerSetup::setupLearningRateStrategy()
{
    const auto alpha_0    = OptimizerSettings::getInitialLearningRate();
    const auto lrStrategy = OptimizerSettings::getLearningRateStrategy();

    switch (lrStrategy)
    {
        using enum LREnum;

        case CONSTANT: return std::make_shared<ConstantLRStrategy>(alpha_0);

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

            return std::make_shared<ConstantDecayLRStrategy>(
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

            return std::make_shared<ExpDecayLR>(
                alpha_0,
                alphaDecayValue,
                alphaFreq
            );
        }

        case LINESEARCH_WOLFE:
        {
            throw UserInputException(
                "The Wolfe line search learning rate strategy is not yet "
                "implemented"
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
void OptimizerSetup::setupMinMaxLR(pq::SharedLearningRate &lrStrategy)
{
    const auto minLR = OptimizerSettings::getMinLearningRate();
    const auto maxLR = OptimizerSettings::getMaxLearningRate();

    if (maxLR.has_value() && minLR >= maxLR.value())
        throw UserInputException(std::format(
            "The minimum learning rate {} is greater or equal to the "
            "maximum learning rate {}, which is not allowed.",
            minLR,
            maxLR.value()
        ));

    lrStrategy->setMinLearningRate(minLR);
    lrStrategy->setMaxLearningRate(maxLR);
}

/**
 * @brief Setup the evaluator
 *
 */
pq::SharedEvaluator OptimizerSetup::setupEvaluator()
{
    pq::SharedEvaluator evaluator;

    if (Settings::getJobtype() == JobType::MM_OPT)
        evaluator = std::make_shared<MMEvaluator>();

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
void OptimizerSetup::setupConvergence(pq::SharedOptimizer &optimizer)
{
    const auto strategyOptional = ConvSettings::getEnConvStrategy();
    const auto defaultStrategy  = ConvSettings::getDefaultEnergyConvStrategy();
    const auto energyStrategy   = strategyOptional.value_or(defaultStrategy);

    const auto useEnergyOptional   = ConvSettings::getUseEnergyConv();
    const auto useMaxForceOptional = ConvSettings::getUseMaxForceConv();
    const auto useRMSForceOptional = ConvSettings::getUseRMSForceConv();

    const auto energyOptional    = ConvSettings::getEnergyConv();
    const auto absEnergyOptional = ConvSettings::getAbsEnergyConv();
    const auto relEnergyOptional = ConvSettings::getRelEnergyConv();
    const auto forceOptional     = ConvSettings::getForceConv();
    const auto maxForceOptional  = ConvSettings::getMaxForceConv();
    const auto rmsForceOptional  = ConvSettings::getRMSForceConv();

    const auto defaultRelEnergy = _REL_ENERGY_CONV_DEFAULT_;
    const auto defaultAbsEnergy = _ABS_ENERGY_CONV_DEFAULT_;
    const auto defaultMaxForce  = _MAX_FORCE_CONV_DEFAULT_;
    const auto defaultRMSForce  = _RMS_FORCE_CONV_DEFAULT_;

    auto relEnergy = energyOptional.value_or(defaultRelEnergy);
    auto absEnergy = energyOptional.value_or(defaultAbsEnergy);

    relEnergy = relEnergyOptional.value_or(relEnergy);
    absEnergy = absEnergyOptional.value_or(absEnergy);

    auto maxForce = forceOptional.value_or(defaultMaxForce);
    auto rmsForce = forceOptional.value_or(defaultRMSForce);

    maxForce = maxForceOptional.value_or(maxForce);
    rmsForce = rmsForceOptional.value_or(rmsForce);

    const Convergence convergence(
        useEnergyOptional,
        useMaxForceOptional,
        useRMSForceOptional,
        relEnergy,
        absEnergy,
        maxForce,
        rmsForce,
        energyStrategy
    );

    optimizer->setConvergence(convergence);
}

/**
 * @brief write setup info
 *
 */
void OptimizerSetup::writeSetupInfo() const
{
    const auto optimizer  = OptimizerSettings::getOptimizer();
    const auto lrStrategy = OptimizerSettings::getLearningRateStrategy();

    const auto &convergence  = _optEngine.getOptimizer().getConvergence();
    const auto  convStrategy = convergence.getEnConvStrategy();

    const auto isEnergyConvEnabled   = convergence.isEnergyConvEnabled();
    const auto isMaxForceConvEnabled = convergence.isMaxForceConvEnabled();
    const auto isRMSForceConvEnabled = convergence.isRMSForceConvEnabled();

    const auto relEnergyConv = convergence.getRelEnergyConvThreshold();
    const auto absEnergyConv = convergence.getAbsEnergyConvThreshold();
    const auto maxForceConv  = convergence.getAbsMaxForceConvThreshold();
    const auto rmsForceConv  = convergence.getAbsRMSForceConvThreshold();

    auto relEnergyConvStr = std::format("{:.2e}", relEnergyConv);
    auto absEnergyConvStr = std::format("{:.2e}", absEnergyConv);
    auto maxForceConvStr  = std::format("{:.2e}", maxForceConv);
    auto rmsForceConvStr  = std::format("{:.2e}", rmsForceConv);

    const auto convStrategyStr = string(convStrategy);

    using enum ConvStrategy;

    if (convStrategy == RELATIVE)
        absEnergyConvStr = "disabled";

    else if (convStrategy == ABSOLUTE)
        relEnergyConvStr = "disabled";

    if (!isEnergyConvEnabled)
    {
        relEnergyConvStr = "disabled";
        absEnergyConvStr = "disabled";
    }

    if (!isMaxForceConvEnabled)
        maxForceConvStr = "disabled";

    if (!isRMSForceConvEnabled)
        rmsForceConvStr = "disabled";

    const auto initialLR = OptimizerSettings::getInitialLearningRate();
    const auto lrFreq    = OptimizerSettings::getLRUpdateFrequency();

    using enum LREnum;

    std::string decayLRStr = "";

    if (lrStrategy == CONSTANT_DECAY || lrStrategy == EXPONENTIAL_DECAY)
    {
        const auto decay         = OptimizerSettings::getLearningRateDecay();
        const auto alphaDecayStr = std::format("{:.2e}", decay.value());
    }

    // clang-format off
    const auto optMsg        = std::format("Optimizer:                   {}", string(optimizer));

    const auto lrMsg         = std::format("Learning rate strategy:      {}", string(lrStrategy));
    const auto initialLRMsg  = std::format("Initial learning rate:       {:.2e}", initialLR);
    const auto lrFreqMsg     = std::format("Learning rate update freq:   {}", lrFreq);
    const auto decayLRMsg    = std::format("Learning rate decay factor:  {}", decayLRStr);

    const auto convStratMsg  = std::format("Convergence strategy:        {}", convStrategyStr);
    const auto energyConvMsg = std::format("Relative Energy convergence: {}", relEnergyConvStr);
    const auto absEnergyMsg  = std::format("Absolute Energy convergence: {}", absEnergyConvStr);
    const auto maxForceMsg   = std::format("Max Force convergence:       {}", maxForceConvStr);
    const auto rmsForceMsg   = std::format("RMS Force convergence:       {}", rmsForceConvStr);
    // clang-format on

    auto &logOutput = _optEngine.getLogOutput();

    logOutput.writeSetupInfo(optMsg);
    logOutput.writeEmptyLine();

    logOutput.writeSetupInfo(lrMsg);
    logOutput.writeSetupInfo(lrFreqMsg);
    logOutput.writeSetupInfo(initialLRMsg);
    if (!decayLRStr.empty())
        logOutput.writeSetupInfo(decayLRMsg);

    logOutput.writeEmptyLine();

    logOutput.writeSetupInfo(convStratMsg);
    logOutput.writeSetupInfo(energyConvMsg);
    logOutput.writeSetupInfo(absEnergyMsg);
    logOutput.writeSetupInfo(maxForceMsg);
    logOutput.writeSetupInfo(rmsForceMsg);

    logOutput.writeEmptyLine();
}