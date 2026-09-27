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

#include <format>
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

namespace setup
{

    /**
     * @brief Wrapper for the optimizer setup
     *
     * @param engine
     */
    void setupOptimizer(engine::Engine &engine)
    {
        if (!settings::Settings::isOptJobType())
            return;

        out::StdoutOutput::writeSetup("Optimizer");
        engine.getLogOutput().writeSetup("Optimizer");

        OptimizerSetup optimizerSetup(
            dynamic_cast<engine::OptEngine &>(engine)
        );
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
    std::shared_ptr<opt::Optimizer> OptimizerSetup::setupEmptyOptimizer()
    {
        const auto nEpochs = settings::TimingsSettings::getNumberOfSteps();
        const auto simBox  = _optEngine.getSimulationBox();
        const auto optimizerType = settings::OptimizerSettings::getOptimizer();

        std::shared_ptr<opt::Optimizer> optimizer;

        switch (optimizerType)
        {
            using enum settings::OptimizerType;

            case STEEPEST_DESCENT:
            {
                optimizer = std::make_shared<opt::SteepestDescent>(nEpochs);
                break;
            }

            case ADAM:
            {
                const auto nAtoms = simBox.getNumberOfAtoms();
                optimizer = std::make_shared<opt::Adam>(nEpochs, nAtoms);
                break;
            }

            case NONE: break;
        }

        if (!optimizer)
            throw exc::UserInputException(
                std::format("Unknown optimizer type {}", string(optimizerType))
            );

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
        const auto alpha_0 =
            settings::OptimizerSettings::getInitialLearningRate();
        const auto lrStrategy =
            settings::OptimizerSettings::getLearningRateStrategy();

        settings::OptimizerSettings::validateLearningRateStrategy();

        switch (lrStrategy)
        {
            using enum settings::LREnum;

            case CONSTANT:
                return std::make_shared<opt::ConstantLRStrategy>(alpha_0);

            case CONSTANT_DECAY:
            {
                const auto alphaDecayValue =
                    settings::OptimizerSettings::getLearningRateDecay().value();
                const auto alphaFreq =
                    settings::OptimizerSettings::getLRUpdateFrequency();

                return std::make_shared<opt::ConstantDecayLRStrategy>(
                    alpha_0,
                    alphaDecayValue,
                    alphaFreq
                );
            }

            case EXPONENTIAL_DECAY:
            {
                const auto alphaDecayValue =
                    settings::OptimizerSettings::getLearningRateDecay().value();
                const auto alphaFreq =
                    settings::OptimizerSettings::getLRUpdateFrequency();

                return std::make_shared<opt::ExpDecayLR>(
                    alpha_0,
                    alphaDecayValue,
                    alphaFreq
                );
            }

            case LINESEARCH_WOLFE:
            case NONE: break;
        }

        throw exc::UserInputException(
            "In order to run the optimizer, you need to specify a learning "
            "rate "
            "strategy."
        );
    }

    /**
     * @brief setup min max learning rate
     *
     * @param lrStrategy as shared pointer reference
     */
    void OptimizerSetup::setupMinMaxLR(
        std::shared_ptr<opt::LearningRateStrategy> &lrStrategy
    )
    {
        const auto minLR = settings::OptimizerSettings::getMinLearningRate();
        const auto maxLR = settings::OptimizerSettings::getMaxLearningRate();

        settings::OptimizerSettings::validateLearningRateBounds();

        lrStrategy->setMinLearningRate(minLR);
        lrStrategy->setMaxLearningRate(maxLR);
    }

    /**
     * @brief Setup the evaluator
     *
     */
    std::shared_ptr<opt::Evaluator> OptimizerSetup::setupEvaluator()
    {
        std::shared_ptr<opt::Evaluator> evaluator;

        if (settings::Settings::getJobtype() == settings::JobType::MM_OPT)
            evaluator = std::make_shared<opt::MMEvaluator>();
        else
        {
            throw exc::UserInputException(
                "Unknown job type for the optimizer in order to setup up the "
                "evaluator"
            );
        }

        evaluator->setCellList(_optEngine.getCellList());
        evaluator->setSimulationBox(_optEngine.getSharedSimulationBox());
        evaluator->setPotential(_optEngine.getPotential());
        evaluator->setForceField(_optEngine.getForceField());
        evaluator->setConstraints(_optEngine.getConstraints());
        evaluator->setIntraNonBonded(_optEngine.getIntraNonBonded());
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
        std::shared_ptr<opt::Optimizer> &optimizer
    )
    {
        const auto strategyOptional =
            settings::ConvSettings::getEnConvStrategy();
        const auto defaultStrategy =
            settings::ConvSettings::getDefaultEnergyConvStrategy();
        const auto energyStrategy = strategyOptional.value_or(defaultStrategy);

        const auto useEnergyOptional =
            settings::ConvSettings::getUseEnergyConv();
        const auto useMaxForceOptional =
            settings::ConvSettings::getUseMaxForceConv();
        const auto useRMSForceOptional =
            settings::ConvSettings::getUseRMSForceConv();

        const auto energyOptional = settings::ConvSettings::getEnergyConv();
        const auto absEnergyOptional =
            settings::ConvSettings::getAbsEnergyConv();
        const auto relEnergyOptional =
            settings::ConvSettings::getRelEnergyConv();
        const auto forceOptional    = settings::ConvSettings::getForceConv();
        const auto maxForceOptional = settings::ConvSettings::getMaxForceConv();
        const auto rmsForceOptional = settings::ConvSettings::getRMSForceConv();

        const auto defaultRelEnergy = defaults::REL_ENERGY_CONV_DEFAULT;
        const auto defaultAbsEnergy = defaults::ABS_ENERGY_CONV_DEFAULT;
        const auto defaultMaxForce  = defaults::MAX_FORCE_CONV_DEFAULT;
        const auto defaultRMSForce  = defaults::RMS_FORCE_CONV_DEFAULT;

        auto relEnergy = energyOptional.value_or(defaultRelEnergy);
        auto absEnergy = energyOptional.value_or(defaultAbsEnergy);

        relEnergy = relEnergyOptional.value_or(relEnergy);
        absEnergy = absEnergyOptional.value_or(absEnergy);

        auto maxForce = forceOptional.value_or(defaultMaxForce);
        auto rmsForce = forceOptional.value_or(defaultRMSForce);

        maxForce = maxForceOptional.value_or(maxForce);
        rmsForce = rmsForceOptional.value_or(rmsForce);

        const opt::Convergence convergence(
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
        const auto optimizer = settings::OptimizerSettings::getOptimizer();
        const auto lrStrategy =
            settings::OptimizerSettings::getLearningRateStrategy();

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

        using enum settings::ConvStrategy;

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

        const auto initialLR =
            settings::OptimizerSettings::getInitialLearningRate();
        const auto lrFreq = settings::OptimizerSettings::getLRUpdateFrequency();

        using enum settings::LREnum;

        std::string decayLRStr;

        if (lrStrategy == CONSTANT_DECAY || lrStrategy == EXPONENTIAL_DECAY)
        {
            const auto decay =
                settings::OptimizerSettings::getLearningRateDecay();
            decayLRStr = std::format("{:.2e}", decay.value());
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

}   // namespace setup
