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

#include "hessianEngine.hpp"

#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "adam.hpp"
#include "constant.hpp"
#include "constantDecay.hpp"
#include "constants.hpp"
#include "convergenceSettings.hpp"
#include "defaults.hpp"
#include "enums/hessian.hpp"
#include "enums/optimizer.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"
#include "expDecay.hpp"
#include "globalTimer.hpp"
#include "hessianBuilder.hpp"
#include "hessianSettings.hpp"
#include "logOutput.hpp"
#include "mmEvaluator.hpp"
#include "optimizer.hpp"
#include "optimizerSettings.hpp"
#include "outputFileSettings.hpp"
#include "physicalData.hpp"
#include "progressbar.hpp"
#include "referencesOutput.hpp"
#include "settings.hpp"
#include "stdoutOutput.hpp"
#include "steepestDescent.hpp"
#include "timingsSettings.hpp"

namespace engine
{

    namespace
    {
        /**
         * @brief write the hessian matrix to a file
         *
         * @param hessian
         */
        [[nodiscard]]
        std::shared_ptr<opt::HessianBuilder> setupHessianBuilder()
        {
            return opt::makeHessianBuilder(
                settings::HessianSettings::getBuilder(),
                settings::HessianSettings::getDisplacement()
            );
        }

        /**
         * @brief setup the learning rate strategy based on user input
         *
         * @return std::shared_ptr<opt::LearningRateStrategy>
         */
        [[nodiscard]]
        std::shared_ptr<opt::LearningRateStrategy> setupLearningRateStrategy()
        {
            const auto alpha0 =
                settings::OptimizerSettings::getInitialLearningRate();
            const auto lrStrategy =
                settings::OptimizerSettings::getLearningRateStrategy();

            settings::OptimizerSettings::validateLearningRateStrategy();

            switch (lrStrategy)
            {
                using enum LearningRate;

                case CONSTANT:
                    return std::make_shared<opt::ConstantLRStrategy>(alpha0);

                case CONSTANT_DECAY:
                {
                    const auto alphaFreq =
                        settings::OptimizerSettings::getLRUpdateFrequency();

                    return std::make_shared<opt::ConstantDecayLRStrategy>(
                        alpha0,
                        settings::OptimizerSettings::getLearningRateDecay()
                            .value(),
                        alphaFreq
                    );
                }

                case EXPONENTIAL_DECAY:
                {
                    const auto alphaFreq =
                        settings::OptimizerSettings::getLRUpdateFrequency();

                    return std::make_shared<opt::ExpDecayLR>(
                        alpha0,
                        settings::OptimizerSettings::getLearningRateDecay()
                            .value(),
                        alphaFreq
                    );
                }

                case LINESEARCH_WOLFE:
                case NONE: break;
            }

            throw exc::UserInputException(
                "In order to run the optimizer, you need to specify a "
                "learning rate strategy."
            );
        }

        /**
         * @brief setup the min and max learning rate for the learning rate
         * strategy
         *
         * @param learningRate
         */
        void setupMinMaxLearningRate(
            std::shared_ptr<opt::LearningRateStrategy> &learningRate
        )
        {
            const auto minLR =
                settings::OptimizerSettings::getMinLearningRate();
            const auto maxLR =
                settings::OptimizerSettings::getMaxLearningRate();

            settings::OptimizerSettings::validateLearningRateBounds();

            learningRate->setMinLearningRate(minLR);
            learningRate->setMaxLearningRate(maxLR);
        }

        /**
         * @brief setup the convergence criteria for the optimizer
         *
         * @param optimizer
         */
        void setupConvergence(std::shared_ptr<opt::Optimizer> &optimizer)
        {
            const auto strategyOptional =
                settings::ConvSettings::getEnConvStrategy();
            const auto defaultStrategy =
                settings::ConvSettings::getDefaultEnergyConvStrategy();
            const auto energyStrategy =
                strategyOptional.value_or(defaultStrategy);

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
            const auto forceOptional = settings::ConvSettings::getForceConv();
            const auto maxForceOptional =
                settings::ConvSettings::getMaxForceConv();
            const auto rmsForceOptional =
                settings::ConvSettings::getRMSForceConv();

            auto relEnergy =
                energyOptional.value_or(defaults::REL_ENERGY_CONV_DEFAULT);
            auto absEnergy =
                energyOptional.value_or(defaults::ABS_ENERGY_CONV_DEFAULT);

            relEnergy = relEnergyOptional.value_or(relEnergy);
            absEnergy = absEnergyOptional.value_or(absEnergy);

            auto maxForce =
                forceOptional.value_or(defaults::MAX_FORCE_CONV_DEFAULT);
            auto rmsForce =
                forceOptional.value_or(defaults::RMS_FORCE_CONV_DEFAULT);

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
         * @brief write the hessian matrix to the file
         */
        void writeHessian(const opt::HessianMatrix &hessian)
        {
            std::ofstream file(settings::HessianSettings::getHessianFile());

            if (file.fail())
                throw exc::UserInputException(
                    "Could not open Hessian file for writing."
                );

            constexpr auto precision = 16;
            file << std::scientific << std::setprecision(precision);

            for (const auto &row : hessian)
            {
                for (size_t col = 0; col < row.size(); ++col)
                {
                    if (col != 0)
                        file << ' ';

                    file << row[col];
                }

                file << '\n';
            }
        }

        /**
         * @brief write the hessian info file
         *
         * @param hessian
         */
        void writeHessianInfo(const opt::HessianMatrix &hessian)
        {
            std::ofstream file(settings::HessianSettings::getHessianInfoFile());

            if (file.fail())
                throw exc::UserInputException(
                    "Could not open Hessian info file for writing."
                );

            file << "format = pq-hessian-info-v1\n";
            file << "hessian_file = "
                 << settings::HessianSettings::getHessianFile() << '\n';
            file << "hessian_builder = "
                 << HessianBuilderTypeMeta::toString(
                        settings::HessianSettings::getBuilder()
                    )
                 << '\n';
            file << "optimize_before_hessian = "
                 << (settings::HessianSettings::optimizeBeforeHessian()
                         ? "true"
                         : "false")
                 << '\n';
            file << "hessian_displacement = "
                 << settings::HessianSettings::getDisplacement() << '\n';
            file << "hessian_definition = -dF_i/dx_j\n";
            file << "hessian_unit = kcal_mol-1_angstrom-2\n";
            file << "rows = " << hessian.size() << '\n';
            file << "columns = " << (hessian.empty() ? 0 : hessian[0].size())
                 << '\n';
        }
    }   // namespace

    void HessianEngine::run()
    {
        auto evaluator = _setupEvaluator();

        if (settings::HessianSettings::optimizeBeforeHessian())
        {
            _setupOptimization(evaluator);
            _runOptimization();
        }

        auto builder = setupHessianBuilder();

        const auto hessian = builder->build(*evaluator, *_simulationBox);

        writeHessian(hessian);
        writeHessianInfo(hessian);

        timings::GlobalTimer::get().stopSimulationTimer();

        references::ReferencesOutput::writeReferencesFile();
        _engineOutput.writeTimingsFile();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime() * MS_TO_S;

        _engineOutput.getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    void HessianEngine::writeOutput() {}

    std::shared_ptr<opt::Evaluator> HessianEngine::_setupEvaluator()
    {
        std::shared_ptr<opt::Evaluator> evaluator;

        if (settings::Settings::getJobtype() == settings::JobType::MM_HESSIAN)
            evaluator = std::make_shared<opt::MMEvaluator>();

        else
            throw exc::UserInputException(
                "Unknown job type for Hessian evaluator setup."
            );

        evaluator->setCellList(getCellList());
        evaluator->setSimulationBox(getSharedSimulationBox());
        evaluator->setPotential(getPotential());
        evaluator->setForceField(getForceField());
        evaluator->setConstraints(getConstraints());
        evaluator->setIntraNonBonded(getIntraNonBonded());
        evaluator->setPhysicalData(getSharedPhysicalData());
        evaluator->setPhysicalDataOld(getSharedPhysicalDataOld());

        return evaluator;
    }

    void HessianEngine::_setupOptimization(
        const std::shared_ptr<opt::Evaluator> &evaluator
    )
    {
        _evaluator            = evaluator;
        _learningRateStrategy = setupLearningRateStrategy();
        _optimizer            = _setupEmptyOptimizer();

        setupConvergence(_optimizer);
        setupMinMaxLearningRate(_learningRateStrategy);

        _learningRateStrategy->setEvaluator(evaluator);
        _learningRateStrategy->setOptimizer(_optimizer);

        _writeOptimizationSetupInfo();
    }

    void HessianEngine::_runOptimization()
    {
        _converged  = false;
        _optStopped = false;

        _evaluator->evaluate();
        _optimizer->updateHistory();

        _nSteps = _optimizer->getNEpochs();

        _writeOptimizationOutput();

        progressbar bar(static_cast<int>(_nSteps), true, std::cout);

        for (size_t i = 0; i < _nSteps; ++i)
        {
            bar.update();

            _takeOptimizationStep();

            if (_converged || _optStopped)
                break;

            _writeOptimizationOutput();
            deleteTmpFiles();
        }

        if (!_converged)
        {
            throw exc::OptException(
                std::format(
                    "Optimizer did not converge after {} epochs.",
                    _optimizer->getNEpochs()
                )
            );
        }

        if (_optStopped)
        {
            auto msg = std::format(
                "Optimizer stopped after {} epochs out of {}. The following "
                "error "
                "messages were raised:\n",
                _step,
                _optimizer->getNEpochs()
            );

            const auto &errorMessages =
                _learningRateStrategy->getErrorMessages();

            for (size_t i = 0; i < errorMessages.size(); ++i)
                msg += std::format("{}) {}\n", i + 1, errorMessages[i]);

            throw exc::OptException(msg);
        }

        const auto msg =
            std::format("Optimizer converged after {} epochs.", _step);

        getLogOutput().writeInfo(msg);
        out::StdoutOutput::writeInfo(msg);
    }

    void HessianEngine::_takeOptimizationStep()
    {
        _optimizer->update(_learningRateStrategy->getLearningRate(), _step);

        _evaluator->evaluate();

        _optimizer->updateHistory();

        _converged = _optimizer->hasConverged();

        if (!_converged)
        {
            _learningRateStrategy->updateLearningRate(_step, _nSteps);

            if (!_learningRateStrategy->getErrorMessages().empty())
                _optStopped = true;

            const auto &msg = _learningRateStrategy->getWarningMessages();

            if (!msg.empty())
            {
                const auto headerMessage = std::format(
                    "Updating learning rate did raise "
                    "the following warnings in epoch {} out of {}:",
                    _step,
                    _optimizer->getNEpochs()
                );
                getLogOutput().writeOptWarning(headerMessage);
                out::StdoutOutput::writeOptWarning(headerMessage);

                for (const auto &message : msg)
                {
                    getLogOutput().writeOptWarning(message);
                    out::StdoutOutput::writeOptWarning(message);
                }
            }
        }

        ++_step;
    }

    void HessianEngine::_writeOptimizationOutput()
    {
        const auto outputFreq =
            settings::OutputFileSettings::getOutputFrequency();
        const auto step0   = settings::TimingsSettings::getStepCount();
        const auto effStep = _step + step0;

        if (0 == _step % outputFreq)
        {
            _engineOutput.writeXyzFile(*_simulationBox, effStep);
            _engineOutput.writeForceFile(*_simulationBox, effStep);
            _engineOutput.writeOptRstFile(*_simulationBox, effStep);
            _engineOutput.writeOptFile(_step, *_optimizer);
        }

        timings::GlobalTimer::get().stopAndRestartSimulationTimer();

        _physicalData->setLoopTime(
            timings::GlobalTimer::get().calculateLoopTime()
        );
        _averagePhysicalData.updateAverages(*_physicalData);

        if (0 == _step % outputFreq)
        {
            _averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

            const auto effStepDouble = static_cast<double>(effStep);

            _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
            _engineOutput.writeInfoFile(effStepDouble, _averagePhysicalData);

            _averagePhysicalData = physicalData::PhysicalData();
        }

        _physicalData->reset();
    }

    std::shared_ptr<opt::Optimizer> HessianEngine::_setupEmptyOptimizer()
    {
        const auto nEpochs = settings::TimingsSettings::getNumberOfSteps();
        const auto optimizerType = settings::OptimizerSettings::getOptimizer();

        std::shared_ptr<opt::Optimizer> optimizer;

        switch (optimizerType)
        {
            using enum OptimizerType;

            case STEEPEST_DESCENT:
            {
                optimizer = std::make_shared<opt::SteepestDescent>(nEpochs);
                break;
            }

            case ADAM:
            {
                const auto nAtoms = getSimulationBox().getNumberOfAtoms();
                optimizer = std::make_shared<opt::Adam>(nEpochs, nAtoms);
                break;
            }

            case NONE: break;
        }

        if (!optimizer)
        {
            throw exc::UserInputException(
                std::format(
                    "Unknown optimizer type {}",
                    OptimizerTypeMeta::toString(optimizerType)
                )
            );
        }

        optimizer->setSimulationBox(getSharedSimulationBox());
        optimizer->setPhysicalData(getSharedPhysicalData());
        optimizer->setPhysicalDataOld(getSharedPhysicalDataOld());

        return optimizer;
    }

    void HessianEngine::_writeOptimizationSetupInfo()
    {
        _engineOutput.getLogOutput().writeSetupInfo(
            std::format(
                "Optimize before Hessian:    {}",
                settings::HessianSettings::optimizeBeforeHessian() ? "true"
                                                                   : "false"
            )
        );
        _engineOutput.getLogOutput().writeSetupInfo(
            std::format(
                "Optimizer:                  {}",
                OptimizerTypeMeta::toString(
                    settings::OptimizerSettings::getOptimizer()
                )
            )
        );
        _engineOutput.getLogOutput().writeSetupInfo(
            std::format(
                "Learning rate strategy:     {}",
                LearningRateMeta::toString(
                    settings::OptimizerSettings::getLearningRateStrategy()
                )
            )
        );
        _engineOutput.getLogOutput().writeEmptyLine();
    }

    std::shared_ptr<physicalData::PhysicalData> HessianEngine::
        getSharedPhysicalDataOld()
    {
        return _physicalDataOld;
    }

    out::OptOutput &HessianEngine::getOptOutput()
    {
        return _engineOutput.getOptOutput();
    }

}   // namespace engine
