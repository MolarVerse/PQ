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
#include "convergenceSettings.hpp"
#include "enums/hessian.hpp"
#include "enums/optimizer.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"
#include "expDecay.hpp"
#include "frame.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "hessianBuilder.hpp"
#include "hessianSettings.hpp"
#include "interrupt.hpp"
#include "logOutput.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizer.hpp"
#include "optimizerSettings.hpp"
#include "outputFileSettings.hpp"
#include "physicalData.hpp"
#include "stdoutOutput.hpp"
#include "steepestDescent.hpp"
#include "terminalProgressBar.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

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
        std::shared_ptr<opt::HessianBuilder> setupHessianBuilder(
            const HessianSettings &hessianSettings
        )
        {
            return opt::makeHessianBuilder(
                hessianSettings.get_builder(),
                hessianSettings.get_displacement()
            );
        }

        /**
         * @brief setup the learning rate strategy based on user input
         *
         * @return std::shared_ptr<opt::LearningRateStrategy>
         */
        [[nodiscard]]
        std::shared_ptr<opt::LearningRateStrategy> setupLearningRateStrategy(
            const OptimizerSettings &settings
        )
        {
            const auto alpha0     = settings.get_initialLearningRate();
            const auto lrStrategy = settings.get_lRStrategy();

            settings.validateLearningRateStrategy();

            switch (lrStrategy)
            {
                using enum LearningRate;

                case CONSTANT:
                    return std::make_shared<opt::ConstantLRStrategy>(alpha0);

                case CONSTANT_DECAY:
                {
                    const auto alphaFreq = settings.get_lRupdateFrequency();

                    return std::make_shared<opt::ConstantDecayLRStrategy>(
                        alpha0,
                        settings.get_learningRateDecay(),
                        alphaFreq
                    );
                }

                case EXPONENTIAL_DECAY:
                {
                    const auto alphaFreq = settings.get_lRupdateFrequency();

                    return std::make_shared<opt::ExpDecayLR>(
                        alpha0,
                        settings.get_learningRateDecay(),
                        alphaFreq
                    );
                }

                case LINESEARCH_WOLFE: break;
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
            std::shared_ptr<opt::LearningRateStrategy> &learningRate,
            const OptimizerSettings                    &settings
        )
        {
            const auto minLR = settings.get_minLearningRate();
            const auto maxLR = settings.getMaxLearningRate();

            settings.validateLearningRateBounds();

            learningRate->setMinLearningRate(minLR);
            learningRate->setMaxLearningRate(maxLR);
        }

        /**
         * @brief setup the convergence criteria for the optimizer
         *
         * @param optimizer
         */
        void setupConvergence(
            std::shared_ptr<opt::Optimizer> &optimizer,
            const ConvSettings              &settings
        )
        {
            const auto energyStrategy = settings.get_energyConvStrategy();

            const auto useEnergyOptional   = settings.getUseEnergyConv();
            const auto useMaxForceOptional = settings.getUseMaxForceConv();
            const auto useRMSForceOptional = settings.getUseRMSForceConv();

            const auto absEnergy = settings.get_absEnergyConv();
            const auto relEnergy = settings.get_relEnergyConv();
            const auto maxForce  = settings.get_maxForceConv();
            const auto rmsForce  = settings.get_rmsForceConv();

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
        void writeHessian(
            const Hessian         &hessian,
            const HessianSettings &hessianSettings
        )
        {
            std::ofstream file(hessianSettings.get_hessianFile());

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

                    file << row[col].numerical_value_in(
                        kcal_per_mol / angstrom2
                    );
                }

                file << '\n';
            }
        }

        /**
         * @brief write the hessian info file
         *
         * @param hessian
         */
        void writeHessianInfo(
            const Hessian         &hessian,
            const HessianSettings &hessianSettings
        )
        {
            std::ofstream file(hessianSettings.get_hessianInfoFile());

            if (file.fail())
                throw exc::UserInputException(
                    "Could not open Hessian info file for writing."
                );

            file << "format = pq-hessian-info-v1\n";
            file << "hessian_file = " << hessianSettings.get_hessianFile()
                 << '\n';
            file << "hessian_builder = "
                 << HessianBuilderTypeMeta::toString(
                        hessianSettings.get_builder()
                    )
                 << '\n';
            file << "optimize_before_hessian = "
                 << (hessianSettings.get_optimizeBeforeHessian() ? "true"
                                                                 : "false")
                 << '\n';
            file << "hessian_displacement = "
                 << hessianSettings.get_displacement() << '\n';
            file << "hessian_definition = -dF_i/dx_j\n";
            file << "hessian_unit = kcal_mol-1_angstrom-2\n";
            file << "rows = " << hessian.size() << '\n';
            file << "columns = " << (hessian.empty() ? 0 : hessian[0].size())
                 << '\n';
        }
    }   // namespace

    void HessianEngine::run()
    {
        auto        evaluator       = _setupEvaluator();
        const auto &hessianSettings = getSettings().hessianSettings;

        if (hessianSettings.get_optimizeBeforeHessian())
        {
            _setupOptimization(evaluator);
            _runOptimization();
        }

        auto builder = setupHessianBuilder(hessianSettings);

        const auto hessian = builder->build(
            *evaluator,
            get_simulationBox(),
            get_physicalData(),
            get_intraNonBonded()
        );

        writeHessian(hessian, hessianSettings);
        writeHessianInfo(hessian, hessianSettings);

        timings::GlobalTimer::get().stopSimulationTimer();

        getEngineOutput().writeReferencesFile();
        getEngineOutput().writeTimingsFile();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime();

        getEngineOutput().getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    /**
     * @brief Writes output files for the Hessian engine.
     *
     * @param outputSettings The settings for output files.
     */
    void HessianEngine::writeOutput(
        JobType /*jobType*/,
        const OutputFileSettings &outputSettings
    )
    {
        _writeOutput(
            outputSettings,
            getSettings().timings.getStartingStepCount()
        );
    }

    /**
     * @brief Writes output files for the Hessian engine.
     *
     * @param outputSettings The settings for output files.
     * @param startingStepCount The step count from which to start writing
     * output files.
     */
    void HessianEngine::_writeOutput(
        const OutputFileSettings &outputSettings,
        size_t                    startingStepCount
    )
    {
        const auto outputFreq = outputSettings.get_outputFrequency();
        const auto effStep    = getStep() + startingStepCount;

        if (0 == getStep() % outputFreq)
        {
            const out::Frame frame{get_simulationBox()};

            getEngineOutput().writeXyzFile(frame, effStep);
            getEngineOutput().writeForceFile(frame, effStep);
            getEngineOutput().writeOptRstFile(get_simulationBox(), effStep);
            getEngineOutput().writeOptFile(getStep(), *_optimizer);
        }

        timings::GlobalTimer::get().stopAndRestartSimulationTimer();

        auto &physicalData        = get_physicalData();
        auto &averagePhysicalData = get_averagePhysicalData();

        physicalData.set_loopTime(
            timings::GlobalTimer::get().calculateLoopTime()
        );
        averagePhysicalData.updateAverages(physicalData);

        if (0 == getStep() % outputFreq)
        {
            averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

            const auto effStepDouble = static_cast<double>(effStep);

            getEngineOutput()
                .writeEnergyFile(effStep, averagePhysicalData, getSettings());
            getEngineOutput().getInfoOutput().write(
                effStepDouble,
                averagePhysicalData,
                getSettings()
            );

            averagePhysicalData = physicalData::PhysicalData();
        }

        physicalData.reset();
    }

    std::shared_ptr<opt::Evaluator> HessianEngine::_setupEvaluator()
    {
        std::shared_ptr<opt::Evaluator> evaluator;

        if (getSettings().generalSettings.getJobtype() == JobType::MM_HESSIAN)
            evaluator = std::make_shared<opt::MMEvaluator>(getSettings());

        else
            throw exc::UserInputException(
                "Unknown job type for Hessian evaluator setup."
            );

        evaluator->setCellList(getCellList());
        evaluator->setPotential(getPotential());
        evaluator->setForceField(getForceField());

        return evaluator;
    }

    void HessianEngine::_setupOptimization(
        const std::shared_ptr<opt::Evaluator> &evaluator
    )
    {
        const auto optimizerSettings = getSettings().optimizerSettings;
        _evaluator                   = evaluator;
        _learningRateStrategy = setupLearningRateStrategy(optimizerSettings);
        _optimizer            = _setupEmptyOptimizer();

        setupConvergence(_optimizer, getSettings().convSettings);
        setupMinMaxLearningRate(_learningRateStrategy, optimizerSettings);

        _learningRateStrategy->setEvaluator(evaluator);
        _learningRateStrategy->setOptimizer(_optimizer);

        _writeOptimizationSetupInfo();
    }

    void HessianEngine::_runOptimization()
    {
        _converged  = false;
        _optStopped = false;

        _evaluator->evaluate(
            get_physicalData(),
            get_simulationBox(),
            get_intraNonBonded()
        );
        _optimizer->updateHistory(
            get_physicalData().getTotalEnergy(),
            get_simulationBox()
        );

        const auto &outputFileSettings = getSettings().outputFiles;
        const auto &jobType = getSettings().generalSettings.getJobtype();
        writeOutput(jobType, outputFileSettings);

        const auto nSteps = getSettings().timings.getNumberOfSteps();

        utilities::TerminalProgressBar bar(static_cast<int>(nSteps));

        for (size_t i = 0; i < nSteps; ++i)
        {
            utilities::throwIfInterrupted();

            bar.update();

            _takeOptimizationStep();

            if (_optStopped)
            {
                break;
            }

            writeOutput(jobType, outputFileSettings);
            deleteTmpFiles();

            if (_converged)
            {
                break;
            }
        }

        OptEngine::throwOnFailure(
            _converged,
            _optStopped,
            getStep(),
            nSteps,
            _learningRateStrategy->getErrorMessages()
        );

        const auto msg =
            std::format("Optimizer converged after {} epochs.", getStep());

        getLogOutput().writeInfo(msg);
        out::StdoutOutput::writeInfo(msg);
    }

    void HessianEngine::_takeOptimizationStep()
    {
        _optimizer->update(
            _learningRateStrategy->getLearningRate(),
            getStep(),
            get_simulationBox()
        );

        _evaluator->evaluate(
            get_physicalData(),
            get_simulationBox(),
            get_intraNonBonded()
        );

        _optimizer->updateHistory(
            get_physicalData().getTotalEnergy(),
            get_simulationBox()
        );

        _converged = _optimizer->hasConverged();

        if (!_converged)
        {
            const auto nSteps = getSettings().timings.getNumberOfSteps();
            _learningRateStrategy->updateLearningRate(getStep(), nSteps);

            if (!_learningRateStrategy->getErrorMessages().empty())
                _optStopped = true;

            const auto &msg = _learningRateStrategy->getWarningMessages();

            if (!msg.empty())
            {
                const auto headerMessage = std::format(
                    "Updating learning rate did raise "
                    "the following warnings in epoch {} out of {}:",
                    getStep(),
                    nSteps
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

        _setStep(getStep() + 1);
    }

    std::shared_ptr<opt::Optimizer> HessianEngine::_setupEmptyOptimizer()
    {
        const auto optimizerType =
            getSettings().optimizerSettings.get_optimizer();

        std::shared_ptr<opt::Optimizer> optimizer;

        switch (optimizerType)
        {
            using enum OptimizerType;

            case STEEPEST_DESCENT:
            {
                optimizer = std::make_shared<opt::SteepestDescent>();
                break;
            }

            case ADAM:
            {
                const auto nAtoms = get_simulationBox().getNumberOfAtoms();
                optimizer         = std::make_shared<opt::Adam>(nAtoms);
                break;
            }
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

        return optimizer;
    }

    void HessianEngine::_writeOptimizationSetupInfo()
    {
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Optimize before Hessian:    {}",
                getSettings().hessianSettings.get_optimizeBeforeHessian()
                    ? "true"
                    : "false"
            )
        );
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Optimizer:                  {}",
                OptimizerTypeMeta::toString(
                    getSettings().optimizerSettings.get_optimizer()
                )
            )
        );
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Learning rate strategy:     {}",
                LearningRateMeta::toString(
                    getSettings().optimizerSettings.get_lRStrategy()
                )
            )
        );
        getEngineOutput().getLogOutput().writeEmptyLine();
    }

    std::shared_ptr<physicalData::PhysicalData> HessianEngine::
        getSharedPhysicalDataOld()
    {
        return _physicalDataOld;
    }

    out::OptOutput &HessianEngine::getOptOutput()
    {
        return getEngineOutput().getOptOutput();
    }

}   // namespace engine
