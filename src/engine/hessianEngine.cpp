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
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "hessianBuilder.hpp"
#include "hessianSettings.hpp"
#include "logOutput.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizer.hpp"
#include "optimizerSettings.hpp"
#include "outputFileSettings.hpp"
#include "physicalData.hpp"
#include "progressbar.hpp"
#include "referencesOutput.hpp"
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
        std::shared_ptr<opt::HessianBuilder> setupHessianBuilder(
            const HessianSettings &hessianSettings
        )
        {
            return opt::makeHessianBuilder(
                hessianSettings.getBuilder(),
                hessianSettings.getDisplacement()
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
            const auto alpha0     = settings.getInitialLearningRate();
            const auto lrStrategy = settings.getLearningRateStrategy();

            settings.validateLearningRateStrategy();

            switch (lrStrategy)
            {
                using enum LearningRate;

                case CONSTANT:
                    return std::make_shared<opt::ConstantLRStrategy>(alpha0);

                case CONSTANT_DECAY:
                {
                    const auto alphaFreq = settings.getLRUpdateFrequency();

                    return std::make_shared<opt::ConstantDecayLRStrategy>(
                        alpha0,
                        settings.getLearningRateDecay(),
                        alphaFreq
                    );
                }

                case EXPONENTIAL_DECAY:
                {
                    const auto alphaFreq = settings.getLRUpdateFrequency();

                    return std::make_shared<opt::ExpDecayLR>(
                        alpha0,
                        settings.getLearningRateDecay(),
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
            const auto minLR = settings.getMinLearningRate();
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
            const auto energyStrategy = settings.getEnConvStrategy();

            const auto useEnergyOptional   = settings.getUseEnergyConv();
            const auto useMaxForceOptional = settings.getUseMaxForceConv();
            const auto useRMSForceOptional = settings.getUseRMSForceConv();

            const auto energyOptional    = settings.getEnergyConv();
            const auto absEnergyOptional = settings.getAbsEnergyConv();
            const auto relEnergyOptional = settings.getRelEnergyConv();
            const auto forceOptional     = settings.getForceConv();
            const auto maxForceOptional  = settings.getMaxForceConv();
            const auto rmsForceOptional  = settings.getRMSForceConv();

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
        void writeHessian(
            const opt::HessianMatrix &hessian,
            const HessianSettings    &hessianSettings
        )
        {
            std::ofstream file(hessianSettings.getHessianFile());

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
        void writeHessianInfo(
            const opt::HessianMatrix &hessian,
            const HessianSettings    &hessianSettings
        )
        {
            std::ofstream file(hessianSettings.getHessianInfoFile());

            if (file.fail())
                throw exc::UserInputException(
                    "Could not open Hessian info file for writing."
                );

            file << "format = pq-hessian-info-v1\n";
            file << "hessian_file = " << hessianSettings.getHessianFile()
                 << '\n';
            file << "hessian_builder = "
                 << HessianBuilderTypeMeta::toString(hessianSettings.getBuilder(
                    ))
                 << '\n';
            file << "optimize_before_hessian = "
                 << (hessianSettings.optimizeBeforeHessian() ? "true" : "false")
                 << '\n';
            file << "hessian_displacement = "
                 << hessianSettings.getDisplacement() << '\n';
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

        if (hessianSettings.optimizeBeforeHessian())
        {
            _setupOptimization(evaluator);
            _runOptimization();
        }

        auto builder = setupHessianBuilder(hessianSettings);

        const auto hessian = builder->build(*evaluator, getSimulationBox());

        writeHessian(hessian, hessianSettings);
        writeHessianInfo(hessian, hessianSettings);

        timings::GlobalTimer::get().stopSimulationTimer();

        references::ReferencesOutput::writeReferencesFile();
        getEngineOutput().writeTimingsFile();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime() * MS_TO_S;

        getEngineOutput().getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    void HessianEngine::writeOutput() {}

    std::shared_ptr<opt::Evaluator> HessianEngine::_setupEvaluator()
    {
        std::shared_ptr<opt::Evaluator> evaluator;

        if (settings::GeneralSettings::getJobtype() == JobType::MM_HESSIAN)
            evaluator = std::make_shared<opt::MMEvaluator>(getSettings());

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

        _evaluator->evaluate();
        _optimizer->updateHistory();

        _setNSteps(_optimizer->getNEpochs());

        _writeOptimizationOutput();

        progressbar bar(static_cast<int>(_getNSteps()), true, std::cout);

        for (size_t i = 0; i < _getNSteps(); ++i)
        {
            bar.update();

            _takeOptimizationStep();

            if (_converged || _optStopped)
                break;

            _writeOptimizationOutput();
            deleteTmpFiles();
        }

        OptEngine::throwOnFailure(
            _converged,
            _optStopped,
            getStep(),
            _optimizer->getNEpochs(),
            _learningRateStrategy->getErrorMessages()
        );

        const auto msg =
            std::format("Optimizer converged after {} epochs.", getStep());

        getLogOutput().writeInfo(msg);
        out::StdoutOutput::writeInfo(msg);
    }

    void HessianEngine::_takeOptimizationStep()
    {
        _optimizer->update(_learningRateStrategy->getLearningRate(), getStep());

        _evaluator->evaluate();

        _optimizer->updateHistory();

        _converged = _optimizer->hasConverged();

        if (!_converged)
        {
            _learningRateStrategy->updateLearningRate(getStep(), _getNSteps());

            if (!_learningRateStrategy->getErrorMessages().empty())
                _optStopped = true;

            const auto &msg = _learningRateStrategy->getWarningMessages();

            if (!msg.empty())
            {
                const auto headerMessage = std::format(
                    "Updating learning rate did raise "
                    "the following warnings in epoch {} out of {}:",
                    getStep(),
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

        _setStep(getStep() + 1);
    }

    void HessianEngine::_writeOptimizationOutput()
    {
        const auto outputFreq =
            settings::OutputFileSettings::getOutputFrequency();
        const auto step0   = settings::TimingsSettings::getStepCount();
        const auto effStep = getStep() + step0;

        if (0 == getStep() % outputFreq)
        {
            getEngineOutput().writeXyzFile(getSimulationBox(), effStep);
            getEngineOutput().writeForceFile(getSimulationBox(), effStep);
            getEngineOutput().writeOptRstFile(getSimulationBox(), effStep);
            getEngineOutput().writeOptFile(getStep(), *_optimizer);
        }

        timings::GlobalTimer::get().stopAndRestartSimulationTimer();

        getPhysicalData().setLoopTime(
            timings::GlobalTimer::get().calculateLoopTime()
        );
        getAveragePhysicalData().updateAverages(getPhysicalData());

        if (0 == getStep() % outputFreq)
        {
            getAveragePhysicalData().makeAverages(
                static_cast<double>(outputFreq)
            );

            const auto effStepDouble = static_cast<double>(effStep);

            getEngineOutput().writeEnergyFile(
                effStep,
                getAveragePhysicalData()
            );
            getEngineOutput().writeInfoFile(
                effStepDouble,
                getAveragePhysicalData()
            );

            getAveragePhysicalData() = physicalData::PhysicalData();
        }

        getPhysicalData().reset();
    }

    std::shared_ptr<opt::Optimizer> HessianEngine::_setupEmptyOptimizer()
    {
        const auto nEpochs = settings::TimingsSettings::getNumberOfSteps();
        const auto optimizerType =
            getSettings().optimizerSettings.getOptimizer();

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
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Optimize before Hessian:    {}",
                getSettings().hessianSettings.optimizeBeforeHessian() ? "true"
                                                                      : "false"
            )
        );
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Optimizer:                  {}",
                OptimizerTypeMeta::toString(
                    getSettings().optimizerSettings.getOptimizer()
                )
            )
        );
        getEngineOutput().getLogOutput().writeSetupInfo(
            std::format(
                "Learning rate strategy:     {}",
                LearningRateMeta::toString(
                    getSettings().optimizerSettings.getLearningRateStrategy()
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
