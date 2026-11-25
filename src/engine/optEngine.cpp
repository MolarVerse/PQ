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

#include "optEngine.hpp"

#include <format>

#include "constraintSettings.hpp"
#include "constraints.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "globalTimer.hpp"
#include "interrupt.hpp"
#include "outputFileSettings.hpp"
#include "simulationBox.hpp"
#include "terminalProgressBar.hpp"
#include "timingsSettings.hpp"

namespace engine
{

    /**
     * @brief run the optimizer
     */
    void OptEngine::run()
    {
        get_simulationBox().updateOldPositions();
        get_constraints().applyShakeToPositions(get_simulationBox());
        _evaluate();

        const auto nEpochs = getSettings().timings.getNumberOfSteps();

        const auto &outputFileSettings = getSettings().outputFiles;
        writeOutput(
            getSettings().generalSettings.getJobtype(),
            outputFileSettings
        );

        utilities::TerminalProgressBar bar(static_cast<int>(nEpochs));

        for (size_t i = 0; i < nEpochs; ++i)
        {
            utilities::throwIfInterrupted();

            bar.update();

            takeStep();

            if (_optStopped)
            {
                break;
            }

            writeOutput(
                getSettings().generalSettings.getJobtype(),
                outputFileSettings
            );
            deleteTmpFiles();

            if (_converged)
            {
                break;
            }
        }

        throwIfStopped(
            _optStopped,
            getStep(),
            nEpochs,
            _learningRateStrategy->getErrorMessages()
        );

        timings::GlobalTimer::get().stopSimulationTimer();

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime();

        getEngineOutput().writeTimingsFile();

        if (_converged)
        {
            const auto msg =
                std::format("Optimizer converged after {} epochs.", getStep());

            getLogOutput().writeInfo(msg);
            out::StdoutOutput::writeInfo(msg);
        }
        else
        {
            const auto msg = notConvergedMessage(nEpochs);

            getLogOutput().writeOptWarning(msg);
            out::StdoutOutput::writeOptWarning(msg);
        }

        getLogOutput().writeEndedNormally(elapsedTime);
        out::StdoutOutput::writeEndedNormally(elapsedTime);
    }

    /**
     * @brief throw if the learning rate strategy stopped the optimization
     *
     * @param optStopped whether the learning rate strategy stopped the run
     * @param step the step at which the run ended
     * @param nEpochs the maximum number of epochs
     * @param errorMessages the errors of the learning rate strategy
     *
     * @throw exc::OptException listing the errors if the run was stopped
     */
    void OptEngine::throwIfStopped(
        bool                            optStopped,
        size_t                          step,
        size_t                          nEpochs,
        const std::vector<std::string> &errorMessages
    )
    {
        if (!optStopped)
        {
            return;
        }

        auto msg = std::format(
            "Optimizer stopped after {} epochs out of {}. The following "
            "error messages were raised:\n",
            step,
            nEpochs
        );

        for (size_t i = 0; i < errorMessages.size(); ++i)
        {
            msg += std::format("{}) {}\n", i + 1, errorMessages[i]);
        }

        throw exc::OptException(msg);
    }

    /**
     * @brief throw if the optimization did not end successfully
     *
     * @param converged whether the optimizer converged
     * @param optStopped whether the learning rate strategy stopped the run
     * @param step the step at which the run ended
     * @param nEpochs the maximum number of epochs
     * @param errorMessages the errors of the learning rate strategy
     *
     * @throw exc::OptException if the run was stopped by the learning rate
     * strategy (listing its errors) or did not converge
     */
    void OptEngine::throwOnFailure(
        bool                            converged,
        bool                            optStopped,
        size_t                          step,
        size_t                          nEpochs,
        const std::vector<std::string> &errorMessages
    )
    {
        throwIfStopped(optStopped, step, nEpochs, errorMessages);

        if (!converged)
        {
            throw exc::OptException(notConvergedMessage(nEpochs));
        }
    }

    /**
     * @brief the message for a run that reached nEpochs without converging
     *
     * @param nEpochs the maximum number of epochs
     *
     * @return std::string
     */
    std::string OptEngine::notConvergedMessage(size_t nEpochs)
    {
        return std::format(
            "Optimizer did not converge after {} epochs.",
            nEpochs
        );
    }

    /**
     * @brief take a step
     */
    void OptEngine::takeStep()
    {
        _optimizer->update(
            _learningRateStrategy->getLearningRate(),
            getStep(),
            get_simulationBox()
        );
        get_constraints().applyShakeToPositions(get_simulationBox());

        _evaluate();

        _converged = _optimizer->hasConverged();

        if (!_converged)
        {
            const auto nEpochs = getSettings().timings.getNumberOfSteps();
            _learningRateStrategy->updateLearningRate(getStep(), nEpochs);

            if (!_learningRateStrategy->getErrorMessages().empty())
                _optStopped = true;

            const auto &msg = _learningRateStrategy->getWarningMessages();

            if (!msg.empty())
            {
                const auto headerMessage = std::format(
                    "Updating learning rate did raise "
                    "the following warnings in epoch {} out of {}:",
                    getStep(),
                    nEpochs
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

    /**
     * @brief evaluate the forces and add the energy to the optimizer history
     *
     * @details with SHAKE or M-SHAKE the forces are projected onto the
     * constraint surface, so the convergence checks see the constrained forces
     */
    void OptEngine::_evaluate()
    {
        _evaluator->evaluate(
            get_physicalData(),
            get_simulationBox(),
            get_intraNonBonded()
        );

        if (getSettings().constraintSettings.isShakeLikeActive())
        {
            get_constraints().calculateConstraintBondRefs(get_simulationBox());
            get_constraints().projectForces(get_simulationBox());
        }

        _optimizer->updateHistory(
            get_physicalData().getTotalEnergy(),
            get_simulationBox()
        );
    }

    /**
     * @brief Writes output files.
     *
     * @details output files are written if the step is a multiple of the output
     * frequency.
     *
     */
    void OptEngine::writeOutput(
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
     * @brief Writes output files.
     *
     * @details output files are written if the step is a multiple of the output
     * frequency.
     *
     */
    void OptEngine::_writeOutput(
        const OutputFileSettings &outputSettings,
        size_t                    startingStepCount
    )
    {
        const auto outputFreq = outputSettings.getOutputFrequency();
        const auto effStep    = getStep() + startingStepCount;

        if (0 == getStep() % outputFreq)
        {
            const out::Frame frame{get_simulationBox()};

            getEngineOutput().writeXyzFile(frame, effStep);
            getEngineOutput().writeForceFile(frame, effStep);
            getEngineOutput().writeOptRstFile(get_simulationBox(), effStep);
            getEngineOutput().writeOptFile(getStep(), *_optimizer);
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

            const auto effStepDouble = static_cast<double>(effStep);

            getEngineOutput().writeEnergyFile(
                effStep,
                get_averagePhysicalData(),
                getSettings()
            );
            getEngineOutput().getInfoOutput().write(
                effStepDouble,
                get_averagePhysicalData(),
                getSettings()
            );

            get_averagePhysicalData() = physicalData::PhysicalData();
        }

        get_physicalData().reset();
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief set the optimizer from a shared pointer
     *
     * @param optimizer
     */
    void OptEngine::setOptimizer(
        const std::shared_ptr<opt::Optimizer> &optimizer
    )
    {
        _optimizer = optimizer;
    }

    /**
     * @brief set the learning rate strategy from a shared pointer
     *
     * @param learningRateStrategy
     */
    void OptEngine::setLearningRateStrategy(
        const std::shared_ptr<opt::LearningRateStrategy> &learningRateStrategy
    )
    {
        _learningRateStrategy = learningRateStrategy;
    }

    /**
     * @brief set the evaluator from a shared pointer
     *
     * @param evaluator
     */
    void OptEngine::setEvaluator(
        const std::shared_ptr<opt::Evaluator> &evaluator
    )
    {
        _evaluator = evaluator;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the optimizer
     *
     */
    opt::Optimizer &OptEngine::getOptimizer() { return *_optimizer; }

    /**
     * @brief get the learning rate strategy
     *
     * @return LearningRate&
     */
    opt::LearningRateStrategy &OptEngine::getLearningRate()
    {
        return *_learningRateStrategy;
    }

    /**
     * @brief get the evaluator
     *
     * @return Evaluator&
     */
    opt::Evaluator &OptEngine::getEvaluator() { return *_evaluator; }

    /**
     * @brief get the convergence
     *
     * @return Convergence&
     */
    opt::Convergence &OptEngine::getConvergence()
    {
        return _optimizer->getConvergence();
    }

    /**
     * @brief get the optimizer as a shared pointer
     *
     * @return std::shared_ptr<Optimizer>
     */
    std::shared_ptr<opt::Optimizer> OptEngine::getSharedOptimizer()
    {
        return _optimizer;
    }

    /**
     * @brief get the learning rate strategy as a shared pointer
     *
     * @return std::shared_ptr<LearningRateStrategy>
     */
    std::shared_ptr<opt::LearningRateStrategy> OptEngine::getSharedLearningRate(
    )
    {
        return _learningRateStrategy;
    }

    /**
     * @brief get the evaluator as a shared pointer
     *
     * @return std::shared_ptr<Evaluator>
     */
    std::shared_ptr<opt::Evaluator> OptEngine::getSharedEvaluator()
    {
        return _evaluator;
    }

    /**
     * @brief get the optimizer output
     *
     * @return out::OptOutput&
     */
    out::OptOutput &OptEngine::getOptOutput()
    {
        return getEngineOutput().getOptOutput();
    }

}   // namespace engine
