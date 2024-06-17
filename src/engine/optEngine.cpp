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

#include <format>   // for format

#include "exceptions.hpp"

using namespace engine;
using namespace opt;

/**
 * @brief run the optimizer
 */
void OptEngine::run()
{
    for (size_t i = 0; i < _optimizer->getNEpochs(); ++i)
    {
        takeStep();

        if (_converged || _optStopped)
            break;

        writeOutput();
    }

    if (!_converged)
    {
        throw customException::OptException(std::format(
            "Optimizer did not converge after {} epochs.",
            _optimizer->getNEpochs()
        ));
    }

    if (_optStopped)
    {
        auto exceptionMessage = std::format(
            "Optimizer stopped after {} epochs out of {}. The following error "
            "messages were raised:\n",
            _step,
            _optimizer->getNEpochs()
        );

        const auto &errorMessages = _learningRateStrategy->getErrorMessages();

        for (size_t i = 0; i < errorMessages.size(); ++i)
            exceptionMessage +=
                std::format("{}) {}\n", i + 1, errorMessages[i]);

        throw customException::OptException(exceptionMessage);
    }

    _timer.stopSimulationTimer();

    const auto elapsedTime = double(_timer.calculateElapsedTime()) * 1e-3;

    _engineOutput.setTimerName("Output");
    _timer.addTimer(_engineOutput.getTimer());

    _constraints.setTimerName("Constraints");
    _timer.addTimer(_constraints.getTimer());

    _cellList.setTimerName("Cell List");
    _timer.addTimer(_cellList.getTimer());

    _potential->setTimerName("Potential");
    _timer.addTimer(_potential->getTimer());

    _intraNonBonded.setTimerName("IntraNonBonded");
    _timer.addTimer(_intraNonBonded.getTimer());

    _physicalData.setTimerName("Physical Data");
    _timer.addTimer(_physicalData.getTimer());

    _engineOutput.writeTimingsFile(_timer);

    if (_converged)
    {
        const auto convergeMessage =
            std::format("Optimizer converged after {} epochs.", _step);

        getLogOutput().writeInfo(convergeMessage);
        getStdoutOutput().writeInfo(convergeMessage);

        getLogOutput().writeEndedNormally(elapsedTime);
        getStdoutOutput().writeEndedNormally(elapsedTime);
    }
}

/**
 * @brief take a step
 */
void OptEngine::takeStep()
{
    _optimizer->update(_learningRateStrategy->getLearningRate());

    _evaluator->updateForces();

    _converged = _optimizer->hasConverged();

    if (!_converged)
    {
        _learningRateStrategy->updateLearningRate();

        if (!_learningRateStrategy->getErrorMessages().empty())
            _optStopped = true;

        const auto &warningMessages =
            _learningRateStrategy->getWarningMessages();

        if (!warningMessages.empty())
        {
            const auto headerMessage = std::format(
                "Updating learning rate did raise "
                "the following warnings in epoch {} out of {}:",
                _step,
                _optimizer->getNEpochs()
            );
            getLogOutput().writeOptWarning(headerMessage);
            getStdoutOutput().writeOptWarning(headerMessage);

            for (const auto &message : warningMessages)
            {
                getLogOutput().writeOptWarning(message);
                getStdoutOutput().writeOptWarning(message);
            }
        }
    }

    ++_step;
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
void OptEngine::setOptimizer(const Optimizer &optimizer)
{
    _optimizer = optimizer.clone();
}

/**
 * @brief set the learning rate strategy from a shared pointer
 *
 * @param learningRateStrategy
 */
void OptEngine::setLearningRateStrategy(const LearningRateStrategy &strategy)
{
    _learningRateStrategy = strategy.clone();
}

/**
 * @brief set the evaluator from a shared pointer
 *
 * @param evaluator
 */
void OptEngine::setEvaluator(const Evaluator &evaluator)
{
    _evaluator = evaluator.clone();
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the optimizer
 *
 * @return std::shared_ptr<Optimizer>
 */
std::shared_ptr<Optimizer> &OptEngine::getOptimizer() { return _optimizer; }

/**
 * @brief get the learning rate strategy
 *
 * @return std::shared_ptr<LearningRateStrategy>
 */
std::shared_ptr<LearningRateStrategy> &OptEngine::getLearningRateStrategy()
{
    return _learningRateStrategy;
}

/**
 * @brief get the evaluator
 *
 * @return std::shared_ptr<Evaluator>
 */
std::shared_ptr<Evaluator> &OptEngine::getEvaluator() { return _evaluator; }

/**
 * @brief get the old physical data reference
 *
 * @return physicalData::PhysicalData&
 */
physicalData::PhysicalData &OptEngine::getPhysicalDataOld()
{
    return _physicalDataOld;
}