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
#include "outputFileSettings.hpp"
#include "progressbar.hpp"
#include "settings.hpp"
#include "timingsSettings.hpp"

using namespace engine;
using namespace opt;
using namespace physicalData;

/**
 * @brief run the optimizer
 */
void OptEngine::run()
{
    _evaluator->evaluate();
    _optimizer->updateHistory();

    _nSteps = _optimizer->getNEpochs();

    progressbar bar(static_cast<int>(_nSteps), true, std::cout);

    for (size_t i = 0; i < _nSteps; ++i)
    {
        bar.update();

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
        auto msg = std::format(
            "Optimizer stopped after {} epochs out of {}. The following error "
            "messages were raised:\n",
            _step,
            _optimizer->getNEpochs()
        );

        const auto &errorMessages = _learningRateStrategy->getErrorMessages();

        for (size_t i = 0; i < errorMessages.size(); ++i)
            msg += std::format("{}) {}\n", i + 1, errorMessages[i]);

        throw customException::OptException(msg);
    }

    _timer.stopSimulationTimer();

    const auto elapsedTime = double(_timer.calculateElapsedTime()) * 1e-3;

    _engineOutput.setTimerName("Output");
    _timer.addTimer(_engineOutput.getTimer());

    _constraints->setTimerName("Constraints");
    _timer.addTimer(_constraints->getTimer());

    _cellList->setTimerName("Cell List");
    _timer.addTimer(_cellList->getTimer());

    _potential->setTimerName("Potential");
    _timer.addTimer(_potential->getTimer());

    _intraNonBonded->setTimerName("IntraNonBonded");
    _timer.addTimer(_intraNonBonded->getTimer());

    _physicalData->setTimerName("Physical Data");
    _timer.addTimer(_physicalData->getTimer());

    _engineOutput.writeTimingsFile(_timer);

    if (_converged)
    {
        const auto msg =
            std::format("Optimizer converged after {} epochs.", _step);

        getLogOutput().writeInfo(msg);
        getStdoutOutput().writeInfo(msg);

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
            getStdoutOutput().writeOptWarning(headerMessage);

            for (const auto &message : msg)
            {
                getLogOutput().writeOptWarning(message);
                getStdoutOutput().writeOptWarning(message);
            }
        }
    }

    ++_step;
}

/**
 * @brief Writes output files.
 *
 * @details output files are written if the step is a multiple of the output
 * frequency.
 *
 */
void OptEngine::writeOutput()
{
    const auto outputFreq = settings::OutputFileSettings::getOutputFrequency();
    const auto step0      = settings::TimingsSettings::getStepCount();
    const auto effStep    = _step + step0;

    if (0 == _step % outputFreq)
    {
        _engineOutput.writeXyzFile(*_simulationBox);
        _engineOutput.writeForceFile(*_simulationBox);
        _engineOutput.writeRstFile(*_simulationBox, _step + step0);
        _engineOutput.writeOptFile(_step, *_optimizer);

        // _engineOutput.writeVirialFile(
        //     effStep,
        //     *_physicalData
        // );   // use physicalData instead of averagePhysicalData

        // _engineOutput.writeStressFile(
        //     effStep,
        //     *_physicalData
        // );   // use physicalData instead of averagePhysicalData

        // _engineOutput.writeBoxFile(effStep, _simulationBox->getBox());
    }

    // NOTE:
    // stop and restart immediately time manager - maximum lost time is en file
    // writing in last step of simulation but on the other hand setup is now
    // included in total simulation time
    // Unfortunately, setup is therefore included in the first looptime output
    // but this is not a big problem - could also be a feature and not a bug
    _timer.stopSimulationTimer();
    _timer.startSimulationTimer();

    _physicalData->setLoopTime(_timer.calculateLoopTime());
    _averagePhysicalData.updateAverages(*_physicalData);

    if (0 == _step % outputFreq)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

        const auto dt            = settings::TimingsSettings::getTimeStep();
        const auto effStepDouble = static_cast<double>(effStep);
        const auto simTime       = effStepDouble * dt * constants::_FS_TO_PS_;

        _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
        // _engineOutput.writeInstantEnergyFile(effStep, _physicalData);
        _engineOutput.writeInfoFile(simTime, _averagePhysicalData);
        // _engineOutput.writeMomentumFile(effStep, _averagePhysicalData);

        _averagePhysicalData = PhysicalData();
    }

    _physicalData->reset();
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
void OptEngine::setOptimizer(const std::shared_ptr<Optimizer> optimizer)
{
    _optimizer = optimizer;
}

/**
 * @brief set the learning rate strategy from a shared pointer
 *
 * @param learningRateStrategy
 */
void OptEngine::setLearningRateStrategy(
    const std::shared_ptr<LearningRateStrategy> learningRateStrategy
)
{
    _learningRateStrategy = learningRateStrategy;
}

/**
 * @brief set the evaluator from a shared pointer
 *
 * @param evaluator
 */
void OptEngine::setEvaluator(const std::shared_ptr<Evaluator> evaluator)
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
 * @return std::shared_ptr<Optimizer>
 */
std::shared_ptr<Optimizer> OptEngine::getSharedOptimizer()
{
    return _optimizer;
}

/**
 * @brief get the learning rate strategy
 *
 * @return std::shared_ptr<LearningRateStrategy>
 */
std::shared_ptr<LearningRateStrategy> OptEngine::getSharedLearningRate()
{
    return _learningRateStrategy;
}

/**
 * @brief get the evaluator
 *
 * @return std::shared_ptr<Evaluator>
 */
std::shared_ptr<Evaluator> OptEngine::getSharedEvaluator()
{
    return _evaluator;
}

/**
 * @brief get the old physical data reference
 *
 * @return PhysicalData&
 */
PhysicalData &OptEngine::getPhysicalDataOld() { return *_physicalDataOld; }

/**
 * @brief get the shared old physical data reference
 *
 * @return SharedPhysicalData&
 */
std::shared_ptr<PhysicalData> OptEngine::getSharedPhysicalDataOld()
{
    return _physicalDataOld;
}

/**
 * @brief get the optimizer output
 *
 * @return output::OptOutput&
 */
output::OptOutput &OptEngine::getOptOutput()
{
    return _engineOutput.getOptOutput();
}