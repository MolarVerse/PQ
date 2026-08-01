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

#include "adam.hpp"
#include "constant.hpp"
#include "constantDecay.hpp"
#include "convergenceSettings.hpp"
#include "defaults.hpp"
#include "exceptions.hpp"
#include "expDecay.hpp"
#include "hessianBuilder.hpp"
#include "hessianSettings.hpp"
#include "logOutput.hpp"
#include "mmEvaluator.hpp"
#include "optimizerSettings.hpp"
#include "outputFileSettings.hpp"
#include "physicalData.hpp"
#include "progressbar.hpp"
#include "referencesOutput.hpp"
#include "settings.hpp"
#include "stdoutOutput.hpp"
#include "steepestDescent.hpp"
#include "timingsSettings.hpp"

using namespace engine;
using namespace opt;
using namespace settings;
using namespace customException;
using namespace physicalData;
using namespace defaults;

void HessianEngine::run()
{
    auto evaluator = setupEvaluator();

    if (HessianSettings::optimizeBeforeHessian())
    {
        setupOptimization(evaluator);
        runOptimization();
    }

    auto builder = setupHessianBuilder();

    const auto hessian = builder->build(*evaluator, *_simulationBox);

    writeHessian(hessian);
    writeHessianInfo(hessian);

    _timer.stopSimulationTimer();

    addTimers();
    references::ReferencesOutput::writeReferencesFile();
    _engineOutput.writeTimingsFile(_timer);

    const auto elapsedTime = double(_timer.calculateElapsedTime()) * 1e-3;
    _engineOutput.getLogOutput().writeEndedNormally(elapsedTime);
    _engineOutput.getStdoutOutput().writeEndedNormally(elapsedTime);
}

void HessianEngine::writeOutput() {}

pq::SharedEvaluator HessianEngine::setupEvaluator()
{
    pq::SharedEvaluator evaluator;

    if (Settings::getJobtype() == JobType::MM_HESSIAN)
        evaluator = std::make_shared<MMEvaluator>();

    else
        throw UserInputException(
            "Unknown job type for Hessian evaluator setup."
        );

    evaluator->setCellList(getSharedCellList());
    evaluator->setSimulationBox(getSharedSimulationBox());
    evaluator->setPotential(getSharedPotential());
    evaluator->setForceField(getSharedForceField());
    evaluator->setConstraints(getSharedConstraints());
    evaluator->setIntraNonBonded(getSharedIntraNonBonded());
    evaluator->setVirial(getSharedVirial());
    evaluator->setPhysicalData(getSharedPhysicalData());
    evaluator->setPhysicalDataOld(getSharedPhysicalDataOld());

    return evaluator;
}

pq::SharedHessianBuilder HessianEngine::setupHessianBuilder() const
{
    return makeHessianBuilder(
        HessianSettings::getBuilder(),
        HessianSettings::getDisplacement()
    );
}

void HessianEngine::setupOptimization(const pq::SharedEvaluator &evaluator)
{
    _evaluator            = evaluator;
    _learningRateStrategy = setupLearningRateStrategy();
    _optimizer            = setupEmptyOptimizer();

    setupConvergence(_optimizer);
    setupMinMaxLearningRate(_learningRateStrategy);

    _learningRateStrategy->setEvaluator(evaluator);
    _learningRateStrategy->setOptimizer(_optimizer);

    writeOptimizationSetupInfo();
}

void HessianEngine::runOptimization()
{
    _converged  = false;
    _optStopped = false;

    _evaluator->evaluate();
    _optimizer->updateHistory();

    _nSteps = _optimizer->getNEpochs();

    writeOptimizationOutput();

    progressbar bar(static_cast<int>(_nSteps), true, std::cout);

    for (size_t i = 0; i < _nSteps; ++i)
    {
        bar.update();

        takeOptimizationStep();

        if (_converged || _optStopped)
            break;

        writeOptimizationOutput();
        deleteTempFiles();
    }

    if (!_converged)
        throw OptException(
            std::format(
                "Optimizer did not converge after {} epochs.",
                _optimizer->getNEpochs()
            )
        );

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

        throw OptException(msg);
    }

    const auto msg = std::format("Optimizer converged after {} epochs.", _step);

    getLogOutput().writeInfo(msg);
    getStdoutOutput().writeInfo(msg);
}

void HessianEngine::takeOptimizationStep()
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

void HessianEngine::writeOptimizationOutput()
{
    const auto outputFreq = OutputFileSettings::getOutputFrequency();
    const auto step0      = TimingsSettings::getStepCount();
    const auto effStep    = _step + step0;

    if (0 == _step % outputFreq)
    {
        _engineOutput.writeXyzFile(*_simulationBox, effStep);
        _engineOutput.writeForceFile(*_simulationBox, effStep);
        _engineOutput.writeOptRstFile(*_simulationBox, effStep);
        _engineOutput.writeOptFile(_step, *_optimizer);
    }

    _timer.stopSimulationTimer();
    _timer.startSimulationTimer();

    _physicalData->setLoopTime(_timer.calculateLoopTime());
    _averagePhysicalData.updateAverages(*_physicalData);

    if (0 == _step % outputFreq)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

        const auto effStepDouble = static_cast<double>(effStep);

        _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(effStepDouble, _averagePhysicalData);

        _averagePhysicalData = PhysicalData();
    }

    _physicalData->reset();
}

pq::SharedOptimizer HessianEngine::setupEmptyOptimizer()
{
    const auto nEpochs       = TimingsSettings::getNumberOfSteps();
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
            const auto nAtoms = getSimulationBox().getNumberOfAtoms();
            optimizer         = std::make_shared<Adam>(nEpochs, nAtoms);
            break;
        }

        case NONE: break;
    }

    if (!optimizer)
        throw UserInputException(
            std::format("Unknown optimizer type {}", string(optimizerType))
        );

    optimizer->setSimulationBox(getSharedSimulationBox());
    optimizer->setPhysicalData(getSharedPhysicalData());
    optimizer->setPhysicalDataOld(getSharedPhysicalDataOld());

    return optimizer;
}

pq::SharedLearningRate HessianEngine::setupLearningRateStrategy()
{
    const auto alpha0     = OptimizerSettings::getInitialLearningRate();
    const auto lrStrategy = OptimizerSettings::getLearningRateStrategy();

    switch (lrStrategy)
    {
        using enum LREnum;

        case CONSTANT: return std::make_shared<ConstantLRStrategy>(alpha0);

        case CONSTANT_DECAY:
        {
            const auto alphaDecay = OptimizerSettings::getLearningRateDecay();

            if (!alphaDecay.has_value())
                throw UserInputException(
                    "You need to specify a learning rate decay factor for the "
                    "constant decay learning rate strategy"
                );

            const auto alphaFreq = OptimizerSettings::getLRUpdateFrequency();

            return std::make_shared<ConstantDecayLRStrategy>(
                alpha0,
                alphaDecay.value(),
                alphaFreq
            );
        }

        case EXPONENTIAL_DECAY:
        {
            const auto alphaDecay = OptimizerSettings::getLearningRateDecay();

            if (!alphaDecay.has_value())
                throw UserInputException(
                    "You need to specify a learning rate decay factor for the "
                    "exponential decay learning rate strategy"
                );

            const auto alphaFreq = OptimizerSettings::getLRUpdateFrequency();

            return std::make_shared<ExpDecayLR>(
                alpha0,
                alphaDecay.value(),
                alphaFreq
            );
        }

        case LINESEARCH_WOLFE:
            throw UserInputException(
                "The Wolfe line search learning rate strategy is not yet "
                "implemented"
            );

        case NONE: break;
    }

    throw UserInputException(
        "In order to run the optimizer, you need to specify a "
        "learning rate strategy."
    );
}

void HessianEngine::setupConvergence(pq::SharedOptimizer &optimizer)
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

    auto relEnergy = energyOptional.value_or(_REL_ENERGY_CONV_DEFAULT_);
    auto absEnergy = energyOptional.value_or(_ABS_ENERGY_CONV_DEFAULT_);

    relEnergy = relEnergyOptional.value_or(relEnergy);
    absEnergy = absEnergyOptional.value_or(absEnergy);

    auto maxForce = forceOptional.value_or(_MAX_FORCE_CONV_DEFAULT_);
    auto rmsForce = forceOptional.value_or(_RMS_FORCE_CONV_DEFAULT_);

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

void HessianEngine::setupMinMaxLearningRate(
    pq::SharedLearningRate &learningRate
)
{
    const auto minLR = OptimizerSettings::getMinLearningRate();
    const auto maxLR = OptimizerSettings::getMaxLearningRate();

    if (maxLR.has_value() && minLR >= maxLR.value())
        throw UserInputException(
            std::format(
                "The minimum learning rate {} is greater or equal to the "
                "maximum learning rate {}, which is not allowed.",
                minLR,
                maxLR.value()
            )
        );

    learningRate->setMinLearningRate(minLR);
    learningRate->setMaxLearningRate(maxLR);
}

void HessianEngine::writeOptimizationSetupInfo()
{
    _engineOutput.getLogOutput().writeSetupInfo(
        std::format(
            "Optimize before Hessian:    {}",
            HessianSettings::optimizeBeforeHessian() ? "true" : "false"
        )
    );
    _engineOutput.getLogOutput().writeSetupInfo(
        std::format(
            "Optimizer:                  {}",
            string(OptimizerSettings::getOptimizer())
        )
    );
    _engineOutput.getLogOutput().writeSetupInfo(
        std::format(
            "Learning rate strategy:     {}",
            string(OptimizerSettings::getLearningRateStrategy())
        )
    );
    _engineOutput.getLogOutput().writeEmptyLine();
}

void HessianEngine::writeHessian(const pq::HessianMatrix &hessian) const
{
    std::ofstream file(HessianSettings::getHessianFile());

    if (file.fail())
        throw UserInputException("Could not open Hessian file for writing.");

    file << std::scientific << std::setprecision(16);

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

void HessianEngine::writeHessianInfo(const pq::HessianMatrix &hessian) const
{
    std::ofstream file(HessianSettings::getHessianInfoFile());

    if (file.fail())
        throw UserInputException(
            "Could not open Hessian info file for writing."
        );

    file << "format = pq-hessian-info-v1\n";
    file << "hessian_file = " << HessianSettings::getHessianFile() << '\n';
    file << "hessian_builder = " << string(HessianSettings::getBuilder())
         << '\n';
    file << "optimize_before_hessian = "
         << (HessianSettings::optimizeBeforeHessian() ? "true" : "false")
         << '\n';
    file << "hessian_displacement = " << HessianSettings::getDisplacement()
         << '\n';
    file << "hessian_definition = -dF_i/dx_j\n";
    file << "hessian_unit = kcal_mol-1_angstrom-2\n";
    file << "rows = " << hessian.size() << '\n';
    file << "columns = " << (hessian.empty() ? 0 : hessian[0].size()) << '\n';
}

void HessianEngine::addTimers()
{
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
}

pq::SharedPhysicalData HessianEngine::getSharedPhysicalDataOld()
{
    return _physicalDataOld;
}

output::OptOutput &HessianEngine::getOptOutput()
{
    return _engineOutput.getOptOutput();
}
