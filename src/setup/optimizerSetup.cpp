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

#include "constantStrategy.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizerSettings.hpp"
#include "settings.hpp"
#include "steepestDescent.hpp"

using setup::OptimizerSetup;

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
    if (!settings::Settings::isOptJobType())
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
    const auto learningRateStrategy = setupLearningRateStrategy();
    const auto optimizer            = setupEmptyOptimizer();
    const auto evaluator            = setupEvaluator();

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
    const auto nEpochs = settings::OptimizerSettings::getNumberOfEpochs();

    std::shared_ptr<opt::Optimizer> optimizer;

    switch (settings::OptimizerSettings::getOptimizer())
    {
        case settings::Optimizer::STEEPEST_DESCENT:
        {
            optimizer = std::make_shared<opt::SteepestDescent>(nEpochs);
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown optimizer type {}",
                string(settings::OptimizerSettings::getOptimizer())
            ));
        }
    }

    optimizer->setSimulationBox(SharedSimBox(_optEngine.getSimulationBoxPtr()));

    return optimizer;
}

/**
 * @brief Setup the learning rate strategy
 *
 */
std::shared_ptr<opt::LearningRateStrategy> OptimizerSetup::
    setupLearningRateStrategy()
{
    const auto alpha_0 = settings::OptimizerSettings::getInitialLearningRate();

    switch (settings::OptimizerSettings::getLearningRateStrategy())
    {
        case settings::LearningRateStrategy::CONSTANT:
        {
            return std::make_shared<opt::ConstantLRStrategy>(alpha_0);
        }
        case settings::LearningRateStrategy::DECAY:
        {
            throw customException::UserInputException(
                "Decay learning rate strategy not implemented yet"
            );
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown learning rate strategy type {}",
                string(settings::OptimizerSettings::getLearningRateStrategy())
            ));
        }
    }
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
        throw customException::UserInputException(
            "Unknown job type for the optimizer in order to setup up the "
            "evaluator"
        );

    evaluator->setCellList(SharedCellList(_optEngine.getCellListPtr()));
    evaluator->setForceField(SharedForceField(_optEngine.getForceFieldPtr()));
    evaluator->setPotential(SharedPotential(_optEngine.getPotentialPtr()));
    evaluator->setSimulationBox(SharedSimBox(_optEngine.getSimulationBoxPtr()));
    evaluator->setVirial(SharedVirial(_optEngine.getVirialPtr()));
    evaluator->setConstraints(SharedConstraints(_optEngine.getConstraintsPtr())
    );
    evaluator->setPhysicalData(SharedPhysicalData(_optEngine.getPhysicalDataPtr(
    )));
    evaluator->setPhysicalData(
        SharedPhysicalData(_optEngine.getPhysicalDataOldPtr())
    );
    evaluator->setIntraNonBonded(
        SharedIntraNonBonded(_optEngine.getIntraNonBondedPtr())
    );

    return evaluator;
}