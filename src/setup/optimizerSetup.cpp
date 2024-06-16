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
#include "convergenceSettings.hpp"
#include "defaults.hpp"
#include "mmEvaluator.hpp"
#include "optEngine.hpp"
#include "optimizerSettings.hpp"
#include "settings.hpp"
#include "steepestDescent.hpp"

using setup::OptimizerSetup;
using namespace settings;

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
    if (!Settings::isOptJobType())
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
    const auto nEpochs = OptimizerSettings::getNumberOfEpochs();

    std::shared_ptr<opt::Optimizer> optimizer;

    switch (OptimizerSettings::getOptimizer())
    {
        case Optimizer::STEEPEST_DESCENT:
        {
            optimizer = std::make_shared<opt::SteepestDescent>(nEpochs);
            break;
        }
        default:
        {
            throw customException::UserInputException(std::format(
                "Unknown optimizer type {}",
                string(OptimizerSettings::getOptimizer())
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
    const auto alpha_0 = OptimizerSettings::getInitialLearningRate();

    switch (OptimizerSettings::getLearningRateStrategy())
    {
        case LearningRateStrategy::CONSTANT:
        {
            return std::make_shared<opt::ConstantLRStrategy>(alpha_0);
        }
        case LearningRateStrategy::DECAY:
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
                string(OptimizerSettings::getLearningRateStrategy())
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

    if (Settings::getJobtype() == JobType::MM_OPT)
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

/**
 * @brief setup convergence
 *
 * @param optimizer as shared pointer reference
 */
void OptimizerSetup::setupConvergence(std::shared_ptr<opt::Optimizer> &optimizer
)
{
    setupConvergenceStrategy(optimizer);

    const auto useEnergyConv   = ConvSettings::getUseEnergyConv();
    const auto useMaxForceConv = ConvSettings::getUseMaxForceConv();
    const auto useRMSForceConv = ConvSettings::getUseRMSForceConv();

    optimizer->setEnableEnergyConv(useEnergyConv);
    optimizer->setEnableMaxForceConv(useMaxForceConv);
    optimizer->setEnableRMSForceConv(useRMSForceConv);

    const auto energyConv    = ConvSettings::getEnergyConv();
    const auto absEnergyConv = ConvSettings::getAbsEnergyConv();
    const auto relEnergyConv = ConvSettings::getRelEnergyConv();

    const auto defRelEnergyConv = defaults::_REL_ENERGY_CONV_DEFAULT_;
    const auto defAbsEnergyConv = defaults::_ABS_ENERGY_CONV_DEFAULT_;

    auto relEnergyConvValue = energyConv.value_or(defRelEnergyConv);
    auto absEnergyConvValue = energyConv.value_or(defAbsEnergyConv);

    relEnergyConvValue = relEnergyConv.value_or(relEnergyConvValue);
    absEnergyConvValue = absEnergyConv.value_or(absEnergyConvValue);

    optimizer->setRelEnergyConv(relEnergyConvValue);
    optimizer->setAbsEnergyConv(absEnergyConvValue);

    const auto forceConv    = ConvSettings::getForceConv();
    const auto relForceConv = ConvSettings::getRelForceConv();
    const auto absForceConv = ConvSettings::getAbsForceConv();

    const auto maxForceConv    = ConvSettings::getMaxForceConv();
    const auto relMaxForceConv = ConvSettings::getRelMaxForceConv();
    const auto absMaxForceConv = ConvSettings::getAbsMaxForceConv();

    const auto rmsForceConv    = ConvSettings::getRMSForceConv();
    const auto relRMSForceConv = ConvSettings::getRelRMSForceConv();
    const auto absRMSForceConv = ConvSettings::getAbsRMSForceConv();

    const auto defRelMaxForceConv = defaults::_REL_MAX_FORCE_CONV_DEFAULT_;
    const auto defAbsMaxForceConv = defaults::_ABS_MAX_FORCE_CONV_DEFAULT_;
    const auto defRelRMSForceConv = defaults::_REL_RMS_FORCE_CONV_DEFAULT_;
    const auto defAbsRMSForceConv = defaults::_ABS_RMS_FORCE_CONV_DEFAULT_;

    auto relMaxForceConvValue = forceConv.value_or(defRelMaxForceConv);
    auto absMaxForceConvValue = forceConv.value_or(defAbsMaxForceConv);
    auto relRMSForceConvValue = forceConv.value_or(defRelRMSForceConv);
    auto absRMSForceConvValue = forceConv.value_or(defAbsRMSForceConv);

    auto isRelAbsConvSet = false;
    isRelAbsConvSet      = isRelAbsConvSet || relForceConv.has_value();
    isRelAbsConvSet      = isRelAbsConvSet || absForceConv.has_value();

    auto isMaxRMSConvSet = false;
    isMaxRMSConvSet      = isMaxRMSConvSet || maxForceConv.has_value();
    isMaxRMSConvSet      = isMaxRMSConvSet || rmsForceConv.has_value();

    if (isRelAbsConvSet && isMaxRMSConvSet)
    {
        throw customException::UserInputException(std::format(
            "You defined at least one of "
            "{\"rel-force-conv\",\"abs-force-conv\"} "
            "and at least one of {\"max-force-conv\",\"rms-force-conv\"}. "
            "These two pairs of convergence criteria are mutually exclusive. "
            "For more details on the hierarchy of convergence criteria, see "
            "the documentation."
        ));
    }
    else if (isRelAbsConvSet)
    {
        relMaxForceConvValue = relForceConv.value_or(relMaxForceConvValue);
        absMaxForceConvValue = absForceConv.value_or(absMaxForceConvValue);
        relRMSForceConvValue = relForceConv.value_or(relRMSForceConvValue);
        absRMSForceConvValue = absForceConv.value_or(absRMSForceConvValue);
    }
    else if (isMaxRMSConvSet)
    {
        relMaxForceConvValue = maxForceConv.value_or(relMaxForceConvValue);
        absMaxForceConvValue = maxForceConv.value_or(absMaxForceConvValue);
        relRMSForceConvValue = rmsForceConv.value_or(relRMSForceConvValue);
        absRMSForceConvValue = rmsForceConv.value_or(absRMSForceConvValue);
    }

    relMaxForceConvValue = relMaxForceConv.value_or(relMaxForceConvValue);
    absMaxForceConvValue = absMaxForceConv.value_or(absMaxForceConvValue);
    relRMSForceConvValue = relRMSForceConv.value_or(relRMSForceConvValue);
    absRMSForceConvValue = absRMSForceConv.value_or(absRMSForceConvValue);

    optimizer->setRelMaxForceConv(relMaxForceConvValue);
    optimizer->setAbsMaxForceConv(absMaxForceConvValue);
    optimizer->setRelRMSForceConv(relRMSForceConvValue);
    optimizer->setAbsRMSForceConv(absRMSForceConvValue);
}

/**
 * @brief setup convergence strategy
 *
 * @param convStrategy as optional reference
 * @param energyConvStrategy as optional reference
 * @param forceConvStrategy as optional reference
 * @param optimizer as shared pointer reference
 */
void OptimizerSetup::setupConvergenceStrategy(
    std::shared_ptr<opt::Optimizer> &optimizer
)
{
    const auto convStrategy       = ConvSettings::getConvStrategy();
    const auto energyConvStrategy = ConvSettings::getEnergyConvStrategy();
    const auto forceConvStrategy  = ConvSettings::getForceConvStrategy();

    const auto defEConvStrategy = ConvSettings::getDefaultEnergyConvStrategy();
    const auto defFConvStrategy = ConvSettings::getDefaultForceConvStrategy();

    // set general conv strategy for both energy and force
    // if set - otherwise use default strategy
    auto energyStrategy = convStrategy.value_or(defEConvStrategy);
    auto forceStrategy  = convStrategy.value_or(defFConvStrategy);

    // if specific energy or force strategy is set, use it
    energyStrategy = energyConvStrategy.value_or(energyStrategy);
    forceStrategy  = forceConvStrategy.value_or(forceStrategy);

    optimizer->setEnergyConvStrategy(energyStrategy);
    optimizer->setForceConvStrategy(forceStrategy);
}