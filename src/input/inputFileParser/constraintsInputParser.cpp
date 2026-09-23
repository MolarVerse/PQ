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

#include "constraintsInputParser.hpp"

#include <cstddef>   // for size_t
#include <optional>
#include <utility>

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "constraints.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"
#include "references.hpp"         // for ReferencesOutput
#include "referencesOutput.hpp"   // for ReferencesOutput

using namespace input;
using namespace settings;
using namespace references;
using namespace exc;

/**
 * @brief Construct a new Input File Parser Constraints:: Input File Parser
 * Constraints object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) shake "<on/off>" 2)
 * shake-tolerance "<double>" 3) shake-iter "<size_t>" 4) rattle-iter "<size_t>"
 * 5) rattle-tolerance "<double>"
 *
 * @param constraints pointer to the constraints object
 */
ConstraintsInputParser::ConstraintsInputParser(
    std::shared_ptr<constraints::Constraints> constraints
)
    : _constraints(std::move(constraints))
{
    addShakeActivatedKeyword();
    addShakeToleranceKeyword();
    addShakeIterationKeyword();
    addRattleIterationKeyword();
    addRattleToleranceKeyword();
    addMShakeToleranceKeyword();
    addMShakeIterationKeyword();
    addDistanceConstraintActivatedKeyword();
}

/**
 * @brief add the shake activated keyword to the parser
 *
 * @details default value is "off"
 */
void ConstraintsInputParser::addShakeActivatedKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "shake",
        .title       = "Shake activation",
        .description = "Keyword to activate or deactivate shake constraints"
    };

    const auto defaultValue = settings::ShakeType::OFF;

    const auto setValue = [constraints = _constraints](auto value)
    {
        switch (value)
        {
            case settings::ShakeType::ON:
            case settings::ShakeType::SHAKE:
            {
                constraints->activateShake();
                ConstraintSettings::activateShake();
                ReferencesOutput::addReferenceFile(RATTLE_FILE);
                break;
            }
            case settings::ShakeType::OFF:
            {
                constraints->deactivateShake();
                constraints->deactivateMShake();
                ConstraintSettings::deactivateShake();
                ConstraintSettings::deactivateMShake();
                break;
            }
            case settings::ShakeType::MSHAKE:
            {
                constraints->activateMShake();
                constraints->activateShake();
                ConstraintSettings::activateMShake();
                ConstraintSettings::activateShake();
                break;
            }
        }
    };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<settings::ShakeType>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue
        }
    );

    addKeyword("shake", adapt(key), false);
}

/**
 * @brief add the shake tolerance keyword to the parser
 *
 * @details default value is 1e-8
 */
void ConstraintsInputParser::addShakeToleranceKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "shake-tolerance",
        .title       = "Shake tolerance",
        .description = "Keyword to set the shake tolerance"
    };

    const auto defaultValue = 1e-8;

    const RangeValidator<double, Greater::GT> rangeValidator{0.0, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setShakeTolerance(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<double>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("shake-tolerance", adapt(key), false);
}

/**
 * @brief add the shake iteration keyword to the parser
 *
 * @details default value is 20
 */
void ConstraintsInputParser::addShakeIterationKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "shake-iter",
        .title       = "Shake iteration",
        .description = "Keyword to set the maximum number of shake iterations"
    };

    const auto defaultValue = 20;

    const RangeValidator<size_t> rangeValidator{1, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setShakeMaxIter(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("shake-iter", adapt(key), false);
}

/**
 * @brief add the rattle tolerance keyword to the parser
 *
 * @details default value is 1e-8
 */
void ConstraintsInputParser::addRattleToleranceKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rattle-tolerance",
        .title       = "Rattle tolerance",
        .description = "Keyword to set the rattle tolerance"
    };

    const auto defaultValue = 1e-8;

    const RangeValidator<double, Greater::GT> rangeValidator{0.0, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setRattleTolerance(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<double>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("rattle-tolerance", adapt(key), false);
}

/**
 * @brief adding Rattle iteration keyword
 *
 * @details default value is 20
 */
void ConstraintsInputParser::addRattleIterationKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rattle-iter",
        .title       = "Rattle iteration",
        .description = "Keyword to set the maximum number of rattle iterations"
    };

    const auto defaultValue = 20;

    const RangeValidator<size_t> rangeValidator{1, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setRattleMaxIter(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("rattle-iter", adapt(key), false);
}

/**
 * @brief adding MShake tolerance keyword
 *
 * @details default value is 1e-8
 */
void ConstraintsInputParser::addMShakeToleranceKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "mshake-tolerance",
        .title       = "MShake tolerance",
        .description = "Keyword to set the MShake tolerance"
    };

    const auto defaultValue = 1e-8;

    const RangeValidator<double, Greater::GT> rangeValidator{0.0, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setMShakeTolerance(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<double>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("mshake-tolerance", adapt(key), false);
}

/**
 * @brief adding MShake iteration keyword
 *
 * @details default value is 20
 */
void ConstraintsInputParser::addMShakeIterationKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "mshake-iter",
        .title       = "MShake iteration",
        .description = "Keyword to set the maximum number of MShake iterations"
    };

    const auto defaultValue = 20;

    const RangeValidator<size_t> rangeValidator{1, std::nullopt};

    const auto setValue = [](auto value)
    { ConstraintSettings::setMShakeMaxIter(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue,
            .validator    = makeShared(rangeValidator)
        }
    );

    addKeyword("mshake-iter", adapt(key), false);
}

/**
 * @brief adding distance constraint activated keyword
 *
 * @details default value is false
 */
void ConstraintsInputParser::addDistanceConstraintActivatedKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "distance-constraints",
        .title       = "Distance constraint activated",
        .description = "Keyword to activate the distance constraint"
    };

    const auto defaultValue = false;

    const auto setValue = [constraints = _constraints](auto value)
    {
        if (value)
        {
            constraints->activateDistanceConstraints();
            ConstraintSettings::activateDistanceConstraints();
        }
        else
        {
            constraints->deactivateDistanceConstraints();
            ConstraintSettings::deactivateDistanceConstraints();
        }
    };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<bool>{
            .metadata     = metaData,
            .defaultValue = defaultValue,
            .onSet        = setValue
        }
    );

    addKeyword("distance-constraints", adapt(key), false);
}
