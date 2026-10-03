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

#include "constraintsSetup.hpp"

#include <format>

#include "constraintSettings.hpp"   // for getShakeMaxIter, getShakeTolerance, getRattleMaxIter, getRattleTolerance
#include "constraints.hpp"    // for Constraints
#include "engine.hpp"         // for Engine
#include "mShakeReader.hpp"   // for readMShake

namespace setup
{

    /**
     * @brief constructs a new Constraints Setup:: Constraints Setup object and
     * calls setup
     *
     * @param engine
     */
    void setupConstraints(engine::Engine &engine)
    {
        if (!engine.isConstraintsActivated())
            return;

        out::StdoutOutput::writeSetup("Constraints");
        engine.getLogOutput().writeSetup("Constraints");

        ConstraintsSetup constraintsSetup(engine);
        constraintsSetup.setup();
    }

    /**
     * @brief constructor
     *
     * @param engine
     */
    ConstraintsSetup::ConstraintsSetup(engine::Engine &engine) : _engine(engine)
    {
    }

    /**
     * @brief sets constraints data in constraints object
     *
     * @details sets tolerances, max iterations, reference bond lengths and
     * timestep
     */
    void ConstraintsSetup::setup()
    {
        setupTolerances();
        setupMaxIterations();
        setupRefBondLengths();
        setupMShake();
        setupDegreesOfFreedom();

        writeSetupInfo();
    }

    /**
     * @brief setup M-SHAKE
     *
     */
    void ConstraintsSetup::setupMShake()
    {
        const auto &constraints = _engine.getConstraints();

        if (!constraints->isMShakeActive())
            return;

        input::mShake::readMShake(_engine);

        constraints->initMShake();
    }

    /**
     * @brief sets constraints tolerances
     *
     */
    void ConstraintsSetup::setupTolerances()
    {
        _shakeTolerance  = settings::ConstraintSettings::getShakeTolerance();
        _rattleTolerance = settings::ConstraintSettings::getRattleTolerance();

        const auto &constraints = _engine.getConstraints();

        constraints->setShakeTolerance(_shakeTolerance);
        constraints->setRattleTolerance(_rattleTolerance);
    }

    /**
     * @brief sets constraints max iterations
     *
     */
    void ConstraintsSetup::setupMaxIterations()
    {
        _shakeMaxIter  = settings::ConstraintSettings::getShakeMaxIter();
        _rattleMaxIter = settings::ConstraintSettings::getRattleMaxIter();

        const auto &constraints = _engine.getConstraints();

        constraints->setShakeMaxIter(_shakeMaxIter);
        constraints->setRattleMaxIter(_rattleMaxIter);
    }

    /**
     * @brief sets constraints reference bond lengths
     *
     */
    void ConstraintsSetup::setupRefBondLengths()
    {
        const auto &constraints = _engine.getConstraints();
        const auto &simBox      = _engine.getSimulationBox();

        constraints->calculateConstraintBondRefs(simBox);
    }

    /**
     * @brief corrects the number of degrees of freedom for the constraints
     *
     */
    void ConstraintsSetup::setupDegreesOfFreedom()
    {
        auto       &simBox      = _engine.getSimulationBox();
        const auto &constraints = _engine.getConstraints();

        _shakeConstraints  = constraints->getNumberOfBondConstraints();
        _mShakeConstraints = constraints->getNumberOfMShakeConstraints(simBox);

        auto dof  = simBox.getDegreesOfFreedom();
        dof      -= _shakeConstraints;
        dof      -= _mShakeConstraints;

        simBox.setDegreesOfFreedom(dof);
    }

    /**
     * @brief write setup information to log output
     *
     */
    void ConstraintsSetup::writeSetupInfo()
    {
        const auto &constraints = _engine.getConstraints();

        writeEnabled();

        if (constraints->isShakeLikeActive())
        {
            writeNConstraintBonds();
            writeTolerance();
        }

        if (constraints->isShakeActive())
            writeMaxIter();

        writeDof();
    }

    /**
     * @brief write enabled message to log output
     *
     */
    void ConstraintsSetup::writeEnabled()
    {
        const auto &constraints = _engine.getConstraints();

        std::string shakeMsg =
            constraints->isShakeActive() ? "enabled" : "disabled";
        std::string mShakeMsg =
            constraints->isMShakeActive() ? "enabled" : "disabled";

        shakeMsg  = std::format("SHAKE:   {}", shakeMsg);
        mShakeMsg = std::format("M-SHAKE: {}", mShakeMsg);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(shakeMsg);
        logOutput.writeSetupInfo(mShakeMsg);
        logOutput.writeEmptyLine();
    }

    /**
     * @brief write degrees of freedom message to log output
     *
     */
    void ConstraintsSetup::writeDof()
    {
        auto &simBox = _engine.getSimulationBox();

        const auto totalDof = simBox.getDegreesOfFreedom();

        const auto shakeDofMsg =
            std::format("SHAKE DOF:   {}", _shakeConstraints);
        const auto mShakeDofMsg =
            std::format("M-SHAKE DOF: {}", _mShakeConstraints);
        const auto totalDofMsg = std::format("Total DOF:   {}", totalDof);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(shakeDofMsg);
        logOutput.writeSetupInfo(mShakeDofMsg);
        logOutput.writeSetupInfo(totalDofMsg);
        logOutput.writeEmptyLine();
    }

    /**
     * @brief write tolerances message to log output
     *
     */
    void ConstraintsSetup::writeTolerance()
    {
        const auto shakeTolMsg =
            std::format("SHAKE Tolerance:  {}", _shakeTolerance);
        const auto rattleTolMsg =
            std::format("RATTLE Tolerance: {}", _rattleTolerance);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(shakeTolMsg);
        logOutput.writeSetupInfo(rattleTolMsg);
        logOutput.writeEmptyLine();
    }

    /**
     * @brief write max iterations message to log output
     *
     */
    void ConstraintsSetup::writeMaxIter()
    {
        const auto shakeMaxIterMsg =
            std::format("SHAKE Max Iter:  {}", _shakeMaxIter);
        const auto rattleMaxIterMsg =
            std::format("RATTLE Max Iter: {}", _rattleMaxIter);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(shakeMaxIterMsg);
        logOutput.writeSetupInfo(rattleMaxIterMsg);
        logOutput.writeEmptyLine();
    }

    /**
     * @brief write number of constraint bonds to log output
     *
     */
    void ConstraintsSetup::writeNConstraintBonds()
    {
        const auto &constraints = _engine.getConstraints();
        auto       &simBox      = _engine.getSimulationBox();

        const auto nShakeBonds  = constraints->getNumberOfBondConstraints();
        const auto nMShakeTypes = constraints->getMShakeReferences().size();
        const auto nMShakeMols =
            constraints->getNumberOfMShakeConstraints(simBox);

        const auto nShakeBondsMsg =
            std::format("Number of SHAKE bonds:       {}", nShakeBonds);
        const auto nMShakeTypesMsg =
            std::format("Number of M-SHAKE types:     {}", nMShakeTypes);
        const auto nMShakeMolsMsg =
            std::format("Number of M-SHAKE molecules: {}", nMShakeMols);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(nShakeBondsMsg);
        logOutput.writeSetupInfo(nMShakeTypesMsg);
        logOutput.writeSetupInfo(nMShakeMolsMsg);
        logOutput.writeEmptyLine();
    }

}   // namespace setup
