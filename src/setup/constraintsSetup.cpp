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

#include "constraintSettings.hpp"
#include "constraints.hpp"
#include "engine.hpp"
#include "mShakeReader.hpp"

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
        const auto &settings = engine.getSettings().constraintSettings;

        if (!settings.isActive())
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
        auto &constraints = _engine.get_constraints();

        if (!_engine.getSettings().constraintSettings.isMShakeActivated())
            return;

        input::mShake::readMShake(_engine);

        constraints.initMShake();
    }

    /**
     * @brief sets constraints reference bond lengths
     *
     */
    void ConstraintsSetup::setupRefBondLengths()
    {
        auto       &constraints = _engine.get_constraints();
        const auto &simBox      = _engine.get_simulationBox();

        constraints.calculateConstraintBondRefs(simBox);
    }

    /**
     * @brief corrects the number of degrees of freedom for the constraints
     *
     */
    void ConstraintsSetup::setupDegreesOfFreedom()
    {
        auto &simBox      = _engine.get_simulationBox();
        auto &constraints = _engine.get_constraints();

        _shakeConstraints  = constraints.getNumberOfBondConstraints();
        _mShakeConstraints = constraints.getNumberOfMShakeConstraints(simBox);

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
        writeEnabled();

        const auto settings = _engine.getSettings().constraintSettings;

        if (settings.isShakeLikeActive())
        {
            writeNConstraintBonds();
            writeTolerance();
        }

        if (settings.isShakeActivated())
            writeMaxIter();

        writeDof();
    }

    /**
     * @brief write enabled message to log output
     *
     */
    void ConstraintsSetup::writeEnabled()
    {
        const auto &settings = _engine.getSettings().constraintSettings;

        std::string shakeMsg =
            settings.isShakeActivated() ? "enabled" : "disabled";
        std::string mShakeMsg =
            settings.isMShakeActivated() ? "enabled" : "disabled";

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
        auto &simBox = _engine.get_simulationBox();

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
        const auto &settings = _engine.getSettings().constraintSettings;

        const auto shakeTolMsg =
            std::format("SHAKE Tolerance:  {}", settings.get_shakeTolerance());
        const auto rattleTolMsg =
            std::format("RATTLE Tolerance: {}", settings.get_rattleTolerance());

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
        const auto &settings = _engine.getSettings().constraintSettings;

        const auto shakeMaxIterMsg =
            std::format("SHAKE Max Iter:  {}", settings.get_shakeMaxIter());
        const auto rattleMaxIterMsg =
            std::format("RATTLE Max Iter: {}", settings.get_rattleMaxIter());

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
        auto &constraints = _engine.get_constraints();
        auto &simBox      = _engine.get_simulationBox();

        const auto nShakeBonds  = constraints.getNumberOfBondConstraints();
        const auto nMShakeTypes = constraints.getMShakeReferences().size();
        const auto nMShakeMols = constraints.getNumberOfMShakeMolecules(simBox);
        const auto nMShakeBonds =
            constraints.getNumberOfMShakeConstraints(simBox);

        const auto nShakeBondsMsg =
            std::format("Number of SHAKE bonds:       {}", nShakeBonds);
        const auto nMShakeTypesMsg =
            std::format("Number of M-SHAKE types:     {}", nMShakeTypes);
        const auto nMShakeMolsMsg =
            std::format("Number of M-SHAKE molecules: {}", nMShakeMols);
        const auto nMShakeBondsMsg =
            std::format("Number of M-SHAKE bonds:     {}", nMShakeBonds);

        auto &logOutput = _engine.getLogOutput();

        logOutput.writeSetupInfo(nShakeBondsMsg);
        logOutput.writeSetupInfo(nMShakeTypesMsg);
        logOutput.writeSetupInfo(nMShakeMolsMsg);
        logOutput.writeSetupInfo(nMShakeBondsMsg);
        logOutput.writeEmptyLine();
    }

}   // namespace setup
