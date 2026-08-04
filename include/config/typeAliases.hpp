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

#ifndef _TYPE_ALIASES_HPP_

#define _TYPE_ALIASES_HPP_

#include <memory>     // for std::shared_ptr
#include <optional>   // for std::optional
#include <vector>     // for std::vector

namespace simulationBox
{
    class Molecule;              // forward declaration
    class MoleculeType;          // forward declaration
    class Atom;                  // forward declaration
    class CellList;              // forward declaration
    class Box;                   // forward declaration
    class SimulationBox;         // forward declaration
    class KokkosSimulationBox;   // forward declaration

}   // namespace simulationBox

namespace potential
{
    class Potential;              // forward declaration
    class PotentialBruteForce;    // forward declaration
    class CoulombPotential;       // forward declaration
    class NonCoulombPair;         // forward declaration
    class NonCoulombPotential;    // forward declaration
    class ForceFieldNonCoulomb;   // forward declaration

    class KokkosLennardJones;   // forward declaration
    class KokkosCoulombWolf;    // forward declaration
    class KokkosPotential;      // forward declaration

}   // namespace potential

namespace constraints
{
    class Constraints;          // forward declaration
    class BondConstraint;       // forward declaration
    class DistanceConstraint;   // forward declaration
    class MShakeReference;      // forward declaration

}   // namespace constraints

namespace engine
{
    class Engine;              // forward declaration
    class MDEngine;            // forward declaration
    class OptEngine;           // forward declaration
    class HessianEngine;       // forward declaration
    class QMMDEngine;          // forward declaration
    class QMMMMDEngine;        // forward declaration
    class RingPolymerEngine;   // forward declaration

}   // namespace engine

namespace settings
{
    enum class ThermostatType;   // forward declaration
    enum class ManostatType;     // forward declaration
    enum class Isotropy;         // forward declaration

}   // namespace settings

namespace integrator
{
    class Integrator;       // forward declaration
    class VelocityVerlet;   // forward declaration
}   // namespace integrator

namespace input
{
    namespace parameterFile
    {
        class ParameterFileSection;   // forward declaration
    }

    namespace restartFile
    {
        class RestartFileSection;   // forward declaration
    }
}   // namespace input

namespace pq
{
    using SharedConstraints = std::shared_ptr<constraints::Constraints>;

    using ParamFileSection       = input::parameterFile::ParameterFileSection;
    using UniqueParamFileSection = std::unique_ptr<ParamFileSection>;
    using UniqueParamFileSectionVec = std::vector<UniqueParamFileSection>;

    using RestartSection          = input::restartFile::RestartFileSection;
    using UniqueRestartSection    = std::unique_ptr<RestartSection>;
    using UniqueRestartSectionVec = std::vector<UniqueRestartSection>;

    /************************
     * integrator namespace *
     ************************/

    using Integrator     = integrator::Integrator;
    using VelocityVerlet = integrator::VelocityVerlet;

    using UniqueIntegrator = std::unique_ptr<Integrator>;

    /**********************
     * settings namespace *
     **********************/

    using ThermostatType = settings::ThermostatType;
    using Isotropy       = settings::Isotropy;
    using ManostatType   = settings::ManostatType;

    /********************
     * engine namespace *
     ********************/

    using Engine            = engine::Engine;
    using MDEngine          = engine::MDEngine;
    using OptEngine         = engine::OptEngine;
    using HessianEngine     = engine::HessianEngine;
    using QMMDEngine        = engine::QMMDEngine;
    using QMMMMDEngine      = engine::QMMMMDEngine;
    using RingPolymerEngine = engine::RingPolymerEngine;

    using UniqueEngine = std::unique_ptr<Engine>;

    /***********************
     * potential namespace *
     ***********************/

    using Potential     = potential::Potential;
    using BruteForcePot = potential::PotentialBruteForce;
    using CoulombPot    = potential::CoulombPotential;
    using NonCoulombPot = potential::NonCoulombPotential;
    using FFNonCoulomb  = potential::ForceFieldNonCoulomb;
    using NonCoulPair   = potential::NonCoulombPair;

    using KokkosLJ        = potential::KokkosLennardJones;
    using KokkosWolf      = potential::KokkosCoulombWolf;
    using KokkosPotential = potential::KokkosPotential;

    using SharedPotential     = std::shared_ptr<potential::Potential>;
    using SharedCoulombPot    = std::shared_ptr<potential::CoulombPotential>;
    using SharedNonCoulombPot = std::shared_ptr<potential::NonCoulombPotential>;
    using SharedNonCoulPair   = std::shared_ptr<potential::NonCoulombPair>;

    using OptSharedNonCoulPair = std::optional<SharedNonCoulPair>;

    using SharedNonCoulPairVec   = std::vector<SharedNonCoulPair>;
    using SharedNonCoulPairVec2d = std::vector<SharedNonCoulPairVec>;
    using SharedNonCoulPairVec3d = std::vector<SharedNonCoulPairVec2d>;
    using SharedNonCoulPairVec4d = std::vector<SharedNonCoulPairVec3d>;

    /**************************
     * constraints namespace *
     **************************/

    using Constraints        = constraints::Constraints;
    using BondConstraint     = constraints::BondConstraint;
    using MShakeReference    = constraints::MShakeReference;
    using MShakeRef          = constraints::MShakeReference;
    using DistanceConstraint = constraints::DistanceConstraint;

    using BondConstraintsVec = std::vector<BondConstraint>;
    using MShakeReferenceVec = std::vector<MShakeReference>;
    using MShakeRefVec       = std::vector<MShakeReference>;
    using DistConstraintsVec = std::vector<DistanceConstraint>;

    /***************************
     * simulationBox namespace *
     ***************************/

    using SimBox       = simulationBox::SimulationBox;
    using KokkosSimBox = simulationBox::KokkosSimulationBox;
    using CellList     = simulationBox::CellList;
    using Molecule     = simulationBox::Molecule;
    using MoleculeType = simulationBox::MoleculeType;
    using Atom         = simulationBox::Atom;
    using Box          = simulationBox::Box;

    using SharedAtom     = std::shared_ptr<simulationBox::Atom>;
    using SharedSimBox   = std::shared_ptr<simulationBox::SimulationBox>;
    using SharedCellList = std::shared_ptr<simulationBox::CellList>;
    using SharedBox      = std::shared_ptr<simulationBox::Box>;

    using SharedAtomVec = std::vector<SharedAtom>;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_
