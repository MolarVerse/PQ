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

#include <cstddef>      // for size_t
#include <deque>        // for std::queue
#include <functional>   // for std::function
#include <memory>       // for std::shared_ptr
#include <set>          // for std::set
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "staticMatrix3x3Class.hpp"
#include "vector3d.hpp"

namespace simulationBox
{
    class Molecule;        // forward declaration
    class MoleculeType;    // forward declaration
    class Atom;            // forward declaration
    class SimulationBox;   // forward declaration
    class CellList;        // forward declaration
    class Box;             // forward declaration

}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration

}   // namespace physicalData

namespace potential
{
    class Potential;             // forward declaration
    class PotentialBruteForce;   // forward declaration
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration

}   // namespace potential

namespace virial
{
    class Virial;            // forward declaration
    class VirialMolecular;   // forward declaration

}   // namespace virial

namespace intraNonBonded
{
    class IntraNonBonded;   // forward declaration

}   // namespace intraNonBonded

namespace forceField
{
    class ForceField;   // forward declaration

}   // namespace forceField

namespace constraints
{
    class Constraints;          // forward declaration
    class BondConstraint;       // forward declaration
    class DistanceConstraint;   // forward declaration
    class MShakeReference;      // forward declaration

}   // namespace constraints

namespace opt
{
    class LearningRateStrategy;
    class ConstantLRStrategy;
    class ConstantDecayLRStrategy;

    class Evaluator;
    class MMEvaluator;

    class Optimizer;
    class SteepestDescent;

}   // namespace opt

namespace output
{
    class RingPolymerRestartFileOutput;
    class RingPolymerTrajectoryOutput;
    class RingPolymerEnergyOutput;

}   // namespace output

namespace engine
{
    class Engine;       // forward declaration
    class QMMDEngine;   // forward declaration

}   // namespace engine

namespace pq
{
    using strings   = std::vector<std::string>;
    using stringSet = std::set<std::string>;

    using stlVector3d     = std::vector<std::vector<std::vector<double>>>;
    using stlVector4d     = std::vector<stlVector3d>;
    using stlVector3dBool = std::vector<std::vector<std::vector<bool>>>;
    using stlVector4dBool = std::vector<stlVector3dBool>;

    using ParseFunc = std::function<void(const strings &, const size_t)>;

    using Vec3D         = linearAlgebra::Vec3D;
    using Vec3DPair     = std::pair<Vec3D, Vec3D>;
    using Vec3DVec      = std::vector<Vec3D>;
    using Vec3DVecDeque = std::deque<std::vector<Vec3D>>;
    using tensor3D      = linearAlgebra::tensor3D;

    using Virial          = virial::Virial;
    using VirialMolecular = virial::VirialMolecular;
    using IntraNonBond    = intraNonBonded::IntraNonBonded;
    using ForceField      = forceField::ForceField;

    using UniqueEngine       = std::unique_ptr<Engine>;
    using SharedIntraNonBond = std::shared_ptr<intraNonBonded::IntraNonBonded>;
    using SharedForceField   = std::shared_ptr<forceField::ForceField>;
    using SharedConstraints  = std::shared_ptr<constraints::Constraints>;
    using SharedVirial       = std::shared_ptr<virial::Virial>;

    /********************
     * engine namespace *
     ********************/

    using Engine     = engine::Engine;
    using QMMDEngine = engine::QMMDEngine;

    /***********************
     * potential namespace *
     ***********************/

    using Potential     = potential::Potential;
    using BruteForcePot = potential::PotentialBruteForce;
    using CoulombPot    = potential::CoulombPotential;
    using NonCoulombPot = potential::NonCoulombPotential;

    using SharedPotential     = std::shared_ptr<potential::Potential>;
    using SharedCoulombPot    = std::shared_ptr<potential::CoulombPotential>;
    using SharedNonCoulombPot = std::shared_ptr<potential::NonCoulombPotential>;

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
    using CellList     = simulationBox::CellList;
    using Molecule     = simulationBox::Molecule;
    using MoleculeType = simulationBox::MoleculeType;
    using Atom         = simulationBox::Atom;
    using Box          = simulationBox::Box;

    using SharedAtom     = std::shared_ptr<simulationBox::Atom>;
    using SharedSimBox   = std::shared_ptr<simulationBox::SimulationBox>;
    using SharedCellList = std::shared_ptr<simulationBox::CellList>;

    /**************************
     * physicalData namespace *
     **************************/

    using PhysicalData       = physicalData::PhysicalData;
    using VecPhysicalData    = std::vector<PhysicalData>;
    using SharedPhysicalData = std::shared_ptr<physicalData::PhysicalData>;

    /*****************
     * opt namespace *
     *****************/

    using Evaluator       = opt::Evaluator;
    using MMEvaluator     = opt::MMEvaluator;
    using SharedEvaluator = std::shared_ptr<opt::Evaluator>;

    using LearningRate       = opt::LearningRateStrategy;
    using ConstantLR         = opt::ConstantLRStrategy;
    using ConstantDecayLR    = opt::ConstantDecayLRStrategy;
    using SharedLearningRate = std::shared_ptr<opt::LearningRateStrategy>;

    using Optimizer       = opt::Optimizer;
    using SteepestDescent = opt::SteepestDescent;
    using SharedOptimizer = std::shared_ptr<opt::Optimizer>;

    /********************
     * output namespace *
     ********************/

    using RPMDRstFileOutput = output::RingPolymerRestartFileOutput;
    using RPMDTrajOutput    = output::RingPolymerTrajectoryOutput;
    using RPMDEnergyOutput  = output::RingPolymerEnergyOutput;

    using UniqueRPMDRstFileOutput = std::unique_ptr<RPMDRstFileOutput>;
    using UniqueRPMDTrajOutput    = std::unique_ptr<RPMDTrajOutput>;
    using UniqueRPMDEnergyOutput  = std::unique_ptr<RPMDEnergyOutput>;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_