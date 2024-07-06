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

#include <chrono>       // for std::chrono
#include <cstddef>      // for size_t
#include <deque>        // for std::queue
#include <functional>   // for std::function
#include <memory>       // for std::shared_ptr
#include <optional>     // for std::optional
#include <set>          // for std::set
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "matrix.hpp"
#include "staticMatrix.hpp"
#include "vector3d.hpp"

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

namespace physicalData
{
    class PhysicalData;   // forward declaration

}   // namespace physicalData

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

}   // namespace potential

namespace virial
{
    class Virial;            // forward declaration
    class MolecularVirial;   // forward declaration

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

    class Convergence;

}   // namespace opt

namespace output
{
    class RingPolymerRestartFileOutput;
    class RingPolymerTrajectoryOutput;
    class RingPolymerEnergyOutput;

}   // namespace output

namespace engine
{
    class Engine;              // forward declaration
    class MDEngine;            // forward declaration
    class OptEngine;           // forward declaration
    class QMMDEngine;          // forward declaration
    class QMMMMDEngine;        // forward declaration
    class RingPolymerEngine;   // forward declaration

}   // namespace engine

namespace manostat
{
    class StochasticRescalingManostat;                  // forward declaration
    class SemiIsotropicStochasticRescalingManostat;     // forward declaration
    class AnisotropicStochasticRescalingManostat;       // forward declaration
    class FullAnisotropicStochasticRescalingManostat;   // forward declaration

}   // namespace manostat

namespace timings
{
    class Timer;
    class GlobalTimer;

}   // namespace timings

namespace thermostat
{
    class Thermostat;   // forward declaration
    class NoseHoover;   // forward declaration

}   // namespace thermostat

namespace settings
{
    enum class ThermostatType;   // forward declaration
    enum class ManostatType;
    enum class Isotropy;

}   // namespace settings

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
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;

    using strings   = std::vector<std::string>;
    using stringSet = std::set<std::string>;

    using stlVectorUL     = std::vector<size_t>;
    using stlVector3d     = std::vector<std::vector<std::vector<double>>>;
    using stlVector4d     = std::vector<stlVector3d>;
    using stlVector3dBool = std::vector<std::vector<std::vector<bool>>>;
    using stlVector4dBool = std::vector<stlVector3dBool>;

    using ParseFunc = std::function<void(const strings &, const size_t)>;

    using Vec3D         = linearAlgebra::Vec3D;
    using Vec3Dul       = linearAlgebra::Vec3Dul;
    using Vec3DPair     = std::pair<Vec3D, Vec3D>;
    using Vec3DVec      = std::vector<Vec3D>;
    using Vec3DVecDeque = std::deque<std::vector<Vec3D>>;
    using tensor3D      = linearAlgebra::tensor3D;

    using IntraNonBond = intraNonBonded::IntraNonBonded;
    using ForceField   = forceField::ForceField;
    using Timer        = timings::Timer;
    using GlobalTimer  = timings::GlobalTimer;

    using SharedIntraNonBond = std::shared_ptr<intraNonBonded::IntraNonBonded>;
    using SharedForceField   = std::shared_ptr<forceField::ForceField>;
    using SharedConstraints  = std::shared_ptr<constraints::Constraints>;

    using ParamFileSection       = input::parameterFile::ParameterFileSection;
    using UniqueParamFileSection = std::unique_ptr<ParamFileSection>;
    using UniqueParamFileSectionVec = std::vector<UniqueParamFileSection>;

    using RestartSection          = input::restartFile::RestartFileSection;
    using UniqueRestartSection    = std::unique_ptr<RestartSection>;
    using UniqueRestartSectionVec = std::vector<UniqueRestartSection>;

    /**********************
     * settings namespace *
     **********************/

    using ThermostatType = settings::ThermostatType;
    using Isotropy       = settings::Isotropy;
    using ManostatType   = settings::ManostatType;

    /************************
     * thermostat namespace *
     ************************/

    using Thermostat = thermostat::Thermostat;
    using NoseHoover = thermostat::NoseHoover;

    /********************
     * virial namespace *
     ********************/

    using Virial          = virial::Virial;
    using MolecularVirial = virial::MolecularVirial;

    using SharedVirial = std::shared_ptr<virial::Virial>;

    /**********************
     * manostat namespace *
     **********************/

    // clang-format off
    using StochasticManostat          = manostat::StochasticRescalingManostat;
    using SemiIsoStochasticManostat   = manostat::SemiIsotropicStochasticRescalingManostat;
    using AnisoStochasticManostat     = manostat::AnisotropicStochasticRescalingManostat;
    using FullAnisoStochasticManostat = manostat::FullAnisotropicStochasticRescalingManostat;
    // clang-format on

    /********************
     * engine namespace *
     ********************/

    using Engine            = engine::Engine;
    using MDEngine          = engine::MDEngine;
    using OptEngine         = engine::OptEngine;
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

    using KokkosLJ   = potential::KokkosLennardJones;
    using KokkosWolf = potential::KokkosCoulombWolf;

    using SharedPotential     = std::shared_ptr<potential::Potential>;
    using SharedCoulombPot    = std::shared_ptr<potential::CoulombPotential>;
    using SharedNonCoulombPot = std::shared_ptr<potential::NonCoulombPotential>;
    using SharedNonCoulPair   = std::shared_ptr<potential::NonCoulombPair>;

    using OptSharedNonCoulPair = std::optional<SharedNonCoulPair>;

    using SharedNonCoulPairVec   = std::vector<SharedNonCoulPair>;
    using SharedNonCoulPairVec2d = std::vector<SharedNonCoulPairVec>;
    using SharedNonCoulPairVec3d = std::vector<SharedNonCoulPairVec2d>;
    using SharedNonCoulPairVec4d = std::vector<SharedNonCoulPairVec3d>;
    using SharedNonCoulPairMat   = linearAlgebra::Matrix<SharedNonCoulPair>;

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

    using Convergence = opt::Convergence;

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