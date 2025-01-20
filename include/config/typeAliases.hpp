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
#include <deque>        // IWYU pragma: keep
#include <functional>   // for std::function
#include <memory>       // for std::shared_ptr
#include <optional>     // for std::optional
#include <set>          // for std::set
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "linearAlgebra.hpp"   // IWYU pragma: keep

#if defined(__SINGLE_PRECISION__)
using Real  = float;
using Realm = float;
#elif defined(__MIXED_PRECISION__)
using Real  = double;
using Realm = float;
#else
using Real  = double;
using Realm = double;
#endif

using cul = const size_t;

namespace simulationBox
{
    class Molecule;        // forward declaration
    class MoleculeType;    // forward declaration
    class Atom;            // forward declaration
    class CellList;        // forward declaration
    class Box;             // forward declaration
    class SimulationBox;   // forward declaration

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
    class EnergyOutput;       // forward declaration
    class InfoOutput;         // forward declaration
    class LogOutput;          // forward declaration
    class RstFileOutput;      // forward declaration
    class StdoutOutput;       // forward declaration
    class TrajectoryOutput;   // forward declaration
    class MomentumOutput;     // forward declaration
    class VirialOutput;       // forward declaration
    class StressOutput;       // forward declaration
    class BoxFileOutput;      // forward declaration
    class TimingsOutput;      // forward declaration
    class OptOutput;          // forward declaration

    class RingPolymerRestartFileOutput;   // forward declaration
    class RingPolymerTrajectoryOutput;    // forward declaration
    class RingPolymerEnergyOutput;        // forward declaration

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
    class Manostat;   // forward declaration

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
    enum class ManostatType;     // forward declaration
    enum class Isotropy;         // forward declaration

}   // namespace settings

namespace integrator
{
    class Integrator;       // forward declaration
    class VelocityVerlet;   // forward declaration
}   // namespace integrator

namespace resetKinetics
{
    class ResetKinetics;   // forward declaration
}   // namespace resetKinetics

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

namespace device
{
    class Device;   // forward declaration
}   // namespace device

namespace pq
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;

    using tupleReal4 = std::tuple<Real, Real, Real, Real>;

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

    using ResetKinetics = resetKinetics::ResetKinetics;

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

    /************************
     * thermostat namespace *
     ************************/

    using Thermostat = thermostat::Thermostat;
    using NoseHoover = thermostat::NoseHoover;

    using UniqueThermostat = std::unique_ptr<Thermostat>;

    /********************
     * virial namespace *
     ********************/

    using Virial          = virial::Virial;
    using MolecularVirial = virial::MolecularVirial;

    using SharedVirial = std::shared_ptr<virial::Virial>;

    /**********************
     * manostat namespace *
     **********************/

    using Manostat = manostat::Manostat;

    using UniqueManostat = std::unique_ptr<Manostat>;

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

    using SharedPotential     = std::shared_ptr<potential::Potential>;
    using SharedNonCoulPair   = std::shared_ptr<potential::NonCoulombPair>;
    using SharedCoulombPot    = std::shared_ptr<potential::CoulombPotential>;
    using SharedNonCoulombPot = std::shared_ptr<potential::NonCoulombPotential>;

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

    using EnergyOutput     = output::EnergyOutput;
    using InfoOutput       = output::InfoOutput;
    using LogOutput        = output::LogOutput;
    using RstFileOutput    = output::RstFileOutput;
    using StdoutOutput     = output::StdoutOutput;
    using TrajectoryOutput = output::TrajectoryOutput;
    using MomentumOutput   = output::MomentumOutput;
    using VirialOutput     = output::VirialOutput;
    using StressOutput     = output::StressOutput;
    using BoxFileOutput    = output::BoxFileOutput;
    using TimingsOutput    = output::TimingsOutput;
    using OptOutput        = output::OptOutput;

    using RPMDRstFileOutput = output::RingPolymerRestartFileOutput;
    using RPMDTrajOutput    = output::RingPolymerTrajectoryOutput;
    using RPMDEnergyOutput  = output::RingPolymerEnergyOutput;

    using UniqueRPMDRstFileOutput = std::unique_ptr<RPMDRstFileOutput>;
    using UniqueRPMDTrajOutput    = std::unique_ptr<RPMDTrajOutput>;
    using UniqueRPMDEnergyOutput  = std::unique_ptr<RPMDEnergyOutput>;

    /********************
     * device namespace *
     ********************/

    using Device = device::Device;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_