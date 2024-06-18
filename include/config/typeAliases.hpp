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
#include <functional>   // for std::function
#include <memory>       // for std::shared_ptr
#include <string>       // for std::string
#include <vector>       // for std::vector

#include "staticMatrix3x3Class.hpp"
#include "vector3d.hpp"

namespace simulationBox
{
    class Molecule;        // forward declaration
    class Atom;            // forward declaration
    class SimulationBox;   // forward declaration
    class CellList;        // forward declaration

}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration

}   // namespace physicalData

namespace potential
{
    class Potential;             // forward declaration
    class PotentialBruteForce;   // forward declaration

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
    class Constraints;   // forward declaration

}   // namespace constraints

namespace opt
{
    class LearningRateStrategy;
    class Evaluator;
    class Optimizer;

}   // namespace opt

namespace pq
{
    using strings = std::vector<std::string>;

    using Vec3D     = linearAlgebra::Vec3D;
    using Vec3DPair = std::pair<Vec3D, Vec3D>;
    using tensor3D  = linearAlgebra::tensor3D;

    using SimBox          = simulationBox::SimulationBox;
    using CellList        = simulationBox::CellList;
    using Molecule        = simulationBox::Molecule;
    using Atom            = simulationBox::Atom;
    using Virial          = virial::Virial;
    using VirialMolecular = virial::VirialMolecular;
    using Potential       = potential::Potential;
    using BruteForcePot   = potential::PotentialBruteForce;
    using PhysicalData    = physicalData::PhysicalData;
    using IntraNonBond    = intraNonBonded::IntraNonBonded;
    using ForceField      = forceField::ForceField;
    using Constraints     = constraints::Constraints;
    using Evaluator       = opt::Evaluator;
    using LearningRate    = opt::LearningRateStrategy;
    using Optimizer       = opt::Optimizer;

    using SharedAtom         = std::shared_ptr<simulationBox::Atom>;
    using SharedSimBox       = std::shared_ptr<simulationBox::SimulationBox>;
    using SharedPhysicalData = std::shared_ptr<physicalData::PhysicalData>;
    using SharedCellList     = std::shared_ptr<simulationBox::CellList>;
    using SharedIntraNonBond = std::shared_ptr<intraNonBonded::IntraNonBonded>;
    using SharedForceField   = std::shared_ptr<forceField::ForceField>;
    using SharedConstraints  = std::shared_ptr<constraints::Constraints>;
    using SharedVirial       = std::shared_ptr<virial::Virial>;
    using SharedPotential    = std::shared_ptr<potential::Potential>;

}   // namespace pq

#endif   // _TYPE_ALIASES_HPP_