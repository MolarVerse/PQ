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

#ifndef _M_SHAKE_HPP_

#define _M_SHAKE_HPP_

#include <vector>   // for vector

#include "mShakeReference.hpp"   // for MShakeReference
#include "matrix.hpp"            // for Matrix
#include "vector3d.hpp"          // for Vec3D

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace constraints
{

    using SimBox = simulationBox::SimulationBox;

    class MShake
    {
       private:
        std::vector<MShakeReference>                   _mShakeReferences;
        std::vector<std::vector<double>>               _mShakeRSquaredRefs;
        std::vector<std::vector<linearAlgebra::Vec3D>> _posBeforeIntegration;

        std::vector<linearAlgebra::Matrix<double>> _mShakeMatrices;
        std::vector<linearAlgebra::Matrix<double>> _mShakeInvMatrices;

       public:
        MShake()  = default;
        ~MShake() = default;

        void initMShake(SimBox &);
        void initMShakeReferences();
        void initPosBeforeIntegration(SimBox &);
        void applyMShake(const double, SimBox &);
        void applyMRattle(const double, SimBox &);

        [[nodiscard]] size_t calculateNumberOfBondConstraints(SimBox &) const;

        [[nodiscard]] double calculateShakeMatrixElement(
            const size_t               i,
            const size_t               j,
            const size_t               k,
            const size_t               l,
            const double               mass_i,
            const double               mass_j,
            const linearAlgebra::Vec3D pos_ij,
            const linearAlgebra::Vec3D pos_kl
        );

        bool isMShakeType(const size_t moltype) const;

        [[nodiscard]] const MShakeReference &findMShakeReference(const size_t
        ) const;
        [[nodiscard]] size_t findMShakeReferenceIndex(const size_t) const;

        [[nodiscard]] const std::vector<MShakeReference> &getMShakeReferences(
        ) const;

        void addMShakeReference(const MShakeReference &mShakeReference);
    };
}   // namespace constraints

#endif   // _M_SHAKE_HPP_