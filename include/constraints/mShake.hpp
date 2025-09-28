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
#include "typeAliases.hpp"       // for SimBox, Vec3D, MShakeRef
#include "vector3d.hpp"          // for Vec3D

namespace constraints
{

    /**
     * @class MShake
     *
     * @brief class containing all information about the mShake algorithm
     *
     * @details it performs the mShake algorithm on all bond constraints
     */
    class MShake
    {
       private:
        pq::MShakeRefVec                 _mShakeReferences;
        std::vector<std::vector<double>> _mShakeRSquaredRefs;

        std::vector<linearAlgebra::Matrix<double>> _mShakeMatrices;
        std::vector<linearAlgebra::Matrix<double>> _mShakeInvMatrices;

       public:
        MShake()  = default;
        ~MShake() = default;

        void initMShake();
        void initMShakeReferences();
        void applyMShake(const double, pq::SimBox &);
        void applyMRattle(pq::SimBox &);

        [[nodiscard]] size_t calcNumberOfMShakeMolecules(pq::SimBox &) const;
        [[nodiscard]] size_t calcNumberOfBondConstraints(pq::SimBox &) const;
        [[nodiscard]] double calcMatrixElement(
            const std::tuple<size_t, size_t, size_t, size_t> &indices,
            const std::pair<double, double>                  &masses,
            const std::pair<pq::Vec3D, pq::Vec3D>            &pos
        ) const;

        [[nodiscard]] bool   isMShakeType(const size_t moltype) const;
        [[nodiscard]] size_t findMShakeReferenceIndex(const size_t) const;
        [[nodiscard]] const pq::MShakeRef    &findMShakeRef(const size_t) const;
        [[nodiscard]] const pq::MShakeRefVec &getMShakeReferences() const;

        void addMShakeReference(const pq::MShakeRef &mShakeReference);
    };
}   // namespace constraints

#endif   // _M_SHAKE_HPP_