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

#ifndef BOX__TRICLINIC_BOX_HPP
#define BOX__TRICLINIC_BOX_HPP

#include <memory>
#include <mstd/types.hpp>

#include "box.hpp"
#include "vector3d.hpp"

struct TestTriclinicBox;

namespace molsys
{
    std::pair<linalg::Vec3D, Angles3D> calcBoxDimAndAnglesFromBoxMatrix(
        const linalg::tensor3D &
    );

    class TriclinicBox;

    /**
     * @class TriclinicBox
     *
     * @brief This class represents the unit cell of a triclinic box
     *
     */
    class TriclinicBox : public Box
    {
       private:
        friend struct ::TestTriclinicBox;
        using Self = TriclinicBox;

        template <typename T, auto... Members>
        using BoxCache = mstd::MemberCache<T, TriclinicBox, Members...>;
        template <auto... Members>
        using TensorCache = BoxCache<linalg::tensor3D, Members...>;

        TensorCache<_anglesPtr()> _trafoMatrix{_calculateTransformationMatrix};

        static constexpr auto        _trafoMatrixPtr = &Self::_trafoMatrix;
        TensorCache<_trafoMatrixPtr> _invTrafoMatrix{_calculateInverse};
        TensorCache<_dimsPtr(), _trafoMatrixPtr> _boxMatrix{
            _calculateBoxMatrix
        };

        static constexpr auto           _boxMatrixPtr = &Self::_boxMatrix;
        TensorCache<_boxMatrixPtr>      _inverse{_calculateInverse};
        BoxCache<double, _boxMatrixPtr> _volume{_calculateVolume};

        bool _allowChangeOfAngles;

       public:
        TriclinicBox(
            const linalg::Vec3D &boxDimensions,
            const linalg::Vec3D &boxAngles,
            bool                 allowChangeOfAngles
        );

        [[nodiscard]]
        std::unique_ptr<Box> clone() const override;

        [[nodiscard]]
        linalg::Vec3D calcShiftVector(const linalg::Vec3D &vec) const override;

        [[nodiscard]]
        linalg::Vec3D toOrthoSpace(const linalg::Vec3D &vec) const override;
        [[nodiscard]]
        linalg::tensor3D toOrthoSpace(
            const linalg::tensor3D &mat
        ) const override;

        [[nodiscard]]
        linalg::Vec3D toSimSpace(const linalg::Vec3D &vec) const override;
        [[nodiscard]]
        linalg::tensor3D toSimSpace(const linalg::tensor3D &mat) const override;

        void applyPBC(linalg::Vec3D &position) const override;
        void scaleBox(const linalg::tensor3D &scalingTensor) override;

        [[nodiscard]] double getMinimalBoxDimension() const override;
        [[nodiscard]] double getVolume() const override;

        [[nodiscard]]
        linalg::tensor3D getBoxMatrix(Periodicity periodicity) const override;
        [[nodiscard]]
        linalg::Vec3D wrapPositionIntoBox(
            const linalg::Vec3D &pos
        ) const override;

        [[nodiscard]]
        bool allowChangeOfAngles() const override;

       private:
        [[nodiscard]]
        static linalg::tensor3D _calculateBoxMatrix(
            const linalg::Vec3D    &boxDimensions,
            const linalg::tensor3D &transformationMatrix
        );

        [[nodiscard]]
        static linalg::tensor3D _calculateTransformationMatrix(
            const Angles3D &boxAngles
        );

        [[nodiscard]]
        static linalg::tensor3D _calculateInverse(
            const linalg::tensor3D &boxMatrix
        );

        [[nodiscard]]
        static double _calculateVolume(const linalg::tensor3D &boxMatrix);

        [[nodiscard]]
        static double _cosAlpha(const Angles3D &boxAngles);
        [[nodiscard]]
        static double _cosBeta(const Angles3D &boxAngles);
        [[nodiscard]]
        static double _cosGamma(const Angles3D &boxAngles);
        [[nodiscard]]
        static double _sinAlpha(const Angles3D &boxAngles);
        [[nodiscard]]
        static double _sinBeta(const Angles3D &boxAngles);
        [[nodiscard]]
        static double _sinGamma(const Angles3D &boxAngles);
    };

}   // namespace molsys

#endif   // BOX__TRICLINIC_BOX_HPP
