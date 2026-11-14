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
#include "unit_types.hpp"
#include "vector3d.hpp"

struct TestTriclinicBox;

namespace molsys
{
    std::pair<Pos3D, Angles3D> calcBoxDimAndAnglesFromBoxMatrix(
        const PosTensor3D &
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
        using InvPosTensorCache = BoxCache<InvPosTensor3D, Members...>;
        template <auto... Members>
        using PosTensorCache = BoxCache<PosTensor3D, Members...>;
        template <auto... Members>
        using QTensorCache = BoxCache<QuantityTensor3D, Members...>;

        QTensorCache<_anglesPtr()> _trafoMatrix{_calculateTransformationMatrix};

        static constexpr auto         _trafoMatrixPtr = &Self::_trafoMatrix;
        QTensorCache<_trafoMatrixPtr> _invTrafoMatrix{_calculateInverse};
        PosTensorCache<_dimsPtr(), _trafoMatrixPtr> _boxMatrix{
            _calculateBoxMatrix
        };

        static constexpr auto            _boxMatrixPtr = &Self::_boxMatrix;
        InvPosTensorCache<_boxMatrixPtr> _inverse{_calculateInverse};
        BoxCache<Volume, _boxMatrixPtr>  _volume{_calculateVolume};

        bool _allowChangeOfAngles;

       public:
        TriclinicBox(
            const Pos3D         &boxDimensions,
            const linalg::Vec3D &boxAngles,
            bool                 allowChangeOfAngles
        );

        [[nodiscard]]
        std::unique_ptr<Box> clone() const override;

        [[nodiscard]]
        Pos3D calcShiftVector(const Pos3D &vec) const override;

        void applyPBC(Pos3D &position) const override;
        void scaleBox(const QuantityTensor3D &scalingTensor) override;

        [[nodiscard]] Pos    getMinimalBoxDimension() const override;
        [[nodiscard]] Volume getVolume() const override;

        [[nodiscard]]
        PosTensor3D getBoxMatrix(Periodicity periodicity) const override;

        [[nodiscard]]
        PosTensor3D getBoxMatrix() const override;

        [[nodiscard]]
        InvPosTensor3D getInverseBoxMatrix() const override;

        [[nodiscard]]
        Pos3D wrapPositionIntoBox(const Pos3D &pos) const override;

        [[nodiscard]]
        bool allowChangeOfAngles() const override;

        [[nodiscard]]
        bool isOrthorhombic() const override;

       private:
        [[nodiscard]]
        static PosTensor3D _calculateBoxMatrix(
            const Pos3D            &boxDimensions,
            const QuantityTensor3D &transformationMatrix
        );

        [[nodiscard]]
        static QuantityTensor3D _calculateTransformationMatrix(
            const Angles3D &boxAngles
        );

        [[nodiscard]]
        static QuantityTensor3D _calculateInverse(
            const QuantityTensor3D &boxMatrix
        );
        static InvPosTensor3D _calculateInverse(const PosTensor3D &boxMatrix);

        [[nodiscard]]
        static Volume _calculateVolume(const PosTensor3D &boxMatrix);

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
