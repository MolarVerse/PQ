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

#include "testUtils.hpp"

#include <memory>

#include "coulombPotential.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "dftbplusRunner.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "engine.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "nonCoulombPotential.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "pyscfRunner.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "qmRunner.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation
#include "turbomoleRunner.hpp"   // IWYU pragma: keep -- needed for explicit template instantiation

namespace test
{
    /**
     * @brief check that the dynamic type of obj matches expectedType
     *
     * @details Works for raw pointers, smart pointers, and plain
     * references alike — dereferences anything pointer-like before
     * comparing typeid, so typeid always reflects the pointee's
     * actual (polymorphic) type rather than the pointer/wrapper type.
     *
     * @tparam T
     * @param obj
     * @param expectedType
     */
    template <typename T>
    void checkType(const T& obj, const std::type_info& expectedType)
    {
        if constexpr (requires { *obj; })
            EXPECT_EQ(typeid(*obj), expectedType);
        else
            EXPECT_EQ(typeid(obj), expectedType);
    }

    // explicit instantiations
    template void checkType<std::unique_ptr<engine::Engine>>(
        const std::unique_ptr<engine::Engine>& engine,
        const std::type_info&                  expectedType
    );
    template void checkType<potential::Potential*>(
        potential::Potential* const& potential,
        const std::type_info&        expectedType
    );
    template void checkType<potential::CoulombPotential*>(
        potential::CoulombPotential* const& potential,
        const std::type_info&               expectedType
    );
    template void checkType<potential::NonCoulombPotential*>(
        potential::NonCoulombPotential* const& potential,
        const std::type_info&                  expectedType
    );

    // QM runners
    template void checkType<QM::QMRunner>(
        QM::QMRunner const&   runner,
        const std::type_info& expectedType
    );
    template void checkType<QM::DFTBPlusRunner>(
        QM::DFTBPlusRunner const& runner,
        const std::type_info&     expectedTypech
    );
    template void checkType<QM::PySCFRunner>(
        QM::PySCFRunner const& runner,
        const std::type_info&  expectedType
    );
    template void checkType<QM::TurbomoleRunner>(
        QM::TurbomoleRunner const& runner,
        const std::type_info&      expectedType
    );

}   // namespace test
