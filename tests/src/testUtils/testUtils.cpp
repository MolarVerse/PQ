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

#include "coulombPotential.hpp"
#include "dftbplusRunner.hpp"
#include "engine.hpp"
#include "nonCoulombPotential.hpp"
#include "pyscfRunner.hpp"
#include "qmRunner.hpp"
#include "turbomoleRunner.hpp"

namespace test
{
    // Explicit template instantiation below needs the complete type of
    // each of these (e.g. for typeid), not just a forward declaration.
    // These give the linter a direct use so the includes above aren't
    // mistaken for unused.
    [[maybe_unused]] constexpr std::size_t engineSize = sizeof(engine::Engine);
    [[maybe_unused]] constexpr std::size_t coulombPotentialSize =
        sizeof(pot::CoulombPotential);
    [[maybe_unused]] constexpr std::size_t nonCoulombPotentialSize =
        sizeof(pot::NonCoulombPotential);
    [[maybe_unused]] constexpr std::size_t qmRunnerSize = sizeof(QM::QMRunner);
    [[maybe_unused]] constexpr std::size_t dftbPlusRunnerSize =
        sizeof(QM::DFTBPlusRunner);
    [[maybe_unused]] constexpr std::size_t pyscfRunnerSize =
        sizeof(QM::PySCFRunner);
    [[maybe_unused]] constexpr std::size_t turbomoleRunnerSize =
        sizeof(QM::TurbomoleRunner);
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
    template void checkType<std::shared_ptr<pot::Potential>>(
        const std::shared_ptr<pot::Potential>& potential,
        const std::type_info&                  expectedType
    );
    template void checkType<pot::CoulombPotential*>(
        pot::CoulombPotential* const& potential,
        const std::type_info&         expectedType
    );
    template void checkType<pot::NonCoulombPotential*>(
        pot::NonCoulombPotential* const& potential,
        const std::type_info&            expectedType
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
