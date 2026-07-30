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

#include "engine.hpp"

namespace test
{
    /**
     * @brief check if engine is of expected type
     *
     * @param engine
     * @param expectedType
     */
    void checkEngineType(
        const std::unique_ptr<engine::Engine>& engine,
        const std::type_info&                  expectedType
    )
    {
        ASSERT_NE(engine, nullptr);
        const auto& engineRef = *engine;
        EXPECT_EQ(typeid(engineRef), expectedType);
    }

    /**
     * @brief check if potential is of expected type
     *
     * @param potential
     * @param expectedType
     */
    void checkPotentialType(
        const potential::Potential* potential,
        const std::type_info&       expectedType
    )
    {
        ASSERT_NE(potential, nullptr);
        EXPECT_EQ(typeid(*potential), expectedType);
    }
}   // namespace test
