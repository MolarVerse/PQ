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

#ifndef _EXPECT_THROW_WITH_MESSAGE_HPP_

#define _EXPECT_THROW_WITH_MESSAGE_HPP_

/**
 * @macro EXPECT_THROW_MSG
 *
 * @brief expects that a statement throws an exception of a given type with a
 * given message
 *
 */
#define EXPECT_THROW_MSG(statement, expected_exception, expected_what) \
    try                                                                \
    {                                                                  \
        statement;                                                     \
        FAIL() << "Expected: " #statement                              \
                  " throws an exception of type " #expected_exception  \
                  ".\n"                                                \
                  "  Actual: it throws nothing.";                      \
    }                                                                  \
    catch (const expected_exception &e)                                \
    {                                                                  \
        EXPECT_EQ(expected_what, std::string{e.what()});               \
    }                                                                  \
    catch (...)                                                        \
    {                                                                  \
        FAIL() << "Expected: " #statement                              \
                  " throws an exception of type " #expected_exception  \
                  ".\n"                                                \
                  "  Actual: it throws a different type.";             \
    }

/**
 * @macro ASSERT_THROW_MSG
 *
 * @brief expects that a statement throws an exception of a given type with a
 * given message
 *
 */
#define ASSERT_THROW_MSG(statement, expected_exception, expected_what) \
    EXPECT_THROW_MSG(statement, expected_exception, expected_what)

#endif   // _EXPECT_THROW_WITH_MESSAGE_HPP_
