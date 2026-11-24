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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <mstd/enum.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "exceptions.hpp"
#include "inputConverter.hpp"
#include "inputParam.hpp"
#include "inputRegistry.hpp"
#include "throwWithMessage.hpp"

// A small MSTD_ENUM used purely to exercise the has_enum_meta<T> path of
// Converter<T> / input::InputKey<T> -- not a real project enum.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TESTENUM_LIST(X) \
    X(mm)                \
    X(qm)                \
    X(md)
MSTD_ENUM(TestJobType, std::uint8_t, TESTENUM_LIST);

// A second test enum with aliases for two of its values and a name that
// overrides an enumerator (as SlakosType does with "3ob").
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TESTALIASED_LIST(X) \
    X(NONE)                 \
    X(THREEOB)              \
    X(VERLET)               \
    X(OTHER)
MSTD_ENUM(TestAliased, std::uint8_t, TESTALIASED_LIST);

template <>
struct mstd::EnumAliases<TestAliased>
{
    static constexpr auto value = mstd::makeAliases<TestAliased>(
        {{"v_verlet", TestAliased::VERLET},
         {"VV", TestAliased::VERLET},
         {"nada", TestAliased::NONE}}
    );
};

template <>
struct mstd::EnumNames<TestAliased>
{
    static constexpr auto value =
        mstd::makeNames<TestAliased>({{TestAliased::THREEOB, "3ob"}});
};

/**
 * @brief tests Converter<double>::tryParse
 *
 */
TEST(TestConverter, tryParseDouble)
{
    EXPECT_EQ(input::Converter<double>::tryParse("1.5"), 1.5);
    EXPECT_EQ(input::Converter<double>::tryParse("not_a_number"), std::nullopt);
    EXPECT_EQ(input::Converter<double>::tryParse("1.5garbage"), std::nullopt);
}

/**
 * @brief tests Converter<bool>::tryParse
 *
 */
TEST(TestConverter, tryParseBool)
{
    EXPECT_EQ(input::Converter<bool>::tryParse("TRUE"), true);
    EXPECT_EQ(input::Converter<bool>::tryParse("ON"), true);
    EXPECT_EQ(input::Converter<bool>::tryParse("FALSE"), false);
    EXPECT_EQ(input::Converter<bool>::tryParse("OFF"), false);
    EXPECT_EQ(input::Converter<bool>::tryParse("maybe"), std::nullopt);
}

/**
 * @brief tests Converter<T> generic integral fallback
 *
 */
TEST(TestConverter, tryParseIntegral)
{
    EXPECT_EQ(input::Converter<size_t>::tryParse("42"), size_t{42});
    EXPECT_EQ(input::Converter<int>::tryParse("-7"), -7);
    EXPECT_EQ(input::Converter<size_t>::tryParse("not_a_number"), std::nullopt);
    EXPECT_EQ(input::Converter<size_t>::tryParse("42abc"), std::nullopt);
}

/**
 * @brief tests Converter<T> for MSTD_ENUM types via generated from_string
 *
 */
TEST(TestConverter, tryParseEnum)
{
    EXPECT_EQ(input::Converter<TestJobType>::tryParse("mm"), TestJobType::mm);
    EXPECT_EQ(input::Converter<TestJobType>::tryParse("qm"), TestJobType::qm);
    EXPECT_EQ(input::Converter<TestJobType>::tryParse("bogus"), std::nullopt);
}

/**
 * @brief the aliases of a value are listed in parentheses behind its name, the
 * enumerator that a name overrides is one of them
 *
 */
TEST(TestConverter, describeDomainGroupsTheAliasesOfAValue)
{
    EXPECT_EQ(
        input::Converter<TestAliased>::describeDomain({}),
        "Allowed values: none (nada), 3ob (threeob), verlet (v_verlet, vv), "
        "other"
    );
}

/**
 * @brief an enum without aliases is a plain list
 *
 */
TEST(TestConverter, describeDomainWithoutAliasesIsAPlainList)
{
    EXPECT_EQ(
        input::Converter<TestJobType>::describeDomain({}),
        "Allowed values: mm, qm, md"
    );
}

/**
 * @brief a value that is not allowed disappears with its aliases
 *
 */
TEST(TestConverter, describeDomainLeavesOutValuesThatAreNotAllowed)
{
    EXPECT_EQ(
        input::Converter<TestAliased>::describeDomain({TestAliased::VERLET}),
        "Allowed values: none (nada), 3ob (threeob), other"
    );
}

/**
 * @brief every spelling that the message lists is accepted by the parser, and
 * the message lists every accepted spelling
 *
 */
TEST(TestConverter, describeDomainListsExactlyTheAcceptedSpellings)
{
    const auto message = input::Converter<TestAliased>::describeDomain({});

    auto text = message.substr(std::string("Allowed values: ").size());
    for (const char separator : {'(', ')', ','})
        std::ranges::replace(text, separator, ' ');

    std::istringstream stream(text);
    std::string        spelling;
    size_t             count = 0;

    while (stream >> spelling)
    {
        EXPECT_TRUE(input::Converter<TestAliased>::tryParse(spelling))
            << spelling;
        ++count;
    }

    // 4 values, 1 overridden name, 3 aliases
    EXPECT_EQ(count, 4 + 1 + 3);
}

/**
 * @brief the messages for numbers say what is expected instead of a
 * placeholder
 *
 */
TEST(TestConverter, describeDomainOfNumbers)
{
    EXPECT_EQ(
        input::Converter<double>::describeDomain({}),
        "Value must be a number"
    );
    EXPECT_EQ(
        input::Converter<double>::describeDomain({0.0, 2.5}),
        "Value must be a number, not allowed: 0, not allowed: 2.5"
    );
    EXPECT_EQ(
        input::Converter<int>::describeDomain({}),
        "Value must be an integer"
    );
    EXPECT_EQ(
        input::Converter<int>::describeDomain({-1}),
        "Value must be an integer, not allowed: -1"
    );
}

/**
 * @brief a value of the wrong type for a double key gets the same message
 *
 */
TEST(TestInputKey, invalidDoubleSaysWhatIsExpected)
{
    input::InputRegistry registry;
    registry.registerKey<double>(
        input::KeyRegistry<double>{.metadata = {.name = "tau", .title = "Tau"}}
    );

    EXPECT_THROW_MSG(
        registry.parseLine({"tau", "=", "abc"}, 3),
        exc::InputFileException,
        "Invalid value \"abc\" for key \"tau\" at line 3 in input file. "
        "Value must be a number"
    );
}

/**
 * @brief tests that an unset key with a default resolves value() to the
 * default, and isSet() is false until explicitly parsed
 *
 */
TEST(TestInputKey, defaultValueBeforeParsing)
{
    input::InputKey<double> key(
        input::KeyRegistry<double>{
            .metadata =
                input::KeyMetadata{
                    .name        = "timestep",
                    .title       = "Timestep",
                    .description = "",
                    .unit        = ""
                },
            .defaultValue = 1.0
        }
    );

    EXPECT_FALSE(key.isSet());
    EXPECT_EQ(key.value(), 1.0);
    EXPECT_FALSE(key.explicitValue().has_value());
    EXPECT_EQ(key.defaultValue(), 1.0);
}

/**
 * @brief tests that parsing sets the explicit value without touching the
 * stored default
 *
 */
TEST(TestInputKey, explicitValueOverridesDefault)
{
    input::InputKey<double> key(
        input::KeyRegistry<double>{
            .metadata =
                input::KeyMetadata{
                    .name        = "timestep",
                    .title       = "Timestep",
                    .description = "",
                    .unit        = ""
                },
            .defaultValue = 1.0
        }
    );

    key.parse({"timestep", "=", "0.5"}, 1);

    EXPECT_TRUE(key.isSet());
    EXPECT_EQ(key.value(), 0.5);
    EXPECT_EQ(key.explicitValue(), 0.5);
    EXPECT_EQ(key.defaultValue(), 1.0);   // untouched
}

/**
 * @brief tests that value() throws when neither an explicit value nor a
 * default exists
 *
 */
TEST(TestInputKey, valueThrowsWithoutDefaultOrExplicit)
{
    input::InputKey<TestJobType> key(
        input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name        = "jobtype",
                    .title       = "Job Type",
                    .description = "",
                    .unit        = ""
                }
        }
    );

    EXPECT_FALSE(key.isSet());
    EXPECT_THROW_MSG(
        const auto _ = key.value(),
        std::logic_error,
        "Key \"jobtype\" has neither a set value nor a default"
    );
    EXPECT_FALSE(key.tryValue().has_value());
}

/**
 * @brief tests that an invalid token throws InputFileException with the
 * generated "expected one of ..." message
 *
 */
TEST(TestInputKey, invalidEnumTokenThrows)
{
    input::InputKey<TestJobType> key(
        input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name        = "jobtype",
                    .title       = "Job Type",
                    .description = "",
                    .unit        = ""
                }
        }
    );

    EXPECT_THROW_MSG(
        key.parse({"jobtype", "=", "bogus"}, 3),
        exc::InputFileException,
        "Invalid value \"bogus\" for key \"jobtype\" at line 3 in input "
        "file. Allowed values: mm, qm, md"
    );
}

/**
 * @brief tests that a per-key errorMessage override replaces the generated
 * message
 *
 */
TEST(TestInputKey, customErrorMessageOverridesGenerated)
{
    input::InputKey<TestJobType> key(
        input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name         = "jobtype",
                    .title        = "Job Type",
                    .description  = "",
                    .unit         = "",
                    .errorMessage = "Unrecognized job type"
                }
        }
    );

    EXPECT_THROW_MSG(
        key.parse({"jobtype", "=", "bogus"}, 4),
        exc::InputFileException,
        "Invalid value \"bogus\" for key \"jobtype\" at line 4 in input file: "
        "Unrecognized job type"
    );
}

/**
 * @brief tests the allowed-values restriction rejects an otherwise valid
 * value that is outside the given subset
 *
 */
TEST(TestInputKey, allowedSubsetRejectsOutOfRangeValue)
{
    input::InputKey<TestJobType> key(
        input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name        = "jobtype",
                    .title       = "Job Type",
                    .description = "",
                    .unit        = ""
                },
            .notAllowed = std::vector<TestJobType>{TestJobType::md}
        }
    );

    key.parse({"jobtype", "=", "mm"}, 1);
    EXPECT_EQ(key.value(), TestJobType::mm);

    key.clearValue();

    EXPECT_THROW_MSG(
        key.parse({"jobtype", "=", "md"}, 2),
        exc::InputFileException,
        "Invalid value \"md\" for key \"jobtype\" at line 2 in input file: not "
        "allowed"
    );
}

/**
 * @brief tests a customParser alias resolving a token the generated
 * from_string doesn't know about
 *
 */
TEST(TestInputKey, customParserAliasResolvesUnknownToken)
{
    input::KeyRegistry<TestJobType>::CustomParser aliasParser =
        [](std::string_view raw) -> std::optional<TestJobType>
    {
        if (raw == "molecular_dynamics")
            return TestJobType::md;
        return TestJobTypeMeta::from_string(raw);
    };

    auto createKey = [&aliasParser]() -> input::InputKey<TestJobType>
    {
        return input::InputKey<TestJobType>(input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name        = "jobtype",
                    .title       = "Job Type",
                    .description = "",
                    .unit        = ""
                },
            .customParser = aliasParser
        });
    };

    auto key = createKey();
    key.parse({"jobtype", "=", "molecular_dynamics"}, 1);
    EXPECT_EQ(key.value(), TestJobType::md);

    key = createKey();
    key.parse({"jobtype", "=", "qm"}, 2);
    EXPECT_EQ(key.value(), TestJobType::qm);
}

/**
 * @brief tests that onSet is invoked with the parsed value
 *
 */
TEST(TestInputKey, onSetCallbackInvoked)
{
    double captured = 0.0;

    input::InputKey<double> key(
        input::KeyRegistry<double>{
            .metadata =
                input::KeyMetadata{
                    .name        = "timestep",
                    .title       = "Timestep",
                    .description = "",
                    .unit        = ""
                },
            .onSet = [&captured](const double &value) { captured = value; }
        }
    );

    key.parse({"timestep", "=", "2.5"}, 1);
    EXPECT_EQ(captured, 2.5);
}

/**
 * @brief tests describe() for an unset key with a default, an explicit
 * value, a unit, and an allowed subset
 *
 */
TEST(TestInputKey, describeReflectsState)
{
    input::InputKey<double> key(
        input::KeyRegistry<double>{
            .metadata =
                input::KeyMetadata{
                    .name        = "timestep",
                    .title       = "Timestep",
                    .description = "Integration timestep for the MD run",
                    .unit        = "fs"
                },
            .defaultValue = 1.0
        }
    );

    EXPECT_EQ(
        key.describe(),
        "timestep (Timestep): Integration timestep for the MD run = 1 "
        "(default) [fs]"
    );

    key.parse({"timestep", "=", "0.5"}, 1);

    EXPECT_EQ(
        key.describe(),
        "timestep (Timestep): Integration timestep for the MD run = 0.5 "
        "[fs]"
    );
}

/**
 * @brief tests describe() includes the allowed-values list when present
 *
 */
TEST(TestInputKey, describeIncludesAllowedList)
{
    input::InputKey<TestJobType> key(
        input::KeyRegistry<TestJobType>{
            .metadata =
                input::KeyMetadata{
                    .name        = "jobtype",
                    .title       = "Job Type",
                    .description = "",
                    .unit        = ""
                },
            .notAllowed = std::vector<TestJobType>{TestJobType::qm},
        }
    );

    EXPECT_EQ(key.describe(), "jobtype (Job Type) = <unset> [not allowed: qm]");
}

/**
 * @brief tests input::InputRegistry: registration, dispatch by key name,
 * unknown key error, and describeAll()
 *
 */
TEST(TestInputRegistry, registerParseAndDescribeAll)
{
    input::InputRegistry registry;

    auto &timestepKey = registry.registerKey<double>(input::KeyRegistry<double>{
        .metadata =
            input::KeyMetadata{
                .name        = "timestep",
                .title       = "Timestep",
                .description = "",
                .unit        = "fs"
            },
        .defaultValue = 1.0
    });

    auto &jobTypeKey =
        registry.registerKey<TestJobType>(input::KeyRegistry<TestJobType>{
            .metadata = input::KeyMetadata{
                .name        = "jobtype",
                .title       = "Job Type",
                .description = "",
                .unit        = ""
            }
        });

    registry.parseLine({"timestep", "=", "0.5"}, 1);
    registry.parseLine({"jobtype", "=", "qm"}, 2);

    EXPECT_EQ(timestepKey.value(), 0.5);
    EXPECT_EQ(jobTypeKey.value(), TestJobType::qm);

    EXPECT_EQ(registry.get<double>("timestep").value(), 0.5);

    const auto descriptions = registry.describeAll();
    EXPECT_EQ(descriptions.size(), size_t{2});
}

/**
 * @brief tests that an unknown key throws InputFileException
 *
 */
TEST(TestInputRegistry, unknownKeyThrows)
{
    input::InputRegistry registry;

    EXPECT_THROW_MSG(
        registry.parseLine({"bogus_key", "=", "1"}, 5),
        exc::InputFileException,
        "Unknown key \"bogus_key\" at line 5 in input file"
    );
}

/**
 * @brief tests Converter<std::unordered_map<std::string, double>>::tryParse
 * with a valid list of key:value pairs
 */
TEST(TestInputParam, mapConverterParsesKeyValuePairs)
{
    using Map = std::unordered_map<std::string, double>;

    const auto parsed = input::Converter<Map>::tryParse("C:0.5,H:-1.25e-1,O:3");

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->size(), 3);
    EXPECT_DOUBLE_EQ(parsed->at("C"), 0.5);
    EXPECT_DOUBLE_EQ(parsed->at("H"), -0.125);
    EXPECT_DOUBLE_EQ(parsed->at("O"), 3.0);

    const auto single = input::Converter<Map>::tryParse("N:2");
    ASSERT_TRUE(single.has_value());
    EXPECT_EQ(single->size(), 1);

    // the last value of a repeated key wins
    const auto repeated = input::Converter<Map>::tryParse("C:1,C:2");
    ASSERT_TRUE(repeated.has_value());
    EXPECT_DOUBLE_EQ(repeated->at("C"), 2.0);
}

/**
 * @brief every malformed pair is rejected
 */
TEST(TestInputParam, mapConverterRejectsMalformedPairs)
{
    using Map = std::unordered_map<std::string, double>;

    for (const std::string raw :
         {"C",              // no separator
          ":1.0",           // empty key
          "C:",             // empty value
          "C:1:2",          // second separator
          "C:one",          // not a number
          "C:1.5garbage",   // trailing characters
          "C:inf",          // not finite
          "C:nan",          // not finite
          "C:1e999",        // out of range
          "C:1,H",          // second pair malformed
          "C:1,:2"})        // second pair has no key
    {
        EXPECT_FALSE(input::Converter<Map>::tryParse(raw).has_value()) << raw;
    }
}

/**
 * @brief the domain of the map converter is described for the error message
 */
TEST(TestInputParam, mapConverterDescribesItsDomain)
{
    using Map = std::unordered_map<std::string, double>;

    EXPECT_EQ(
        input::Converter<Map>::describeDomain({}),
        "Value must be a comma-separated list of key:value pairs, where the "
        "key is a string and the value is a double."
    );
}
