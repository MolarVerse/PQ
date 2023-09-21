#ifndef _EXPECT_THROW_WITH_MESSAGE_HPP_

#define _EXPECT_THROW_WITH_MESSAGE_HPP_

/**
 * @macro EXPECT_THROW_MSG
 *
 * @brief expects that a statement throws an exception of a given type with a given message
 *
 */
#define EXPECT_THROW_MSG(statement, expected_exception, expected_what)                                                           \
    try                                                                                                                          \
    {                                                                                                                            \
        statement;                                                                                                               \
        FAIL() << "Expected: " #statement " throws an exception of type " #expected_exception ".\n"                              \
                  "  Actual: it throws nothing.";                                                                                \
    }                                                                                                                            \
    catch (const expected_exception &e)                                                                                          \
    {                                                                                                                            \
        EXPECT_EQ(expected_what, std::string{e.what()});                                                                         \
    }                                                                                                                            \
    catch (...)                                                                                                                  \
    {                                                                                                                            \
        FAIL() << "Expected: " #statement " throws an exception of type " #expected_exception ".\n"                              \
                  "  Actual: it throws a different type.";                                                                       \
    }

/**
 * @macro ASSERT_THROW_MSG
 *
 * @brief expects that a statement throws an exception of a given type with a given message
 *
 */
#define ASSERT_THROW_MSG(statement, expected_exception, expected_what)                                                           \
    EXPECT_THROW_MSG(statement, expected_exception, expected_what)

#endif   // _EXPECT_THROW_WITH_MESSAGE_HPP_
