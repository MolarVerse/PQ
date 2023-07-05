#include "testGuffDatReader.hpp"

#include "exceptions.hpp"

TEST_F(TestGuffDatReader, parseLineErrorMoltypeNotFound)
{
    auto line = std::vector<std::string>{"3", "1", "1"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);

    line = std::vector<std::string>{"1", "1", "3"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);
}

TEST_F(TestGuffDatReader, parseLineErrorAtomtypeNotFound)
{
    auto line = std::vector<std::string>{"1", "0", "2", "3"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);

    line = std::vector<std::string>{"1", "1", "2", "1"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}