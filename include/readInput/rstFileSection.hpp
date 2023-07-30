#ifndef _RST_FILE_SECTION_HPP_

#define _RST_FILE_SECTION_HPP_

#include "engine.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

#include <fstream>
#include <gtest/gtest_prod.h>
#include <string>
#include <vector>

// Forward declarations for friend test
class TestAtomSection_testProcessAtomLine_Test;

namespace readInput
{
    class RstFileSection;
    class BoxSection;
    class NoseHooverSection;
    class StepCountSection;
    class AtomSection;
}   // namespace readInput

/**
 * @class RstFileSection
 *
 * @brief Base class for all sections of a .rst file
 *
 */
class readInput::RstFileSection
{
  public:
    virtual ~RstFileSection() = default;

    int                 _lineNumber;
    std::ifstream      *_fp;
    virtual std::string keyword()                                             = 0;
    virtual bool        isHeader()                                            = 0;
    virtual void        process(std::vector<std::string> &, engine::Engine &) = 0;
};

/**
 * @class BoxSection
 *
 * @brief Reads the box section of a .rst file
 *
 */
class readInput::BoxSection : public readInput::RstFileSection
{
  public:
    std::string keyword() override { return "box"; }
    bool        isHeader() override;
    void        process(std::vector<std::string> &, engine::Engine &) override;
};

/**
 * @class NoseHooverSection
 *
 * @brief Reads the Nose-Hoover section of a .rst file
 *        TODO: This section is not yet implemented
 *
 */
class readInput::NoseHooverSection : public readInput::RstFileSection
{
  public:
    std::string keyword() override { return "chi"; }
    bool        isHeader() override;
    void        process(std::vector<std::string> &, engine::Engine &) override;
};

/**
 * @class StepCountSection
 *
 * @brief Reads the step count section of a .rst file
 *
 */
class readInput::StepCountSection : public readInput::RstFileSection
{
  public:
    std::string keyword() override { return "step"; }
    bool        isHeader() override;
    void        process(std::vector<std::string> &, engine::Engine &) override;
};

/**
 * @class AtomSection
 *
 * @brief Reads the atom section of a .rst file
 *
 */
class readInput::AtomSection : public readInput::RstFileSection
{
  private:
    void processAtomLine(std::vector<std::string> &, simulationBox::Molecule &) const;
    void checkAtomLine(std::vector<std::string> &, std::string &, const simulationBox::Molecule &);

    FRIEND_TEST(::TestAtomSection, testProcessAtomLine);

  public:
    std::string keyword() override { return ""; }
    bool        isHeader() override;
    void        process(std::vector<std::string> &, engine::Engine &) override;
};

#endif   // _RST_FILE_SECTION_HPP_