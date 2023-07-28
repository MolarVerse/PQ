#ifndef _TOPOLGY_SECTION_HPP_

#define _TOPOLGY_SECTION_HPP_

#include "engine.hpp"

namespace setup
{
    class TopologySection;
    class ShakeSection;
}   // namespace setup

/**
 * @class TopologySection
 *
 * @brief base class for reading topology file sections
 *
 */
class setup::TopologySection
{
  protected:
    int            _lineNumber;
    std::ifstream *_fp;

  public:
    virtual ~TopologySection() = default;

    virtual std::string keyword()                                             = 0;
    virtual void        process(std::vector<std::string> &, engine::Engine &) = 0;

    void setLineNumber(int lineNumber) { _lineNumber = lineNumber; }
    void setFp(std::ifstream *fp) { _fp = fp; }

    int getLineNumber() const { return _lineNumber; }
};

/**
 * @class ShakeSection
 *
 * @brief reads shake section of topology file
 *
 */
class setup::ShakeSection : public setup::TopologySection
{
  public:
    std::string keyword() override { return "shake"; }
    void        process(std::vector<std::string> &, engine::Engine &) override;
};

#endif   // _TOPOLGY_SECTION_HPP_