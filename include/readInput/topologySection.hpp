#ifndef _TOPOLOGY_SECTION_HPP_

#define _TOPOLOGY_SECTION_HPP_

#include "engine.hpp"

namespace readInput
{
    class TopologySection;
    class ShakeSection;
    class BondSection;
    class AngleSection;
    class DihedralSection;
    class ImproperDihedralSection;
}   // namespace readInput

/**
 * @class TopologySection
 *
 * @brief base class for reading topology file sections
 *
 */
class readInput::TopologySection
{
  protected:
    int            _lineNumber;
    std::ifstream *_fp;

  public:
    virtual ~TopologySection() = default;

    void process(std::vector<std::string> &, engine::Engine &);

    virtual std::string keyword()                                                    = 0;
    virtual void        processSection(std::vector<std::string> &, engine::Engine &) = 0;
    virtual void        endedNormally(bool) const                                    = 0;

    void setLineNumber(int lineNumber) { _lineNumber = lineNumber; }
    void setFp(std::ifstream *fp) { _fp = fp; }

    int getLineNumber() const { return _lineNumber; }
};

/**
 * @class BondSection
 *
 * @brief reads bond section of topology file
 *
 */
class readInput::BondSection : public readInput::TopologySection
{
  public:
    std::string keyword() override { return "bonds"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        endedNormally(bool) const override;
};

/**
 * @class AngleSection
 *
 * @brief reads angle section of topology file
 *
 */
class readInput::AngleSection : public readInput::TopologySection
{
  public:
    std::string keyword() override { return "angles"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        endedNormally(bool) const override;
};

/**
 * @class DihedralSection
 *
 * @brief reads dihedral section of topology file
 *
 */
class readInput::DihedralSection : public readInput::TopologySection
{
  public:
    std::string keyword() override { return "dihedrals"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        endedNormally(bool) const override;
};

/**
 * @class ImproperDihedralSection
 *
 * @brief reads improper dihedral section of topology file
 *
 */
class readInput::ImproperDihedralSection : public readInput::TopologySection
{
  public:
    std::string keyword() override { return "impropers"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        endedNormally(bool) const override;
};

/**
 * @class ShakeSection
 *
 * @brief reads shake section of topology file
 *
 */
class readInput::ShakeSection : public readInput::TopologySection
{
  public:
    std::string keyword() override { return "shake"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        endedNormally(bool) const override;
};

#endif   // _TOPOLOGY_SECTION_HPP_