#ifndef _PARAMETER_FILE_SECTION_HPP_

#define _PARAMETER_FILE_SECTION_HPP_

#include "engine.hpp"

namespace readInput::parameterFile
{
    class ParameterFileSection;
    class TypesSection;
    // class BondSection;
    // class AngleSection;
    // class DihedralSection;
    // class ImproperDihedralSection;
}   // namespace readInput::parameterFile

/**
 * @class TopologySection
 *
 * @brief base class for reading topology file sections
 *
 */
class readInput::parameterFile::ParameterFileSection
{
  protected:
    int            _lineNumber;
    std::ifstream *_fp;

  public:
    virtual ~ParameterFileSection() = default;

    virtual void process(std::vector<std::string> &, engine::Engine &);

    virtual std::string keyword()                                                    = 0;
    virtual void        processSection(std::vector<std::string> &, engine::Engine &) = 0;
    void                endedNormally(bool);

    void setLineNumber(int lineNumber) { _lineNumber = lineNumber; }
    void setFp(std::ifstream *fp) { _fp = fp; }

    int getLineNumber() const { return _lineNumber; }
};

/**
 * @class TypesSection
 *
 * @brief reads types line section of parameter file
 *
 */
class readInput::parameterFile::TypesSection : public readInput::parameterFile::ParameterFileSection
{
  public:
    std::string keyword() override { return "types"; }
    void        process(std::vector<std::string> &, engine::Engine &) override;
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
};

// /**
//  * @class BondSection
//  *
//  * @brief reads bond section of topology file
//  *
//  */
// class readInput::BondSection : public readInput::TopologySection
// {
//   public:
//     std::string keyword() override { return "bonds"; }
//     void        processSection(std::vector<std::string> &, engine::Engine &) override;
// };

// /**
//  * @class AngleSection
//  *
//  * @brief reads angle section of topology file
//  *
//  */
// class readInput::AngleSection : public readInput::TopologySection
// {
//   public:
//     std::string keyword() override { return "angles"; }
//     void        processSection(std::vector<std::string> &, engine::Engine &) override;
// };

// /**
//  * @class DihedralSection
//  *
//  * @brief reads dihedral section of topology file
//  *
//  */
// class readInput::DihedralSection : public readInput::TopologySection
// {
//   public:
//     std::string keyword() override { return "dihedrals"; }
//     void        processSection(std::vector<std::string> &, engine::Engine &) override;
// };

// /**
//  * @class ImproperDihedralSection
//  *
//  * @brief reads improper dihedral section of topology file
//  *
//  */
// class readInput::ImproperDihedralSection : public readInput::TopologySection
// {
//   public:
//     std::string keyword() override { return "impropers"; }
//     void        processSection(std::vector<std::string> &, engine::Engine &) override;
// };

#endif   // _PARAMETER_FILE_SECTION_HPP_