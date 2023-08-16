#ifndef _PARAMETER_FILE_SECTION_HPP_

#define _PARAMETER_FILE_SECTION_HPP_

#include "engine.hpp"
#include "forceField.hpp"

namespace readInput::parameterFile
{
    class ParameterFileSection;
    class TypesSection;
    class BondSection;
    class AngleSection;
    class DihedralSection;
    class ImproperDihedralSection;
    class NonCoulombicsSection;
}   // namespace readInput::parameterFile

/**
 * @class ParameterFileSection
 *
 * @brief base class for reading parameter file sections
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
    void         endedNormally(bool);

    virtual std::string keyword()                                                    = 0;
    virtual void        processSection(std::vector<std::string> &, engine::Engine &) = 0;
    virtual void        processHeader(std::vector<std::string> &, engine::Engine &)  = 0;

    void setLineNumber(int lineNumber) { _lineNumber = lineNumber; }
    void setFp(std::ifstream *fp) { _fp = fp; }

    [[nodiscard]] int getLineNumber() const { return _lineNumber; }
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
    void        processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
};

/**
 * @class BondSection
 *
 * @brief reads bond section of parameter file
 *
 */
class readInput::parameterFile::BondSection : public readInput::parameterFile::ParameterFileSection
{
  public:
    std::string keyword() override { return "bonds"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
};

/**
 * @class AngleSection
 *
 * @brief reads angle section of parameter file
 *
 */
class readInput::parameterFile::AngleSection : public readInput::parameterFile::ParameterFileSection
{
  public:
    std::string keyword() override { return "angles"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
};

/**
 * @class DihedralSection
 *
 * @brief reads dihedral section of parameter file
 *
 */
class readInput::parameterFile::DihedralSection : public readInput::parameterFile::ParameterFileSection
{
  public:
    std::string keyword() override { return "dihedrals"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
};

/**
 * @class ImproperDihedralSection
 *
 * @brief reads improper dihedral section of parameter file
 *
 */
class readInput::parameterFile::ImproperDihedralSection : public readInput::parameterFile::ParameterFileSection
{
  public:
    std::string keyword() override { return "impropers"; }
    void        processHeader(std::vector<std::string> &, engine::Engine &) override{};   // TODO: implement
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
};

/**
 * @class NonCoulombicsSection
 *
 * @brief reads non-coulombics section of parameter file
 *
 */
class readInput::parameterFile::NonCoulombicsSection : public readInput::parameterFile::ParameterFileSection
{
  private:
    potential::NonCoulombType _nonCoulombType;

  public:
    std::string keyword() override { return "noncoulombics"; }
    void        processSection(std::vector<std::string> &, engine::Engine &) override;
    void        processHeader(std::vector<std::string> &, engine::Engine &) override;
    void        processLJ(std::vector<std::string> &, engine::Engine &) const;
    void        processBuckingham(std::vector<std::string> &, engine::Engine &) const;
    void        processMorse(std::vector<std::string> &, engine::Engine &) const;

    void setNonCoulombType(potential::NonCoulombType nonCoulombicType) { _nonCoulombType = nonCoulombicType; }
    [[nodiscard]] potential::NonCoulombType getNonCoulombType() const { return _nonCoulombType; }
};

#endif   // _PARAMETER_FILE_SECTION_HPP_