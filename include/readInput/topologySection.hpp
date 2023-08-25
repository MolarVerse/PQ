#ifndef _TOPOLOGY_SECTION_HPP_

#define _TOPOLOGY_SECTION_HPP_

#include <iosfwd>   // for ifstream
#include <string>   // for string, allocator
#include <vector>   // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput::topology
{
    /**
     * @class TopologySection
     *
     * @brief base class for reading topology file sections
     *
     */
    class TopologySection
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
    class BondSection : public TopologySection
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
    class AngleSection : public TopologySection
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
    class DihedralSection : public TopologySection
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
    class ImproperDihedralSection : public TopologySection
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
    class ShakeSection : public TopologySection
    {
      public:
        std::string keyword() override { return "shake"; }
        void        processSection(std::vector<std::string> &, engine::Engine &) override;
        void        endedNormally(bool) const override;
    };

}   // namespace readInput::topology

#endif   // _TOPOLOGY_SECTION_HPP_