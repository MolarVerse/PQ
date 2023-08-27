#ifndef _TOPOLOGY_SECTION_HPP_

#define _TOPOLOGY_SECTION_HPP_

#include <iosfwd>   // for ifstream
#include <string>   // for string, allocator
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

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

        void process(std::vector<std::string> &lineElements, engine::Engine &);

        virtual std::string keyword()                                                                = 0;
        virtual void        processSection(std::vector<std::string> &lineElements, engine::Engine &) = 0;
        virtual void        endedNormally(bool) const                                                = 0;

        void setLineNumber(const int lineNumber) { _lineNumber = lineNumber; }
        void setFp(std::ifstream *fp) { _fp = fp; }

        [[nodiscard]] int getLineNumber() const { return _lineNumber; }
    };

}   // namespace readInput::topology

#endif   // _TOPOLOGY_SECTION_HPP_