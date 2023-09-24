#ifndef _OUTPUT_HPP_

#define _OUTPUT_HPP_

#include <cstddef>   // for size_t
#include <fstream>
#include <string>
#include <string_view>   // for string_view

namespace output
{
    /**
     * @class Output
     *
     * @brief Base class for output files
     *
     */
    class Output
    {
      protected:
        std::string   _fileName;
        std::ofstream _fp;
        int           _rank;

        void openFile();

      public:
        explicit Output(const std::string &filename) : _fileName(filename){};

        void setFilename(const std::string_view &);
        void close() { _fp.close(); }

        /********************************
         * standard getters and setters *
         ********************************/

        std::string getFilename() const { return _fileName; }
    };

}   // namespace output

#endif   // _OUTPUT_HPP_