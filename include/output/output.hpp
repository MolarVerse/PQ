#ifndef _OUTPUT_HPP_

#define _OUTPUT_HPP_

#include <fstream>              // for ofstream
#include <gtest/gtest_prod.h>   // for FRIEND_TEST
#include <string>               // for string
#include <string_view>          // for string_view

class TestOutput_testSpecialSetFilename_Test;   // Friend test class

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

        FRIEND_TEST(::TestOutput, testSpecialSetFilename);

        /********************************
         * standard getters and setters *
         ********************************/

        std::string getFilename() const { return _fileName; }
    };

}   // namespace output

#endif   // _OUTPUT_HPP_