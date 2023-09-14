#ifndef _QM_MD_ENGINE_HPP_

#define _QM_MD_ENGINE_HPP_

#include "engine.hpp"     // for Engine
#include "qmRunner.hpp"   // for QMRunner

#include <memory>   // for unique_ptr

namespace engine
{

    /**
     * @class QMMDEngine
     *
     * @brief Contains all the information needed to run a QM MD simulation
     *
     */
    class QMMDEngine : public Engine
    {
      private:
        std::unique_ptr<QM::QMRunner> _qmRunner = nullptr;

      public:
        void takeStep() override;

        template <typename T>
        void setQMRunner(T runner)
        {
            _qmRunner = std::make_unique<T>(runner);
        }

        [[nodiscard]] QM::QMRunner *getQMRunner() const { return _qmRunner.get(); }
    };

}   // namespace engine

#endif   // _QM_MD_ENGINE_HPP_