add_library(evaluator
    evaluator.cpp

    mmEvaluator.cpp
)

target_include_directories(evaluator
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/evaluator
)

target_link_libraries(evaluator
    PRIVATE
    linearAlgebra
)

install(TARGETS evaluator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)