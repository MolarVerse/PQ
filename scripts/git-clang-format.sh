git-clang-format \
    --binary "$(command -v clang-format)" \
    --extensions c,cc,cpp,cxx,h,hh,hpp,hxx \
    origin/dev \
    -- src include tests apps benchmarks

