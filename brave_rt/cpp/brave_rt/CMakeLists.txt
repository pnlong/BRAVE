cmake_minimum_required(VERSION 3.16)
project(brave_rt_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(BRAVE_RT_NATIVE "Tune for the build machine (-march=native / -mcpu=native)" OFF)
if(BRAVE_RT_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  else()
    add_compile_options(-mcpu=native)
  endif()
endif()

add_library(brave_rt_engine
  src/pqmf.cpp
  src/brave_engine.cpp
  src/descriptor_extract.cpp
  src/cyclegan_engine.cpp
)
target_include_directories(brave_rt_engine PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(brave_rt_engine PUBLIC Threads::Threads)

add_executable(pqmf_roundtrip tools/pqmf_roundtrip.cpp)
target_link_libraries(pqmf_roundtrip PRIVATE brave_rt_engine)

add_executable(sidechain_smoke tools/sidechain_smoke.cpp)
target_link_libraries(sidechain_smoke PRIVATE brave_rt_engine)

add_executable(parity_cli tools/parity_cli.cpp)
target_link_libraries(parity_cli PRIVATE brave_rt_engine)

add_executable(rtf_bench tools/rtf_bench.cpp)
target_link_libraries(rtf_bench PRIVATE brave_rt_engine)

add_executable(parity_stream_layers tools/parity_stream_layers.cpp)
target_link_libraries(parity_stream_layers PRIVATE brave_rt_engine)

add_executable(soak tools/soak.cpp)
target_link_libraries(soak PRIVATE brave_rt_engine)

add_executable(cyclegan_parity_cli tools/cyclegan_parity_cli.cpp)
target_link_libraries(cyclegan_parity_cli PRIVATE brave_rt_engine)

add_executable(cyclegan_rtf_bench tools/cyclegan_rtf_bench.cpp)
target_link_libraries(cyclegan_rtf_bench PRIVATE brave_rt_engine)

add_executable(sidechain_parity_cli tools/sidechain_parity_cli.cpp)
target_link_libraries(sidechain_parity_cli PRIVATE brave_rt_engine)

add_executable(fader_parity_cli tools/fader_parity_cli.cpp)
target_link_libraries(fader_parity_cli PRIVATE brave_rt_engine)

add_executable(extract_smoke tools/extract_smoke.cpp)
target_link_libraries(extract_smoke PRIVATE brave_rt_engine)
