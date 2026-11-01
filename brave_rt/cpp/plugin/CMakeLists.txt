cmake_minimum_required(VERSION 3.22)
project(BraveRTPlugin VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BRAVE_RT_NATIVE "Tune for the build machine" ON)
if(BRAVE_RT_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  else()
    add_compile_options(-mcpu=native)
  endif()
endif()

set(BRAVE_RT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(BRAVE_RT_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../brave_rt")
set(JUCE_PATH "${BRAVE_RT_ROOT}/third_party/JUCE" CACHE PATH "Path to JUCE")
if(NOT EXISTS "${JUCE_PATH}/CMakeLists.txt")
  message(FATAL_ERROR "JUCE not found at ${JUCE_PATH}")
endif()

# Engine as a plain library (avoid nested project() which breaks juceaide parent).
add_library(brave_rt_engine STATIC
  "${BRAVE_RT_ENGINE_DIR}/src/pqmf.cpp"
  "${BRAVE_RT_ENGINE_DIR}/src/brave_engine.cpp"
  "${BRAVE_RT_ENGINE_DIR}/src/descriptor_extract.cpp"
  "${BRAVE_RT_ENGINE_DIR}/src/cyclegan_engine.cpp"
)
set_target_properties(brave_rt_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
# SYSTEM: engine headers use int↔size_t indexing; JUCE's -Wsign-conversion
# would otherwise flood the Mac/Clang build when Plugin*.cpp includes them.
target_include_directories(brave_rt_engine SYSTEM PUBLIC "${BRAVE_RT_ENGINE_DIR}/include")
find_package(Threads REQUIRED)
target_link_libraries(brave_rt_engine PUBLIC Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  target_compile_options(brave_rt_engine PRIVATE -Wno-sign-conversion -Wno-conversion)
endif()

add_subdirectory("${JUCE_PATH}" juce_build EXCLUDE_FROM_ALL)

set(BRAVE_RT_FORMATS VST3)
if(APPLE)
  list(APPEND BRAVE_RT_FORMATS AU)
endif()

juce_add_plugin(BraveRT
  COMPANY_NAME "BRAVE"
  BUNDLE_ID "edu.mit.brave.bravert"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  PLUGIN_MANUFACTURER_CODE Brav
  PLUGIN_CODE Brt1
  FORMATS ${BRAVE_RT_FORMATS}
  PRODUCT_NAME "BraveRT"
)

target_sources(BraveRT
  PRIVATE
    PluginProcessor.cpp
    PluginEditor.cpp
)

target_compile_definitions(BraveRT
  PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(BraveRT
  PRIVATE
    brave_rt_engine
    juce::juce_audio_utils
  PUBLIC
    juce::juce_recommended_config_flags
    # Skip LTO on Linux CI boxes — link is huge/slow; enable on Mac release if wanted.
    # juce::juce_recommended_lto_flags
    # Skip juce_recommended_warning_flags: -Wsign-conversion/-Wconversion flood
    # from engine headers (int sizes × size_t indexing). Engine is SYSTEM-included.
)

target_include_directories(BraveRT PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Third-party noise (Apple Clang): Steinberg VST3 SDK memset on FVariant, etc.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  target_compile_options(BraveRT PRIVATE
    -Wno-nontrivial-memcall
    -Wno-sign-conversion
    -Wno-conversion
  )
  # SharedCode / format targets inherit via INTERFACE sometimes; also set on VST3/AU.
  foreach(_fmt VST3 AU)
    if(TARGET BraveRT_${_fmt})
      target_compile_options(BraveRT_${_fmt} PRIVATE
        -Wno-nontrivial-memcall
        -Wno-sign-conversion
        -Wno-conversion
      )
    endif()
  endforeach()
endif()
