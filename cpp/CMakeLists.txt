cmake_minimum_required(VERSION 3.16)
project(backstop_guard CXX)

# Header-only: link `backstop_guard` to get the include path.
add_library(backstop_guard INTERFACE)
target_include_directories(backstop_guard INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(backstop_guard INTERFACE cxx_std_17)

add_executable(guard_replay guard_replay.cpp)
target_link_libraries(guard_replay PRIVATE backstop_guard)
