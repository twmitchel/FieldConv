cmake_minimum_required(VERSION 3.10.0)

project(fcutils)

# == Compile option to fix static/dynamic library linking
add_compile_options(-fPIC)

# == Deps
add_subdirectory(deps/geometry-central)
add_subdirectory(deps/pybind11)

# == Build our project stuff
pybind11_add_module(fcutils src/main.cpp)
target_include_directories(fcutils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(fcutils PRIVATE geometry-central)
