cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 2 CACHE STRING "version" FORCE)
endif()
string(REGEX REPLACE [[([0-9]+)\.([0-9]+)\.([0-9]+).*]] [[\1.\2.\3]] SKBUILD_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
project(cuvec LANGUAGES C CXX VERSION "${SKBUILD_PROJECT_VERSION}")

option(CUVEC_CUDA_OPTIONAL "Make CUDA optional rather than forced" ON)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA arch" FORCE)
endif()
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED) # NumPy
if(NOT CUVEC_CUDA_OPTIONAL)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
else()
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
  else()
    message(WARNING "Could not find CUDA: using CPU")
    add_compile_definitions(CUVEC_DISABLE_CUDA)
    #list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      add_definitions("-x c++")
    endif()
  endif()
endif()
if(SKBUILD)
  find_package(SWIG 4.0)
  if(SWIG_FOUND)
    include(${SWIG_USE_FILE})
    set(${CMAKE_PROJECT_NAME}_SWIG_SRC "${CMAKE_CURRENT_LIST_DIR}/include/${CMAKE_PROJECT_NAME}.i")
    set_source_files_properties("${${CMAKE_PROJECT_NAME}_SWIG_SRC}" PROPERTIES CPLUSPLUS ON)
  endif()
endif()
cmake_policy(POP)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(CUVEC_DEBUG "Print out CUDA malloc & free operations" OFF)
if(CUVEC_DEBUG)
  add_compile_definitions(CUVEC_DEBUG)
endif(CUVEC_DEBUG)
message(STATUS "CuVec debugging: ${CUVEC_DEBUG}")

set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION ${CMAKE_PROJECT_NAME}/include)

# main project

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})

if(SKBUILD)
  python_add_library(${PROJECT_NAME} MODULE WITH_SOABI ${SRC})
else()
  add_library(${PROJECT_NAME} SHARED ${SRC})
endif()
add_library(AMYPAD::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
if(CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
else()
  set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE AMYPAD${PROJECT_NAME}Targets.cmake
  NAMESPACE AMYPAD:: DESTINATION ${CMAKE_PROJECT_NAME}/cmake)

# alternative swvec module
if(SWIG_FOUND AND SKBUILD)
  if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set_source_files_properties(src/swvec.i PROPERTIES COMPILE_DEFINITIONS _CUVEC_HALF=__half)
  endif()
  set_source_files_properties(src/swvec.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(src/swvec.i PROPERTIES USE_TARGET_INCLUDE_DIRECTORIES ON)
  swig_add_library(swvec LANGUAGE python
    OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}/${CMAKE_PROJECT_NAME}"
    SOURCES src/swvec.i)
  if(WIN32)
    set_property(TARGET swvec PROPERTY SUFFIX ".${Python_SOABI}.pyd")
  else()
    set_property(TARGET swvec PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()
  if(CUDAToolkit_FOUND)
    target_link_libraries(swvec PRIVATE Python::Module CUDA::cudart_static)
  else()
    target_link_libraries(swvec PRIVATE Python::Module)
  endif()
  target_include_directories(swvec PUBLIC
    "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
    "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
  set_target_properties(swvec PROPERTIES
    CXX_STANDARD 11
    VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
    INTERFACE_swvec_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
  set_property(TARGET swvec APPEND PROPERTY COMPATIBLE_INTERFACE_STRING swvec_MAJOR_VERSION)
  install(TARGETS swvec
    INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
    LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
endif()

add_subdirectory(src/example_mod)
add_subdirectory(src/example_swig)

# install project

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${CMAKE_PROJECT_VERSION}" COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
