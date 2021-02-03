project(example_swig)
file(GLOB SRC LIST_DIRECTORIES false "*.cu")
file(GLOB ISRC LIST_DIRECTORIES false "*.i")

#include_directories(${Python3_INCLUDE_DIRS})
#include_directories(${Python3_NumPy_INCLUDE_DIRS})

if(SWIG_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  set_source_files_properties(${PROJECT_NAME}.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(${PROJECT_NAME}.i PROPERTIES USE_TARGET_INCLUDE_DIRECTORIES ON)
  swig_add_library(${PROJECT_NAME} LANGUAGE python SOURCES ${ISRC} ${SRC})
  swig_link_libraries(${PROJECT_NAME} CUDA::cudart_static)
else()
  add_library(${PROJECT_NAME} MODULE ${SRC})
  target_link_libraries(${PROJECT_NAME} CUDA::cudart_static)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")

if(SKBUILD)
  python_extension_module(${PROJECT_NAME})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME}
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
