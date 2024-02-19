project(example_swig)
file(GLOB SRC LIST_DIRECTORIES false "*.cu")
file(GLOB ISRC LIST_DIRECTORIES false "*.i")

if(SWIG_FOUND)
  if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
  endif()
  set_source_files_properties(${PROJECT_NAME}.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(${PROJECT_NAME}.i PROPERTIES USE_TARGET_INCLUDE_DIRECTORIES ON)
  swig_add_library(${PROJECT_NAME} LANGUAGE python
    OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}/${CMAKE_PROJECT_NAME}"
    SOURCES ${ISRC} ${SRC})
  if(WIN32)
    set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX ".${Python_SOABI}.pyd")
  else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()
  if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Python::Module CUDA::cudart_static)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE Python::Module)
    set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  endif()
else()
  add_library(${PROJECT_NAME} MODULE ${SRC})
  if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} CUDA::cudart_static)
  else()
    set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  endif()
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME}
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
