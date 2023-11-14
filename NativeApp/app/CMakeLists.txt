cmake_minimum_required(VERSION 3.4.1)
set(TARGET pytorch_nativeapp)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 17)

set(build_DIR ${CMAKE_SOURCE_DIR}/build)

set(pytorch_testapp_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
file(GLOB pytorch_testapp_SOURCES
  ${pytorch_testapp_cpp_DIR}/pytorch_nativeapp.cpp
)

add_library(${TARGET} SHARED
    ${pytorch_testapp_SOURCES}
)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android-2.1.0.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android-2.1.0.aar/jni/${ANDROID_ABI}")

target_compile_options(${TARGET} PRIVATE
  -fexceptions
)

target_include_directories(${TARGET} PRIVATE ${PYTORCH_INCLUDE_DIRS})

set(BUILD_SUBDIR ${ANDROID_ABI})

find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)
find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${TARGET}
  ${PYTORCH_LIBRARY}
  ${FBJNI_LIBRARY}
  log)
