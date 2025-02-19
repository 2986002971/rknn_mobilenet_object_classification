# @author        Liang Huanyan
# @email         lianghuanyan@alientek.com
# http://www.openedv.com/forum.php

cmake_minimum_required(VERSION 3.8)
project(ATK_MobileNet_Classification)
message(STATUS "cmake version ${CMAKE_VERSION}")

# IPO设置
cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

# 工具链设置
set(TOOLCHAIN_DIR /opt/atk-dlrv1126-toolchain)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot)

# C++标准设置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 优化选项
add_definitions(-O3)                     # 最高级别优化
add_definitions(-march=armv7-a)          # ARM v7-A 架构
add_definitions(-mfpu=neon-vfpv4)       # 启用 NEON 和 VFPv4
add_definitions(-mfloat-abi=hard)        # 硬浮点
add_definitions(-ffast-math)             # 快速数学运算
add_definitions(-ftree-vectorize)        # 启用向量化
add_definitions(-fomit-frame-pointer)    # 省略帧指针
add_definitions(-funroll-loops)          # 循环展开
add_definitions(-flto)                   # 链接时优化

# 警告控制
add_definitions(-Wall)                   # 启用所有警告
add_definitions(-Wno-write-strings)      # 忽略字符串警告
add_definitions(-Wno-return-type)        # 忽略返回类型警告
add_definitions(-Wno-sign-compare)       # 忽略符号比较警告

# 添加mongoose源文件
set(MONGOOSE_SOURCES mongoose.c)

# 设置所需的库
set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_dnn)

# 包含目录
include_directories(${SYSROOT})
include_directories(${SYSROOT}/rknn)
include_directories(.)
include_directories(./include)

# 添加可执行文件
add_executable(atk_mobilenet_object_classification 
    atk_mobilenet_object_classification.cpp 
    ${MONGOOSE_SOURCES}
)

# 链接库
target_link_libraries(atk_mobilenet_object_classification
    ${OPENCV_LIBS}
    rknn_api
    pthread
    dl
)

# 链接时优化
if(supported)
    message(STATUS "IPO/LTO enabled")
    set_target_properties(atk_mobilenet_object_classification PROPERTIES 
        INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO/LTO not supported: <${error}>")
endif()

# 设置优化标志
target_compile_options(atk_mobilenet_object_classification PRIVATE
    -fno-strict-aliasing          # 允许类型转换优化
    -fvisibility=hidden          # 隐藏符号以优化链接
    -fdata-sections             # 优化数据段
    -ffunction-sections         # 优化函数段
    -fexceptions               # 启用异常处理
)

# 链接器优化
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--as-needed")