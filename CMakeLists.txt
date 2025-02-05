# @author        Liang Huanyan
# @email         lianghuanyan@alientek.com
# http://www.openedv.com/forum.php

cmake_minimum_required(VERSION 3.8)
message(STATUS "cmake version ${CMAKE_VERSION}")

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

# 调试信息（发布版本可以注释掉）
# add_definitions(-g -ggdb -gdwarf)

# 警告控制
add_definitions(-Wall)                   # 启用所有警告
add_definitions(-Wno-write-strings)      # 忽略字符串警告
add_definitions(-Wno-return-type)        # 忽略返回类型警告
add_definitions(-Wno-sign-compare)       # 忽略符号比较警告

# 添加mongoose源文件
set(MONGOOSE_SOURCES
    mongoose.c
)

set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_features2d opencv_flann opencv_highgui opencv_freetype)
set(RKMEDIA_LIBS easymedia od_share rga rkaiq rknn_api sample_common_isp)
	
include_directories(${SYSROOT})
include_directories(${SYSROOT}/rga)
include_directories(${SYSROOT}/easymedia)
include_directories(${SYSROOT}/rkaiq/uAPI)
include_directories(${SYSROOT}/rkaiq/xcore)
include_directories(${SYSROOT}/rkaiq/algos)
include_directories(${SYSROOT}/rkaiq/common)
include_directories(${SYSROOT}/rkaiq/iq_parser)
include_directories(${SYSROOT}/rknn)
include_directories(.)
include_directories(./include)
include_directories(./include/3rdparty)
include_directories(./include/3rdparty/drm/include/libdrm)
include_directories(./include/3rdparty/rga/include)

add_definitions(-DRKAIQ)

# 修改可执行文件的源文件列表，添加mongoose.c
add_executable(atk_mobilenet_object_classification 
    atk_mobilenet_object_classification.cpp 
    drm_func.c 
    rga_func.c
    ${MONGOOSE_SOURCES}
)

# 添加pthread库依赖（Mongoose需要）
target_link_libraries(atk_mobilenet_object_classification 
    ${OPENCV_LIBS} 
    ${RKMEDIA_LIBS} 
    pthread 
    dl
)

# 链接时优化
set_target_properties(atk_mobilenet_object_classification PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

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