include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS 
*.h
../lstm/*.h 
../lstm/*.cuh)


    include_directories(${OpenCV_INCLUDE_DIRS} )
    cuda_add_executable(test
      ${HEADERS}
      test_main.cu
      test_util.cu
      test_layer.cu
      ../lstm/gateLayer.cu
      ../lstm/basicLayer.cu
      ../lstm/dataLoader.cu
      ../lstm/denseLayer.cu
      ../lstm/embedLayer.cu
      ../lstm/util.cu

 
      OPTIONS --cudart=shared  -arch sm_52 -use_fast_math
      DEBUG -Xptxas -v -g -G 
    )
    target_link_libraries( 
      test

      ${CUDA_CUDA_LIBRARY}
      ${CUDA_CUDART_LIBRARY}
      ${CUDA_CURAND_LIBRARY}
      ${OpenCV_LIBS}
    )

