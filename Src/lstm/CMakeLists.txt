include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)


    include_directories(${OpenCV_INCLUDE_DIRS} )
    cuda_add_executable(sentimentAnalyzer
      ${HEADERS}
      lstm.cu
      sentimentAnalyzer.cpp
      dataLoader.cu
      basicLayer.cu
      gateLayer.cu
      denseLayer.cu
      embedLayer.cu
      util.cu

      OPTIONS --cudart=shared  -arch sm_52 -use_fast_math
      DEBUG -Xptxas -v -g -G 
    )
    target_link_libraries( 
      sentimentAnalyzer

      ${CUDA_CUDA_LIBRARY}
      ${CUDA_CUDART_LIBRARY}
      ${CUDA_CURAND_LIBRARY}
      ${OpenCV_LIBS}
    )

