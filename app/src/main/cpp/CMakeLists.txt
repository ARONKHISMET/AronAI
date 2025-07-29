cmake_minimum_required(VERSION 3.18.1)
project(llama_jni LANGUAGES C CXX)

# ==== Общие настройки ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==== Подключение потоков ====
find_package(Threads REQUIRED)

# ==== GGML ====
add_library(ggml STATIC
    ggml/ggml.c
    ggml/ggml-alloc.c
    ggml/ggml-backend.c
)

target_compile_definitions(ggml PRIVATE GGML_USE_PTHREADS)
target_include_directories(ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ggml)
target_link_libraries(ggml PRIVATE Threads::Threads)

# ==== LLaMA ====
add_library(llama STATIC
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache-unified.cpp
    llama-kv-cache-unified-iswa.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampling.cpp
    llama-vocab.cpp
    unicode.cpp
    unicode-data.cpp
)

target_include_directories(llama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml
)
target_link_libraries(llama PUBLIC ggml Threads::Threads)

# ==== JNI Wrapper ====
add_library(llama_jni SHARED
    llama_jni.cpp
)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml
)

target_link_libraries(llama_jni PRIVATE
    llama
    Threads::Threads
)

# ==== Android-specific ====
if(ANDROID)
    message(STATUS "🟢 Configuring for Android")

    target_link_libraries(llama_jni PRIVATE
        log        # __android_log_print
        android    # Native activity, etc.
        atomic     # Atomic operations
        m          # Math
    )
else()
    message(STATUS "🟡 Not Android — native desktop build")
endif()

# ==== Автокопирование моделей ====
# Не обязательно, если используется assets/ через Android API
# Закомментировано как необязательное
# file(GLOB_RECURSE MODEL_FILES "${CMAKE_SOURCE_DIR}/../assets/models/*.gguf")
# foreach(file ${MODEL_FILES})
#     configure_file(${file} ${CMAKE_BINARY_DIR}/assets/models/ COPYONLY)
# endforeach()

# ==== Логирование информации о сборке ====
message(STATUS "🔧 CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "🔧 CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
message(STATUS "🔧 CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
