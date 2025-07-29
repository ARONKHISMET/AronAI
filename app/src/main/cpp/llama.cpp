#include <jni.h>
#include <string>

// Подключаем LLaMA/ggml
#include "llama-impl.h"
#include "llama-chat.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <ctime>
#include <vector>
#include <stdexcept>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// Простейшая JNI-функция для проверки связки
extern "C"
JNIEXPORT jstring JNICALL
Java_com_aronai_app_LlamaBridge_runLlama(JNIEnv* env, jobject, jstring prompt) {
    const char* input = env->GetStringUTFChars(prompt, nullptr);
    std::string result = "[LLaMA JNI] Введённый prompt: ";
    result += input;
    env->ReleaseStringUTFChars(prompt, input);
    return env->NewStringUTF(result.c_str());
}

// Инициализация бэкенда GGML
extern "C"
JNIEXPORT void JNICALL
Java_com_aronai_app_LlamaBridge_initBackend(JNIEnv* env, jobject) {
    ggml_time_init();
    struct ggml_init_params params = { 0, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    ggml_free(ctx);
}

// Вывод информации о системе
extern "C"
JNIEXPORT jstring JNICALL
Java_com_aronai_app_LlamaBridge_getSystemInfo(JNIEnv* env, jobject) {
    static std::string s;
    s.clear();
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto* reg = ggml_backend_reg_get(i);
        auto* get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature* features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }
    return env->NewStringUTF(s.c_str());
}
