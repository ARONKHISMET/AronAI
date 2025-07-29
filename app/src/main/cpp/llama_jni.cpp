#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"

static llama_model *model = nullptr;
static llama_context *ctx = nullptr;

// Инициализация модели
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aronai_modules_LlamaBridge_initModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, 0);
    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(path, model_params);
    if (!model) {
        env->ReleaseStringUTFChars(modelPath, path);
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx = llama_init_from_model(model, ctx_params);
    env->ReleaseStringUTFChars(modelPath, path);

    return ctx != nullptr ? JNI_TRUE : JNI_FALSE;
}

// ⬇️ ВСТАВЬ СЮДА — прямо после initModel
extern "C" JNIEXPORT jstring JNICALL
Java_com_aronai_modules_LlamaBridge_generateText(JNIEnv *env, jobject, jstring inputText) {
    if (!model || !ctx) {
        return env->NewStringUTF("❌ Ошибка: модель не инициализирована.");
    }

    const char *input = env->GetStringUTFChars(inputText, 0);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    const int n_ctx = llama_n_ctx(ctx);
    const int n_predict = 64;
    const int top_k = 40;
    const float top_p = 0.95f;
    const float temp = 0.7f;
    const float repeat_penalty = 1.1f;

    std::vector<llama_token> embd_inp(n_ctx);
    const int n_input = llama_tokenize(
        vocab,
        input,
        strlen(input),
        embd_inp.data(),
        embd_inp.size(),
        true,
        false
    );

    if (n_input <= 0) {
        env->ReleaseStringUTFChars(inputText, input);
        return env->NewStringUTF("❌ Ошибка токенизации.");
    }

    int n_threads = std::thread::hardware_concurrency(); // или зафиксируй вручную

    llama_eval(ctx, embd_inp.data(), n_input, 0, n_threads);

    std::string output;
    std::vector<llama_token> embd;
    int n_consumed = 0;

    for (int i = 0; i < n_predict; i++) {
        if (!embd.empty()) {
            llama_eval(ctx, embd.data(), embd.size(), n_input + n_consumed, n_threads);
            n_consumed += embd.size();
            embd.clear();
        }

        const float *logits = llama_get_logits(ctx);
        std::vector<llama_token_data> candidates;

        for (int token_id = 0; token_id < llama_n_vocab(model); ++token_id) {
            candidates.push_back({ token_id, logits[token_id], 0.0f });
        }

        llama_token_data_array candidates_p = { candidates.data(), (size_t)candidates.size(), false };

        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
        llama_sample_temp(ctx, &candidates_p, temp);
        llama_sample_repetition_penalty(ctx, &candidates_p, embd_inp.data(), embd_inp.size(), repeat_penalty);

        llama_token new_token = llama_sample_token(ctx, &candidates_p);
        embd.push_back(new_token);

        char token_text[512];
        int len = llama_token_to_piece(vocab, new_token, token_text, sizeof(token_text), false);
        if (len > 0) output.append(token_text, len);

        if (new_token == llama_token_eos(model)) break;
    }

    env->ReleaseStringUTFChars(inputText, input);
    return env->NewStringUTF(output.c_str());
}

// Очистка ресурсов
extern "C" JNIEXPORT void JNICALL
Java_com_aronai_modules_LlamaBridge_freeModel(JNIEnv *, jobject) {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}
