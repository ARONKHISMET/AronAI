// llama_jni.cpp — исправленная и работоспособная версия JNI (замените прежний файл)
#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "llama.h"

static llama_model *model = nullptr;
static llama_context *ctx = nullptr;

// helper: sample index from probability distribution
static int sample_from_probs(const std::vector<double> &probs) {
    double r = ((double)rand() / (double)RAND_MAX) * std::accumulate(probs.begin(), probs.end(), 0.0);
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return (int)i;
    }
    return (int)probs.size() - 1;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_aronai_modules_LlamaBridge_initModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, 0);

    llama_model_params mparams = llama_model_default_params();
    // keep defaults, or tweak via mparams if needed
    model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!model) return JNI_FALSE;

    llama_context_params cparams = llama_context_default_params();
    // разумное фиксированное значение потоков (избегаем std::thread::hardware_concurrency проблем на некоторых NDK)
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    ctx = llama_init_from_model(model, cparams);
    return ctx ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_aronai_modules_LlamaBridge_generateText(JNIEnv *env, jobject, jstring inputText) {
    if (!model || !ctx) {
        return env->NewStringUTF("❌ Ошибка: модель не инициализирована.");
    }

    const char *input = env->GetStringUTFChars(inputText, 0);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        env->ReleaseStringUTFChars(inputText, input);
        return env->NewStringUTF("❌ Ошибка: не удалось получить словарь модели.");
    }

    const int n_ctx = (int)llama_n_ctx(ctx);
    const int n_predict = 64;           // можно параметризовать
    const int top_k = 40;
    const double top_p = 0.95;
    const double temp = 0.7;
    const double repeat_penalty = 1.1;

    // 1) токенизация
    std::vector<llama_token> embd_inp(n_ctx + 8); // запас
    int32_t n_input = llama_tokenize(vocab,
                                     input,
                                     (int32_t)strlen(input),
                                     embd_inp.data(),
                                     (int32_t)embd_inp.size(),
                                     true,  // add_special (BOS/EOS если модель настроена)
                                     false  // parse_special
    );

    if (n_input <= 0) {
        env->ReleaseStringUTFChars(inputText, input);
        return env->NewStringUTF("❌ Ошибка токенизации.");
    }

    embd_inp.resize(n_input);

    // 2) оценить вход (prompt)
    llama_batch batch{};
    batch.n_tokens = (int32_t)embd_inp.size();
    batch.token = embd_inp.data();
    batch.embd = nullptr;
    batch.pos = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id = nullptr;
    batch.logits = nullptr;

    if (llama_decode(ctx, batch) < 0) {
        env->ReleaseStringUTFChars(inputText, input);
        return env->NewStringUTF("❌ Ошибка при llama_decode(prompt).");
    }

    // 3) подготовка к генерации
    std::string output;
    std::vector<llama_token> generated; generated.reserve(n_predict);

    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    // seed
    srand((unsigned)time(nullptr));

    for (int step = 0; step < n_predict; ++step) {
        // logits для последнего токена
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;

        // копируем и применяем temperature & repeat_penalty
        std::vector<double> adjusted_logits(vocab_size);
        double max_logit = -1e300;
        for (int i = 0; i < vocab_size; ++i) {
            double l = (double)logits[i] / (double)temp; // temperature
            // простая репит-пениализация: если токен уже в prompt или в generated — делим логит
            for (llama_token t : embd_inp) if (t == i) { l /= repeat_penalty; break; }
            for (llama_token t : generated)  if (t == i) { l /= repeat_penalty; break; }
            adjusted_logits[i] = l;
            if (l > max_logit) max_logit = l;
        }

        // softmax (в рациональной форме) и получение распределения
        // но сначала применим top-k и top-p: найдем кандидатов
        std::vector<int> idxs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) idxs[i] = i;
        // сортируем по adjusted_logits убыв.
        std::partial_sort(idxs.begin(), idxs.begin() + std::min((int)idxs.size(), top_k), idxs.end(),
                          [&](int a, int b){ return adjusted_logits[a] > adjusted_logits[b]; });
        int use_k = std::min((int)idxs.size(), top_k);

        // теперь формируем top-k vector и применим top-p
        struct Pair { int id; double logit; };
        std::vector<Pair> cand;
        cand.reserve(use_k);
        for (int i = 0; i < use_k; ++i) {
            cand.push_back({ idxs[i], adjusted_logits[idxs[i]] });
        }

        // вычислим softmax на этих кандидатов — сначала exp(logit - max)
        double ssum = 0.0;
        std::vector<double> probs(cand.size());
        double local_max = cand.empty() ? 0.0 : cand[0].logit;
        for (size_t i = 0; i < cand.size(); ++i) {
            probs[i] = std::exp(cand[i].logit - local_max);
            ssum += probs[i];
        }
        for (size_t i = 0; i < probs.size(); ++i) probs[i] /= ssum;

        // top-p: отсортируем кандидатов по probs убыв. и оставим минимальное подмножество с суммой >= top_p
        std::vector<size_t> order(probs.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return probs[a] > probs[b]; });
        double cum = 0.0;
        size_t keep = 0;
        for (; keep < order.size(); ++keep) {
            cum += probs[order[keep]];
            if (cum >= top_p) { ++keep; break; }
        }
        if (keep == 0) keep = 1; // всегда хотя бы один

        // нулевая вероятность для остальных
        std::vector<double> final_probs(cand.size(), 0.0);
        double sum_keep = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            final_probs[order[i]] = probs[order[i]];
            sum_keep += final_probs[order[i]];
        }
        if (sum_keep <= 0.0) break;
        for (size_t i = 0; i < final_probs.size(); ++i) final_probs[i] /= sum_keep;

        // семпл
        int chosen_idx = sample_from_probs(final_probs); // индекс в cand
        int chosen_token = cand[chosen_idx].id;

        // добавляем токен в generated и декодируем его (один токен)
        generated.push_back((llama_token)chosen_token);

        // call decode for the new token
        llama_batch b2{};
        b2.n_tokens = 1;
        b2.token = generated.empty() ? nullptr : &generated.back();
        b2.embd = nullptr;
        b2.pos = nullptr;
        b2.n_seq_id = nullptr;
        b2.seq_id = nullptr;
        b2.logits = nullptr;

        if (llama_decode(ctx, b2) < 0) {
            break;
        }

        // convert token -> text piece
        char piece[512];
        int p_len = llama_token_to_piece(vocab, (llama_token)chosen_token, piece, (int)sizeof(piece), 0, false);
        if (p_len > 0) {
            output.append(piece, p_len);
        }

        // stop on eos
        if (llama_vocab_eos(vocab) == (llama_token)chosen_token) break;
    }

    env->ReleaseStringUTFChars(inputText, input);
    return env->NewStringUTF(output.c_str());
}

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
