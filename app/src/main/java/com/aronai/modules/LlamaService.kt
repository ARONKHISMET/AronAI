package com.aronai.modules

import android.content.Context
import com.aronai.app.LlamaBridge
import android.util.Log
import java.io.File

class LlamaService(context: Context) {

    private var initialized = false

    init {
        try {
            val file = File(context.filesDir, "mistral.gguf")
            if (!file.exists()) {
                context.assets.open("models/mistral.gguf").use { input ->
                    file.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            initialized = LlamaBridge.initModel(file.absolutePath)
            Log.i("LlamaService", if (initialized) "✅ Модель загружена" else "❌ Ошибка загрузки")
        } catch (e: Exception) {
            Log.e("LlamaService", "❌ Исключение: ${e.message}")
        }
    }

    fun generateResponse(prompt: String): String {
        return if (initialized) {
            try {
                LlamaBridge.generateText(prompt)
            } catch (e: Exception) {
                "❌ Ошибка генерации: ${e.message}"
            }
        } else {
            "❌ Модель не инициализирована."
        }
    }
}
