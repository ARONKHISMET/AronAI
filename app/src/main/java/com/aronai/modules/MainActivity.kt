package com.aronai.app

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.aronai.app.ui.theme.AronAITheme
import com.aronai.ui.ChatScreen
import java.io.File

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Копируем модель из assets в /data/data/.../files/
        val modelPath = copyModelIfNeeded(this)

        // Загружаем модель через JNI
        val isOk = LlamaBridge.initModel(modelPath)
        if (!isOk) {
            Log.e("MainActivity", "❌ Не удалось загрузить модель: $modelPath")
        } else {
            Log.i("MainActivity", "✅ Модель успешно загружена: $modelPath")
        }

        // Отображаем UI
        setContent {
            AronAITheme {
                ChatScreen()
            }
        }
    }

    // Копирование модели из assets в локальный путь (один раз)
    private fun copyModelIfNeeded(context: Context): String {
        val assetPath = "models/mistral.gguf"
        val targetFile = File(context.filesDir, "mistral.gguf")

        if (!targetFile.exists()) {
            try {
                context.assets.open(assetPath).use { input ->
                    targetFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i("MainActivity", "✅ Модель скопирована в ${targetFile.absolutePath}")
            } catch (e: Exception) {
                Log.e("MainActivity", "❌ Ошибка копирования модели: ${e.message}")
            }
        }

        return targetFile.absolutePath
    }
}
