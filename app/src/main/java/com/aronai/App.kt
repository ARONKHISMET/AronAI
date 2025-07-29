package com.aronai

import android.app.Application
import android.util.Log

class App : Application() {

    override fun onCreate() {
        super.onCreate()

        // ✅ Подключение нативной библиотеки .so (llama_jni.so)
        try {
            System.loadLibrary("llama_jni")  // Без префикса lib и без расширения .so
            Log.i("AppInit", "✅ llama_jni библиотека загружена успешно")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("AppInit", "❌ Ошибка загрузки llama_jni: ${e.message}")
        }
    }
}
