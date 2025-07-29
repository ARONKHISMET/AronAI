package com.aronai.app

object LlamaBridge {

    init {
        System.loadLibrary("llama_jni")
    }

    @JvmStatic external fun generateText(prompt: String): String
    @JvmStatic external fun runLlama(prompt: String): String
    @JvmStatic external fun initModel(path: String): Boolean
    @JvmStatic external fun initBackend()
    @JvmStatic external fun getSystemInfo(): String
}
