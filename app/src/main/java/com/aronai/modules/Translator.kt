package com.aronai.modules

class Translator {
    fun translateToEnglish(text: String): String {
        // Заглушка — оффлайн перевод будет на базе Marian-NMT
        return "[EN] $text"
    }

    fun translateToRussian(text: String): String {
        return "[RU] $text"
    }
}
