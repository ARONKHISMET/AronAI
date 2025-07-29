package com.aronai.modules

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.*

class VoiceTTS(context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech = TextToSpeech(context, this)
    private var isReady = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale("ru", "RU"))
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("VoiceTTS", "❌ Русский язык не поддерживается.")
            } else {
                isReady = true
                Log.i("VoiceTTS", "✅ TextToSpeech готов.")
            }
        } else {
            Log.e("VoiceTTS", "❌ Ошибка инициализации TextToSpeech.")
        }
    }

    fun speak(text: String) {
        if (isReady) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        } else {
            Log.e("VoiceTTS", "❗ TTS не готов. Попробуйте позже.")
        }
    }

    fun stop() {
        if (tts.isSpeaking) {
            tts.stop()
        }
    }

    fun shutdown() {
        tts.shutdown()
    }
}
