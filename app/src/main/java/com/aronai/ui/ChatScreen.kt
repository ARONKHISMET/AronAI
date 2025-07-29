package com.aronai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.aronai.modules.LlamaService
import com.aronai.modules.VoiceSTT
import com.aronai.modules.VoiceTTS
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Composable
fun ChatScreen() {
    val context = LocalContext.current
    val llamaService = remember { LlamaService(context) }
    val voiceTTS = remember { VoiceTTS(context) }

    var input by remember { mutableStateOf("") }
    var chat by remember { mutableStateOf("🤖 AronAI: Привет! Я готов помочь.\n") }
    var isProcessing by remember { mutableStateOf(false) }

    val coroutineScope = rememberCoroutineScope()

    val voiceSTT = remember {
        VoiceSTT(context) { recognizedText ->
            input = recognizedText
        }
    }

    Column(
        modifier = Modifier
            .padding(16.dp)
            .fillMaxSize()
    ) {
        Text(
            text = chat,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = input,
            onValueChange = { input = it },
            label = { Text("Введите ваш запрос") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    val prompt = input.trim()
                    if (prompt.isNotEmpty()) {
                        isProcessing = true
                        coroutineScope.launch(Dispatchers.IO) {
                            val response = llamaService.generateResponse(prompt)
                            chat += "\n👤 Вы: $prompt\n🤖 AronAI: $response\n"
                            voiceTTS.speak(response)
                            input = ""
                            isProcessing = false
                        }
                    }
                },
                modifier = Modifier.weight(1f),
                enabled = !isProcessing
            ) {
                Text(if (isProcessing) "..." else "Отправить")
            }

            Button(
                onClick = { voiceSTT.startListening() },
                modifier = Modifier.weight(1f),
                enabled = !isProcessing
            ) {
                Text("🎧 Распознать")
            }

            Button(
                onClick = { voiceTTS.speak("Привет, чем могу помочь?") },
                modifier = Modifier.weight(1f),
                enabled = !isProcessing
            ) {
                Text("🎙️ Озвучить")
            }
        }
    }
}
