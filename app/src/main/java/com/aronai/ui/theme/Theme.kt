package com.aronai.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val NeonColorScheme = darkColorScheme(
    primary = NeonGreen,
    secondary = NeonBlue,
    tertiary = NeonPink,
    background = NeonBlack,
    surface = NeonSurface,
    onPrimary = NeonBlack,
    onSecondary = NeonWhite,
    onBackground = NeonWhite,
    onSurface = NeonWhite
)

@Composable
fun AronAITheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = NeonColorScheme,
        typography = Typography,
        content = content
    )
}
