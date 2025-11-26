import os
import re
import nltk
import fire
import glob
import librosa
import pandas as pd
from tqdm import tqdm
from syllables import estimate as syllable_estimate
from typing import Optional


def count_syllables(word: str) -> int:
    """
    Count syllables in a single word using a rule-based approach as a fallback.
    """
    word = word.lower().strip()
    if not word or not word.isalpha():  # Skip non-alphabetic words (e.g., numbers)
        return 0

    # Basic syllable counting rules
    vowels = 'aeiouy'
    syllable_count = 0
    prev_char_was_vowel = False

    for i, char in enumerate(word):
        if char in vowels:
            # Count diphthongs as one syllable (simplified)
            if i > 0 and char == 'i' and word[i-1] in 'aeou':
                continue
            if not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    # Adjust for common English word endings
    if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
        syllable_count = max(1, syllable_count - 1)  # Silent 'e' (e.g., "cake")
    elif word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1  # Words like "table", "bottle"
    elif word.endswith('ed') and len(word) > 2 and word[-3] not in vowels:
        syllable_count = max(1, syllable_count - 1)  # Non-syllabic "ed" (e.g., "walked")

    # Ensure at least one syllable for valid words
    return max(1, syllable_count)


def count_syllables_in_sentence(sentence: str) -> Optional[int]:
    """
    Count total syllables in a given English sentence.
    Returns None if the input is invalid.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return None

    # Remove punctuation and normalize the sentence
    sentence = re.sub(r'[^\w\s-]', '', sentence.lower())

    # Tokenize into words
    words = nltk.word_tokenize(sentence)

    total_syllables = 0
    for word in words:
        # Skip non-alphabetic tokens (e.g., numbers, symbols)
        if not any(c.isalpha() for c in word):
            continue
        # Prefer syllables library for accuracy, fall back to custom function
        try:
            syllables = syllable_estimate(word)
        except (ValueError, AttributeError):
            syllables = count_syllables(word)
        total_syllables += syllables

    return total_syllables


def cal_speechrate(
    audio_path,
    output_file,
    target2transcript,
    sr=16000,
    ):
    audios = sorted(glob.glob(os.path.join(audio_path, "*.wav")))

    results = []
    for audio_file in tqdm(audios):
        try:
            filename = os.path.basename(audio_file)
            text = target2transcript[filename]
            dur = librosa.get_duration(filename=audio_file, sr=sr)
            n_syl = count_syllables_in_sentence(text)
            results.append([filename, text, n_syl, dur, n_syl / dur])
        except:
            continue

    df = pd.DataFrame(results, columns=['File', 'Text', 'NSyl', 'Dur', 'SpeechRate'])
    df.to_csv(output_file)

    speechrate_mean, speechrate_std = df['SpeechRate'].mean(), df['SpeechRate'].std()
    print('='*100, '\nSpeechRate :', speechrate_mean, '±', speechrate_std)

    return [speechrate_mean, speechrate_std]
