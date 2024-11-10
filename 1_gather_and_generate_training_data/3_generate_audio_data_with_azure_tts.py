import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import azure.cognitiveservices.speech as speechsdk
import itertools as it
from tqdm import tqdm
import os

settings = get_settings()

pitch_ranges = ['low', 'medium', 'high']
rate_ranges = ['slow', 'medium', 'fast']
style_degrees = [ '0.5', '1', '1.5', '2']

do_skip = False
skip_until = ''

wake_word_dir = os.path.join(settings.raw_wake_word_training_dir, 'azure')
negate_words_dir = os.path.join(settings.raw_negative_words_training_dir, 'azure')

if not os.path.exists(wake_word_dir):
    os.makedirs(wake_word_dir)
if not os.path.exists(negate_words_dir):
    os.makedirs(negate_words_dir)

def generate_data(output_dir, words, languages):
    global do_skip

    for language in languages:

        synthesizer = speechsdk.SpeechSynthesizer(speechsdk.SpeechConfig(subscription=settings.azure_key, region=settings.azure_region))
        get_voices_result = synthesizer.get_voices_async(language).get()

        variables = it.product(get_voices_result.voices,                                
                                words,
                                pitch_ranges,
                                rate_ranges,
                                style_degrees
        )

        old_voice_name = ''
        progress_iter = tqdm(variables, f'Generate samples')
        for voice, word, pitch, rate, style_degree in progress_iter:            
            if do_skip:
                if voice.short_name != skip_until:
                    if old_voice_name != voice.short_name:
                        print(f'Skip voice {voice.short_name}')
                        old_voice_name = voice.short_name
                    continue
                do_skip = False

            progress_iter.set_description(f'Generate samples for voice "{voice.short_name}" with word "{word}"')
            for style in voice.style_list:
                if style == 'whispering':
                    continue

                file_name = f'{language}_{voice.short_name}_{style}_{style_degree}_{pitch}_{rate}_{word.replace("?", "_q")}.wav'
                file_path = os.path.join(output_dir, file_name)
                if style == '':
                    if style_degree == '1':
                        synthesize(word, language, voice.short_name, file_path, style, style_degree, pitch, rate)
                else:
                    synthesize(word, language, voice.short_name, file_path, style, style_degree, pitch, rate)

def synthesize(text, language, voice, output_file_path, style, style_degree, pitch, rate):
    speech_config = speechsdk.SpeechConfig(subscription=settings.azure_key, region=settings.azure_region)
    speech_config.speech_synthesis_language = language 
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    prosody = f'<prosody rate="{rate}" pitch="{pitch}">{text}</prosody>'
    express_as = f'<mstts:express-as style="{style}" styledegree="{style_degree}">{prosody}</mstts:express-as>'
    if style == '':
        express_as = f'{prosody}'
        
    ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US"><voice name="{voice}">{express_as}</voice></speak>'
    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled. Status = {result.reason}; Cancel = {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

        exit()

generate_data(wake_word_dir, settings.wake_words, settings.azure_tts_languages)
generate_data(negate_words_dir, settings.negative_words, settings.azure_tts_languages_negative_words)