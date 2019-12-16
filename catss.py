import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import librosa
import librosa.display
from audiotsm import phasevocoder, wsola
import speech_recognition as sr
from jiwer import wer
from scipy.io import wavfile
from pypesq import pypesq
import soundfile as sf
import pyrubberband as pyrb
from scipy.spatial import distance
from scipy.io.wavfile import write, read

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 20)


def opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch,
                   usingWords):
    print("First:", output_name)
    if usingWords:
        wav_basename = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.wav'
    else:
        wav_basename = sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.wav'
    full_wav_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/' + wav_basename
    os.system("SMILExtract -C %s -l 2 -I %s -O %s" % (
    configfile, full_wav_path, output_name))  # Can be commented out if you only want to extract metrics

    return sentence_dict


def opensmile_compare_two(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, outputfile,
                          configfile, plot, usingWords):
    # data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'

    # If Prosody is called, first we need to create the stretched audio.
    if configfile == 'opensmile/config/prosodyAcf.conf' or configfile == 'prosodyAcf.conf' or configfile == 'config/prosodyAcf.conf':
        sentence_dict = match_duration(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch,
                                       outputfile, configfile, plot, usingWords)
        if usingWords:
            output_name = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.wav_prosody.csv'
            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)
        else:
            output_name = sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.wav_prosody.csv'
            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)
    else:
        if usingWords:
            output_name = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.csv'
            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)
            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)
        else:
            output_name = sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.csv'
            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)

            sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                           sentencetext, sentencestretch, usingWords)

    return sentence_dict


def pesq(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords):
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    # It's necessary to use the stretched audio in the PESQ calculation. Using POLQA would remove this constraint.
    if usingWords is True:
        reference_speech = data_path + sentencenumber + '_' + 'X_' + sentencetext + '_orig.wav'
        synthetic_speech = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_stretched.wav'
    else:
        reference_speech = data_path + sentencenumber + '_' + 'X_orig.wav'
        synthetic_speech = data_path + sentencenumber + '_' + sentencemodel + '_stretched.wav'

    print("Now calculating PESQ")

    try:
        rate, reference_speech = wavfile.read(reference_speech)
        rate, synthetic_speech = wavfile.read(synthetic_speech)
        try:
            wbpesq = pypesq(rate, reference_speech, synthetic_speech, 'wb')
        except:
            wbpesq = 'error'
    except:
        wbpesq = 'error'
        nbpesq = 'error'

    try:
        wbpesq = pypesq(rate, reference_speech, synthetic_speech, 'wb')
    except:
        wbpesq = 'error'

    try:
        nbpesq = pypesq(rate, reference_speech, synthetic_speech, 'nb')
    except:
        nbpesq = 'error'

    sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['PESQ wideband'] = wbpesq
    sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['PESQ narrowband'] = nbpesq

    return sentence_dict


def plot_overlay(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                 list_of_files_to_process):
    print(list_of_files_to_process)
    for file in list_of_files_to_process:
        if ("_X_orig" in os.path.basename(file)) or (
                ("_X" not in os.path.basename(file) and ("_stretched" in os.path.basename(file)))):
            file = file
            file = file + '_prosody.csv'
            x = []
            y = []
            with open(file, 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=';')
                next(plots)
                for row in plots:
                    try:
                        x.append(float(row[1]))
                    except:
                        break
                    else:
                        y.append(float(row[3]))
                file = file.replace(".wav_prosody.csv", "")
                plt.plot(x, y, label=file)
    plt.legend()
    plt.title('Overlay of Pitch Contours')
    plt.xlabel('Frame Time')
    plt.ylabel('F0')
    plt.show()


def plot_subplot(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                 list_of_files_to_process):
    index = 221
    fig = plt.figure()
    listoflabels = []
    for file in list_of_files_to_process:
        if ("_X_orig" in os.path.basename(file)) or (
                ("_X" not in os.path.basename(file) and ("_stretched" in os.path.basename(file)))):
            file = file + '_prosody.csv'
            x = []
            y = []
            print("This is file:", file)
            with open(file, 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=';')
                next(plots)
                for row in plots:
                    try:
                        x.append(float(row[1]))
                    except:
                        break
                    else:
                        y.append(float(row[3]))
                currentname = str(index) + 'name'
                listoflabels.append(currentname)
                if index == 221:
                    first = fig.add_subplot(index)
                    first.plot(x, y, label=file)
                    first.legend()
                    plt.title('Pitch contours')
                else:
                    currentname = fig.add_subplot(index, sharex=first, sharey=first)
                    currentname.plot(x, y, label=file)
                    currentname.legend()
            index = index + 1
    fig = plt.legend()
    fig = plt.show()


def get_audio_duration(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords):
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    if usingWords:
        file = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.wav'
    else:
        file = sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.wav'
    duration = librosa.get_duration(filename=data_path + file)
    if sentencestretch is 'orig':
        sentence_dict[sentencenumber][sentencemodel][sentencetext]['orig']['Duration'] = duration
    else:
        sentence_dict[sentencenumber][sentencemodel][sentencetext]['stretched']['Duration'] = duration

    return duration, sentence_dict


def match_duration(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, outputfile, configfile,
                   plot, usingWords):
    print(sentencenumber, sentencemodel, sentencetext)
    sentencestretch = 'orig'  # Make sure we use the original files as bases
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    if usingWords:
        reference_speech = data_path + sentencenumber + '_X_' + sentencetext + '_' + sentencestretch + '.wav'
        speech_to_be_stretched = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.wav'
    else:
        reference_speech = data_path + sentencenumber + '_X_' + sentencestretch + '.wav'
        speech_to_be_stretched = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.wav'
    print(sentencetext)

    dur_reference_speech, sentence_dict = get_audio_duration(sentence_dict, sentencenumber, 'X', sentencetext, 'orig',
                                                             usingWords)
    dur_speech_to_be_stretched, sentence_dict = get_audio_duration(sentence_dict, sentencenumber, sentencemodel,
                                                                   sentencetext, 'orig', usingWords)

    if not speech_to_be_stretched.endswith("stretched.wav"):
        sentencestretch = 'stretched'  # Now we use the stretched files/want to name the output as stretched
        if usingWords:
            output_filename = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + 'stretched' + '.wav'
            output_name = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + 'stretched.wav' + '_prosody.csv'
        else:
            output_filename = data_path + sentencenumber + '_' + sentencemodel + '_' + 'stretched' + '.wav'
            output_name = sentencenumber + '_' + sentencemodel + '_' + 'stretched.wav' + '_prosody.csv'

        ratio = dur_reference_speech / dur_speech_to_be_stretched
        sentence_dict[sentencenumber][sentencemodel][sentencetext]['stretched']['Ratio'] = ratio
        ratio = 1 / ratio
        y, sr = sf.read(speech_to_be_stretched)
        y_stretch = pyrb.time_stretch(y, sr, ratio)

        print("Successfully stretched file with a ratio of ", ratio)
        sf.write(output_filename, y_stretch, sr, format="WAV", subtype='PCM_16')
        sentence_dict = opensmile_base(configfile, output_name, sentence_dict, sentencenumber, sentencemodel,
                                       sentencetext, sentencestretch, usingWords)
        dur_reference_speech, sentence_dict = get_audio_duration(sentence_dict, sentencenumber, sentencemodel,
                                                                 sentencetext,
                                                                 sentencestretch, usingWords)

        # Keep these instructions in case the others stop working again
        # Read mono wav file
        # print(speech_to_be_stretched)
        # y2, sr2 = sf.read('/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/Testing/116_AAE.wav')
        # sf.write('/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/Testing/116testing.wav',y2,sr2)
        # y_synth, sr_synth = librosa.load('/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/Testing/116testing.wav',sr=None)
        # sr, y = read(speech_to_be_stretched)
        # Usage from https://github.com/bmcfee/pyrubberband
        # sf.write(os.path.relpath(output_filename, '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/Testing/'), y_stretch, sr)
        # write(output_filename, sr, y_stretch)

    return sentence_dict


def intelligibility(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords):
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio'
    if usingWords is True:
        audio = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + 'orig' + '.wav'
    else:
        audio = data_path + sentencenumber + '_' + sentencemodel + '_' + 'orig' + '.wav'
    print("Now calculating intelligibility.")
    asraudio, sentence_dict = asr(audio, sentence_dict, sentencenumber, sentencemodel, sentencetext, 'orig', usingWords)
    wer_results = wer(sentencetext, asraudio)
    sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['WER'] = wer_results

    return sentence_dict


def asr(audio, sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords):
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    if usingWords is True:
        audiofile = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + 'orig' + '.wav'
    else:
        audiofile = data_path + sentencenumber + '_' + sentencemodel + '_' + 'orig' + '.wav'

    r = sr.Recognizer()
    with sr.AudioFile(audiofile) as source:
        audio = r.record(source)  # read the entire audio file
        # Switch the two following lines to use CMUsphinx instead of Google.
        # recognizedspeech = r.recognize_google(audio)
        try:
            # recognizedspeech = r.recognize_google(audio)
            recognizedspeech = r.recognize_sphinx(audio)
        except:
            # If there's a problem with speech recognition, do not break but write 'unknown'.
            recognizedspeech = 'Unknown'
    sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['asr-text'] = recognizedspeech

    return recognizedspeech, sentence_dict


def print_final_results():
    # Because the default format has many columns but only 5 rows, I transpose them to make it easier to read on the screen.
    if os.path.isfile('0_results.csv'):
        pd.read_csv('0_results.csv', sep=';', header=None).T.to_csv('0_results_vertical.csv', sep=';', header=False,
                                                                    index=False)
    else:
        print("No 0_results file to print.")


def spectrogram(audio_path):
    audio_path = '../Audio/' + audio_path
    y, sampling_rate = librosa.load(audio_path)
    Spectrogram = librosa.feature.melspectrogram(y, sr=sampling_rate, fmax=8000, power=1)
    log_Spectrogram = librosa.power_to_db(Spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_Spectrogram, sr=sampling_rate, x_axis='time', y_axis='mel')
    plt.title('mel spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


def compare_prosody(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                    list_of_files_to_process):
    data_path_ref = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    data_path_synth = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/opensmile/'

    for file in list_of_files_to_process:
        if "_X" in os.path.basename(file):
            if usingWords is True:
                file = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_' + sentencestretch + '.wav'
            else:
                file = sentencenumber + '_' + sentencemodel + '_' + sentencestretch + '.wav'
            output_name = data_path_synth + file + '_prosody.csv'
            prosody_ref = pd.read_csv(output_name, sep=';', error_bad_lines=False)
            prosody_ref = prosody_ref[['frameTime', 'F0_sma']]
            prosody_ref = prosody_ref.dropna(how='any')
            prosody_ref.to_csv(data_path_synth + file + '_prosody_comparison.csv', mode='a', sep=';', header=False,
                               index=False)
            reference_wave = data_path_ref + file
            y_ref, sr_ref = sf.read(reference_wave)
        else:
            if usingWords is True:
                file = sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_stretched.wav'
                reference_wave = data_path_ref + sentencenumber + '_X_' + sentencetext + '_orig.wav'
            else:
                file = sentencenumber + '_' + sentencemodel + '_stretched.wav'
                reference_wave = data_path_ref + sentencenumber + '_X_orig.wav'

            output_name = data_path_synth + file + '_prosody.csv'
            prosody_synth = pd.read_csv(output_name, sep=';', error_bad_lines=False)
            prosody_synth = prosody_synth[['frameTime', 'F0_sma']]
            prosody_synth = prosody_synth.dropna(how='any')
            prosody_synth.to_csv(file + '_prosody_comparison.csv', mode='a', sep=';', header=False, index=False)

            synth = data_path_ref + file

            y_ref, sr_ref = sf.read(reference_wave)
            y_synth, sr_synth = sf.read(synth)

            # Measuring different distance metrics -- kept commented out because dtw is slow
            # dist_euclidean = distance.euclidean(y_ref, y_synth)
            # sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['euclidian'] = dist_euclidean
            # dist_cityblock = distance.cityblock(y_ref, y_synth)
            # sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['cityblock'] = dist_cityblock
            # dist_dtw, path = fastdtw(y_ref, y_synth, dist=euclidean)
            # sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['dtw'] = dist_dtw

    return sentence_dict


def read_audio_spectrum(x, **kwd_args):
    # Code from https://github.com/jhetherly/EnglishSpeechUpsampler/blob/master/plots/plot_comparative_spectrogram.py
    return librosa.core.stft(x, **kwd_args)


def lsd(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords):
    data_path = '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'
    if usingWords is True:
        reference_speech = data_path + sentencenumber + '_X_' + sentencetext + '_orig.wav'
        synthesized_speech = data_path + sentencenumber + '_' + sentencemodel + '_' + sentencetext + '_stretched.wav'
    else:
        reference_speech = data_path + sentencenumber + '_X' + '_orig.wav'
        synthesized_speech = data_path + sentencenumber + '_' + sentencemodel + '_stretched.wav'
    # Code adapted from https://github.com/jhetherly/EnglishSpeechUpsampler/blob/master/plots/plot_comparative_spectrogram.py
    y_ref, sr_ref = librosa.load(reference_speech, sr=None)
    y_synth, sr_synth = librosa.load(synthesized_speech, sr=None)

    reference_speech_spectrogram = read_audio_spectrum(y_ref)
    synthesized_speech_spectrogram = read_audio_spectrum(y_synth)

    reference_speech_spectrogram = np.where(reference_speech_spectrogram == 0, reference_speech_spectrogram + 0.0000001,
                                            reference_speech_spectrogram)
    synthesized_speech_spectrogram = np.where(synthesized_speech_spectrogram == 0,
                                              synthesized_speech_spectrogram + 0.0000001,
                                              synthesized_speech_spectrogram)

    ref_X = np.log10(np.abs(reference_speech_spectrogram) ** 2)
    synth_X = np.log10(np.abs(synthesized_speech_spectrogram) ** 2)

    synth_X_diff_squared = (ref_X - synth_X) ** 2
    synth_lsd = np.mean(np.sqrt(np.mean(synth_X_diff_squared, axis=0)))

    sentence_dict[sentencenumber][sentencemodel][sentencetext]['stretched']['lsd'] = synth_lsd
    sentence_dict[sentencenumber][sentencemodel][sentencetext]['orig']['lsd'] = 0

    return sentence_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_wav', required=False, help='Path to the reference audio (human audio) to use.')
    parser.add_argument('--synthesized_wav', required=False, help='Path to the synthesized audio to use.')
    parser.add_argument('--outputfile', required=True, help='Name to call the output file.')
    parser.add_argument('--configfile', required=False, help='Name of the config file to choose, with path.')
    parser.add_argument('--sentencenumber', required=False,
                        help='Number of the sentence to process in all three models.')
    parser.add_argument('--plot', required=False,
                        help='Which plotting option would you like for pitch contours? Overlay or subplots?')
    parser.add_argument('--sentencetext', required=False,
                        help='To calculate intelligibility, please enter the sentence sentencetext.')
    parser.add_argument('--show_spectrograms', required=True, help='Do you want to see the spectrograms? True/False.')

    args = parser.parse_args()
    outputfile = args.outputfile
    sentencenumber = args.sentencenumber
    plot = args.plot
    show_spectrogram = args.show_spectrograms
    configfile = 'config/' + args.configfile
    sentence_dict = {}
    usingWords = False
    list_of_files_to_process = []
    print()
    print()

    if args.reference_wav is not None:
        reference_wav = args.reference_wav
    if args.synthesized_wav is not None:
        synthesized_wav = args.synthesized_wav
        list_of_files_to_process.append(synthesized_wav)

    if sentencenumber is not None:
        for audio_filename in os.listdir(
                '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'):
            if audio_filename.startswith(sentencenumber):
                list_of_files_to_process.append(audio_filename)
                print(list_of_files_to_process)

    elif (args.reference_wav is None) and (args.synthesized_wav is None):
        for audio_filename in os.listdir(
                '/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/mygithubfiles/CATS/Audio/'):
            if (not audio_filename.endswith("stretched.wav")) and (audio_filename.endswith(".wav")):
                list_of_files_to_process.append(audio_filename)
                print(list_of_files_to_process)

    for file in list_of_files_to_process:
        sentencestretch = 'orig'
        if (not file.endswith("stretched.wav")) and file.endswith(".wav"):
            sentence_start = file.split('.')[0]
            elements = sentence_start.split('_')
            sentencenumber = elements[0]
            sentencemodel = elements[1]
            if usingWords:
                sentencetext = elements[2]
            else:
                if args.sentencetext is not None:
                    sentencetext = args.sentencetext  ## If dealing with sentences, text should have been given.
                else:
                    sentencetext = "Unknown"  # but just in case it's not...

            if sentencenumber not in sentence_dict:
                sentence_dict[sentencenumber] = {}
            else:
                sentence_dict = sentence_dict

            # Create the empty dictionary entries to be filled in later.
            if sentencemodel not in sentence_dict[sentencenumber]:
                sentence_dict[sentencenumber][sentencemodel] = {}
            if sentencetext not in sentence_dict[sentencenumber][sentencemodel]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext] = {}
            sentencestretch = 'stretched'
            if sentencestretch not in sentence_dict[sentencenumber][sentencemodel][sentencetext]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch] = {}
            sentencestretch = 'orig'
            if sentencestretch not in sentence_dict[sentencenumber][sentencemodel][sentencetext]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch] = {}
            duration = 'Duration'
            if duration not in sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch][duration] = {}
            if 'lsd' not in sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['lsd'] = {}
            if 'pesqnb' not in sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]:
                sentence_dict[sentencenumber][sentencemodel][sentencetext][sentencestretch]['pesqnb'] = {}

    for file in list_of_files_to_process:
        # This is very inefficient since you have to go through the files twice, but in my case it was the best way to make
        # sure the dictionary was created.
        sentencestretch = 'orig'
        if file.endswith(".wav"):
            print("Looking at this file now, to process:", file)
            sentence_start = file.split('.')[0]  # strip filename to remove the file type
            elements = sentence_start.split('_')  # strip the filename to get individual elements
            sentencenumber = elements[0]
            sentencemodel = elements[1]

            if usingWords:
                sentencetext = elements[2]
            else:
                if args.sentencetext is not None:
                    sentencetext = args.sentencetext  ## If dealing with sentences, text should have been given.
                else:
                    sentencetext = "Unknown"  # but just in case it's not...

            # Once you have the elements, extract the features and calculate the metrics.
            sentence_dict = opensmile_compare_two(sentence_dict, sentencenumber, sentencemodel, sentencetext,
                                                  sentencestretch, outputfile, configfile, plot, usingWords)

            sentence_dict = lsd(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords)

            sentence_dict = pesq(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch,
                                 usingWords)

            sentence_dict = intelligibility(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch,
                                            usingWords)

    if not usingWords:
        if configfile == 'opensmile/config/prosodyAcf.conf' or configfile == 'config/prosodyAcf.conf' or configfile == 'prosodyAcf.conf':
            print("Now preparing pitch contour comparison.")
            compare_prosody(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                            list_of_files_to_process)
            if plot == 'overlay':
                plot_overlay(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                             list_of_files_to_process)
            elif plot == 'subplot':
                plot_subplot(sentence_dict, sentencenumber, sentencemodel, sentencetext, sentencestretch, usingWords,
                             list_of_files_to_process)

    if show_spectrogram == "True":
        for file in list_of_files_to_process:
            print("Now sending ", file, " to spectrogram.")
            spectrogram(file)

    print(sentence_dict)

    # Build the dictionary
    rows = []
    for sentence in sentence_dict:
        for model in sentence_dict[sentence]:
            for word in sentence_dict[sentence][model]:
                for stretch in sentence_dict[sentence][model][word]:
                    for key, value in sentence_dict[sentence][model][word][stretch].items():
                        rows.append([sentence, model, word, stretch, key, value])

    # Create dataframe from the dictionary, so that you can import it easily into scikit-learn
    df = pd.DataFrame(rows)

    # Code from https://stackoverflow.com/questions/37840043/pandas-unstack-column-values-into-new-columns
    df.columns = ['Sentencenumber', 'modelnumber', 'sentencetext', 'stretched', 'metric', 'value']
    df_nice = (df.pivot_table(index=['Sentencenumber', 'modelnumber', 'sentencetext', 'stretched'],
                              columns='metric',
                              values='value',
                              aggfunc='first')
               .reset_index()
               .rename_axis(None, axis=1))

    # Export the dataframe to csv file
    pd.DataFrame(df_nice).to_csv('0_results_dataframe.csv')

    print(df_nice)


if __name__ == '__main__':
    main()
