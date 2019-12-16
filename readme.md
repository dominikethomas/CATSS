# CATSS
Comparative Analysis of Tacotron Speech Synthesis

## Requirements
Install the following as per their individual requirements, all in the same folder.


[ludlows/python-pesq](https://github.com/ludlows/python-pesq)

[audEERING's openSMILE](https://www.audeering.com/opensmile/)

[Muges/audiotsm](https://github.com/Muges/audiotsm)

[Uberi/speech_recognition](https://github.com/Uberi/speech_recognition)

[jitsi/asr-wer](https://github.com/jitsi/asr-wer) 


Please note that openSMILE cannot be used with virtual environments. Graphical capabilities are required for the pitch contour visualizations.

## When you don't have permissions
If you don't have permissions to install on the machine, install the requirements using a virtual environment, and then use the full path to this python directory when calling the CATS code from the commandline. 
Ex: instead of python3 cats.py, use /fullpath/to/folder/python3 cats.py

## Audio files
Save the reference audio files, and audio to be evaluated, in a single folder. Ideally, they will be named as follows:
n_X.wav, n_model1.wav, n_model2.wav etc.
Where n is a sentence number.
The code works 'out-of-the-box' with audio files named as follows: 1_X.wav, 1_AAE.wav, 1_VAE.wav, 1_GST.wav. Otherwise, some code needs to be changed.

## Example of command line operation
`/mount/arbeitsdaten/thesis-dp-1/synthesis/thomasde/update/synthesis/venv/bin/python3 ../cats.py --sentence='119' --outputfile='119' --plot overlay  --text "Different couples react in many different ways to these beginning signs of aging."`

### Other notes
The openSMILE ProsodyAcf.conf configuration file is, by default, very verbose. To change its verbosity level, modify the printLevelStats to lower numbers (0 is completely silent, 6 is the default verbose.)

The results and features files exported from this will be formatted following the German standards, with a decimal comma and periods separating the thousands. On a US/Canadian computer, this causes problems in viewing the files in Excel, for example.
