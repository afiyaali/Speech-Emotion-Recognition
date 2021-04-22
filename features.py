import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#ALL emotions in RAVDESS dataset
emotions={
  '01':'Neutral',
  '02':'Calm',
  '03':'Happy',
  '04':'Sad',
  '05':'Angry',
  '06':'Fearful',
  '07':'Disgusted',
  '08':'Surprised'
}

def load_data(sound_directory, emotions_to_observe):

    """
    This function will loop through every file inside the RAVDESS directory and will
    load the data in and extract featurse IFF the emotion of the file is within The
    emotions_to_observe list
    sound_directory (str): the directory of sound_data
    emotions_to_observe (list(str)): the emotions that will be used in model training/prediction
    """
    x,y=[],[]
    for file in glob.glob(sound_directory):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in emotions_to_observe:
            continue
        feature=extract_feature(file)
        x.append(feature) #Feature Vector
        y.append(emotion) #Class Variable

    return np.array(x),y

def extract_feature(file_name):

    """
    This function will extract features from the wav file in terms of mfcc, chroma, and melspec,
    mfcc --> Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    chroma --> Pertains to the 12 different pitch classes
    mel --> Mel Spectrogram Frequency
    file_name (str): the filepath corresponding the .wav file that will have it's features extracted
    """
    if not os.path.exists(file_name):
        return "File Doesn't Exist"

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate

        #Processing signal in the time-frequency domain
        try:
            stft=np.abs(librosa.stft(X))
        except Exception as e:
            print(e)
            return

        #Initializing result array of lists
        result=np.array([])
        try:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        except Exception as e:
            print(e)
            return
        try:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        except Exception as e:
            print(e)
            return
        try:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        except Exception as e:
            print(e)
            return
        return result