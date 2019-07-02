import os
import librosa

def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Create sample directory and save output samples to disk
    """
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx, sample in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx+1))
        sample = sample[0]
        librosa.output.write_wav(output_path, sample, fs)