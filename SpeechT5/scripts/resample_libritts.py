from pathlib import Path
from shutil import copyfile
import multiprocessing as mp
from tqdm import tqdm

import soundfile as sf
import librosa

#LibriTTS
# 1.6G    /root/data/libritts/LibriTTS/dev-clean
# 1.5G    /root/data/libritts/LibriTTS/test-clean
# 9.1G    /root/data/libritts/LibriTTS/train-clean-100
# 33G     /root/data/libritts/LibriTTS/train-clean-360
# 44G     /root/data/libritts/LibriTTS

#LibriTTS_16k

SRC_ROOT = Path("/home/ec2-user/data/raw/LibriTTS")
DST_ROOT = Path("/home/ec2-user/data/raw/LibriTTS_16k")
TARGET_SR = 16000

# Try to use soxr if available for faster resampling.
try:
    import soxr  # type: ignore
    _HAS_SOXR = True
except Exception:
    _HAS_SOXR = False


def _resample_audio(audio, sr, target_sr):
    if sr == target_sr:
        return audio
    if _HAS_SOXR:
        return soxr.resample(audio, sr, target_sr)
    return librosa.resample(audio, sr, target_sr)


def _process_one(path_str):
    src_path = Path(path_str)
    if not src_path.is_file():
        return

    dst_path = Path(str(src_path).replace("LibriTTS", "LibriTTS_16k"))
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.suffix.lower() == ".wav":
        audio, fs = sf.read(str(src_path))
        audio = _resample_audio(audio, fs, TARGET_SR)
        sf.write(str(dst_path), audio, TARGET_SR)
    else:
        copyfile(str(src_path), str(dst_path))


def main():
    DST_ROOT.mkdir(exist_ok=True)
    # Stream files instead of building a huge list of Paths in memory.
    files = (str(p) for p in SRC_ROOT.rglob("*") if p.is_file())
    # We still need a total for tqdm; compute once, then stream again.
    total = sum(1 for _ in SRC_ROOT.rglob("*") if _.is_file())
    print(f"Found {total} files in total.")

    # Use a process pool; keep tqdm in the parent process.
    workers = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_one, files, chunksize=16),
            desc="Resampling",
            total=total,
        ):
            pass


if __name__ == "__main__":
    main()
