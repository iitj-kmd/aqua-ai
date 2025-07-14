# audio_manager.py
import threading
import time
import io
import pyaudio
import wave
from gtts import gTTS
import ffmpeg


class AudioThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self.message = None
        self.running = True

    def run(self):
        while self.running:
            if self.message:
                message_to_speak = self.message
                self.message = None

                try:
                    # Generate audio in-memory as MP3
                    audio_buffer = io.BytesIO()
                    tts = gTTS(text=message_to_speak, lang='en', slow=False)
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                    # Convert MP3 to WAV in memory using ffmpeg-python
                    # This is the code that replaces pydub and avoids the error
                    process = (
                        ffmpeg
                        .input('pipe:0', format='mp3')
                        .output('pipe:1', format='wav', acodec='pcm_s16le')
                        .run_async(pipe_stdin=True, pipe_stdout=True)
                    )

                    wav_data, _ = process.communicate(input=audio_buffer.read())

                    # Play the WAV data with PyAudio
                    p = pyaudio.PyAudio()
                    wav_stream = io.BytesIO(wav_data)
                    wf = wave.open(wav_stream, 'rb')

                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)

                    chunk_size = 1024
                    data = wf.readframes(chunk_size)
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)

                    stream.stop_stream()
                    stream.close()
                    p.terminate()

                except ffmpeg.Error as e:
                    # This will print FFmpeg's error message if it fails
                    print(f"FFmpeg error: {e.stderr.decode()}")
                except Exception as e:
                    print(f"Error playing audio: {e}")

            time.sleep(0.05)

    def say_message(self, message):
        self.message = message

    def stop(self):
        self.running = False