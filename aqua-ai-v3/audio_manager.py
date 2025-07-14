import pyttsx3
import threading
import time

class AudioThread(threading.Thread):
    """
    A separate thread to handle a single, non-blocking audio message
    at a time.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True  # Allows the program to exit even if this thread is running
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.message = None
        self.running = True

    def run(self):
        """
        The main loop for the audio thread. It waits for a message,
        speaks it, and then waits for the next one.
        """
        while self.running:
            if self.message:
                self.lock.acquire()
                try:
                    self.engine = pyttsx3.init()
                    self.engine.say(self.message)
                    self.engine.runAndWait()
                except RuntimeError as e:
                    print(f"Audio engine error during playback: {e}")
                finally:
                    self.lock.release()
                self.message = None  # Clear the message after it's spoken

            # Small delay to prevent the thread from consuming high CPU
            time.sleep(0.01)

    def say_message(self, message):
        """
        Sets the message to be spoken. If a message is already being
        spoken, the new message will replace it.
        """
        with self.lock:
            self.message = message

    def stop(self):
        """
        Gracefully stops the audio thread.
        """
        self.running = False
        self.engine.stop()