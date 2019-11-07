# System modules
from queue import Queue
from threading import Thread
import time
from playsound import playsound

# Local modules
import feedparser

# Set up some global variables
num_fetch_threads = 1
enclosure_queue = Queue()


def downloadEnclosures(i, q):
    """This is the worker thread function.
    It processes items in the queue one after
    another.  These daemon threads go into an
    infinite loop, and only exit when
    the main thread ends.
    """
    while True:
        playsound('fast.wav')
        time.sleep(1)
        q.task_done()


# Set up some threads to fetch the enclosures
for i in range(num_fetch_threads):
    worker = Thread(target=downloadEnclosures, args=(i, enclosure_queue,))
    worker.setDaemon(True)
    worker.start()

# Download the feed(s) and put the enclosure URLs into
# the queue.
for url in range(3):
    enclosure_queue.put(url)
        
# Now wait for the queue to be empty, indicating that we have
# processed all of the downloads.
print ('*** Main thread waiting')
enclosure_queue.join()
print ('*** Done')