PROVIDES = [ "WorkerInfo" ]

import threading, queue, time

WORKER_QUEUE = queue.Queue()
STOP = threading.Event()
THREAD = threading.Thread(daemon=True)

def _init():
    THREAD._target = worker_loop
    THREAD.start()
    print("Worker started.")

def _update(bb):
    event = str(time.time())[-3:]
    print(f"Put event {event} in queue..")
    WORKER_QUEUE.put(f"Event from Blackboard {event}")
    time.sleep(0.3)

def worker_loop():
    while not STOP.is_set():
        while not WORKER_QUEUE.empty():
            print("Worker got:", WORKER_QUEUE.get())
        time.sleep(1)

def _close():
    STOP.set()
    if THREAD is not None:
        THREAD.join()
    print("Worker stopped cleanly.")

