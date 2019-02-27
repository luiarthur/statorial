import time

class Timer(object):
    """
    Usage:
    with Timer('Model training'):
        time.sleep(2)
        x = 1
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(self.name, end=' ')

        elapsed = time.time() - self.tstart
        print('time: {}s'.format(round(elapsed)))

