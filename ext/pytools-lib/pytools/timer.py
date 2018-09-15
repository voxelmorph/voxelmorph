''' A collection of general python utilities '''

import time

class Timer(object):
    """
    modified from:
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    a helper class for timing
    use:
    with Timer('foo_stuff'):
    # do some foo
    # do some stuff
    as an alternative to 
    t = time.time()
    # do stuff
    elapsed = time.time() - t
    """

    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            if self.name:
                print('[%s]' % self.name, end="")
            print('Elapsed: %6.4s' % (time.time() - self.tstart))
