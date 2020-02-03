class Observable(object):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def run_callback(self, name, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, name)(*args, **kwargs)
