import time


class StopWatch:
    def __init__(self):
        self.start = time.perf_counter()

    def reset(self):
        self.start = time.perf_counter()

    def stop(self, msg="Time it took: "):
        ret = time.perf_counter() - self.start
        print(msg + str(ret))
        self.reset()
        return ret
