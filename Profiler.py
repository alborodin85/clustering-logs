import time


class Profiler:
    def __init__(self):
        self.points = {}
        self.lastTimestamp = time.time_ns()
        self.prodTime = 0
        self.totalTime = 0

    def start(self):
        self.lastTimestamp = time.time_ns()
        print('Profiler: ' + 'start')

    def addPoint(self, pointName: str, isProd=False):
        currentTime = time.time_ns()
        durationNs = currentTime - self.lastTimestamp
        durationSec = durationNs / 1000 / 1000 / 1000
        self.points[pointName] = durationSec
        self.totalTime += durationSec
        if isProd:
            self.prodTime += durationSec
        self.lastTimestamp = currentTime
        print('Profiler: ' + pointName + f' ({durationSec:.3f} c)')

    def print(self):
        print('\ntimeProfiler')
        print(self.points)
        print()
        print(f'total Time: {self.totalTime} s')
        print(f'production Time: {self.prodTime} s')
        print()
