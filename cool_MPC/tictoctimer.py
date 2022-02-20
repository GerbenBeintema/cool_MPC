
import time
class Tictoctimer(object):
    def __init__(self):
        self.time_acc = 0
        self.timer_running = False
        self.start_times = dict()
        self.acc_times = dict()
    @property
    def time_elapsed(self):
        if self.timer_running:
            return self.time_acc + time.time() - self.start_t
        else:
            return self.time_acc
    
    def start(self):
        self.timer_running = True
        self.start_t = time.time()
        
    def pause(self):
        self.time_acc += time.time() - self.start_t
        self.timer_running = False
    
    def tic(self,name):
        self.start_times[name] = time.time()
    
    def toc(self,name):
        if self.acc_times.get(name) is None:
            self.acc_times[name] = time.time() - self.start_times[name]
        else:
            self.acc_times[name] += time.time() - self.start_times[name]

    def percent(self):
        elapsed = self.time_elapsed
        R = sum([item for key,item in self.acc_times.items()])
        return ', '.join([key + f' {item/elapsed:.1%}' for key,item in self.acc_times.items()]) +\
                f', others {1-R/elapsed:.1%}'