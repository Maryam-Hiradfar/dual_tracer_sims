# core/protocol.py
# this is the injection protocol definition
# depending on the injection delay (delta_min), the early-window cutoff time is defined
#the numbers in this script need to be tuned according to the actual protocol used and the simulation settings
#t_cut_max_min is the maximum cutoff time for the early-window in minutes (this is the time window for which only the early data is used in the NNLS fitting)
from dataclasses import dataclass

@dataclass
class Protocol:
    delta_min: float                 # FDG injection delay (min)
    t_cut_max_min: float = 25.0      # early-window cutoff for sequential NNLS

    def early_cut(self):
        if self.delta_min > 0:
            return min(self.delta_min, self.t_cut_max_min)
        return 10.0
