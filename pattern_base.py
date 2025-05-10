from abc import ABC, abstractmethod

class PatternDetector(ABC):
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def sidebar_params(self, st):
        pass

    @abstractmethod
    def run_detection(self, st, params):
        pass
