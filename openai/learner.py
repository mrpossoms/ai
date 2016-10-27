import gym
import posix_ipc as ipc
import struct

class Learner:
    def __init__(self, enviroment_name, learner_id=42):
        self._learner_id = learner_id

        self._scores = []
        self._improvement_rates = []
        self._epoch_info = []

        self._shm_learning = ipc.SharedMemory(name=learner_id, flags=ipc.O_CREAT, size=8)

        self._fp_learning = open(name="{0}_session.txt".format(learner_id), mode="x+")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fp_learning.close()

    def record_epoch(self, score, with_info=None):
        if with_info:
            if self.epoch_count() != len(self._epoch_info):
                raise
            else:
                self._epoch_info += [with_info]

        self._improvement_rates += [score - self._scores[-1]]
        self._scores += [score]

        sample = struct.pack('ff', (self.score(), self.improvement_rate()))
        self._fp_learning.seek(0, ipc.SEEK_SET)
        self._fp_learning.write(sample)

    def epoch_count(self):
        return len(self._scores)

    def scores(self, starting_at=None, to=-1):
        if starting_at is not None:
            return self._scores[starting_at:to]

        return self._scores

    def score(self, at_epoch=-1):
        return self._scores[at_epoch]

    def improvement_rate(self, at_epoch=-1):
        return self._improvement_rates[at_epoch]

    # Inheriting classes should override this
    def info_to_str(self, info):
        return str(info)

    def info(self, starting_at=None, to=-1):
        if starting_at is not None:
            return self._epoch_info[starting_at:to]

        return self._epoch_info
