import pandas as pd
import wandb

from ever.core.dist import main_process_only


class ScoreTracker:
    def __init__(self):
        self._data = {'step': []}

    def append(self, scores: dict, step):
        self._data['step'].append(step)
        for k, v in scores.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

        if wandb.run is not None:
            wandb.log(scores, step=step)

    @property
    def scores(self):
        return self._data

    @main_process_only
    def to_csv(self, path):
        pd.DataFrame(data=self._data).to_csv(path, index=False)

    def highest_score(self, name):
        if len(self) == 0:
            return {'step': -1, name: float('-inf')}
        idx = self._data[name].index(max(self._data[name]))
        return {k: v[idx] for k, v in self._data.items()}

    def lowest_score(self, name):
        if len(self) == 0:
            return {'step': -1, name: float('inf')}
        idx = self._data[name].index(min(self._data[name]))
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return len(self._data['step'])
