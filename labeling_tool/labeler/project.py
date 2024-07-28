import os
import json

class Project:
    def __init__(self, name: str, project_dir: str, dataset_dir: str, max_backups: int = 10):
        self._name = name
        self._project_dir = project_dir
        self._dataset_dir = dataset_dir

        self._max_backups = max_backups

    @property
    def name(self) -> str:
        return self._name

    @property
    def project_file(self) -> str:
        return os.path.join(self.project_dir, f'{self.name}.json')

    @property
    def dataset_dir(self) -> str:
        return self._dataset_dir

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.project_dir, '.cache')

    @property
    def backup_dir(self) -> str:
        return os.path.join(self.project_dir, '.backup')

    @property
    def max_backups(self) -> int:
        return self._max_backups

    def save(self):
        data = {
            'name': self._name,
            'dataset': self._dataset_dir,
            'max_backups': self.max_backups,
        }

        os.makedirs(self.project_dir, exist_ok=True)
        with open(self.project_file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))

    @staticmethod
    def load(project_file: str) -> 'Project':
        with open(project_file, 'r') as f:
            data = json.load(f)

        return Project(data['name'], os.path.dirname(project_file), data['dataset'], max_backups=data['max_backups'])
