import fire

from ever.util._main import create_project


def create(path):
    create_project(path)


if __name__ == '__main__':
    fire.Fire(dict(create=create))
