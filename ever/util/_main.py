import os


def create_project(path):
    dirs = ['configs', 'module', 'data']
    dirs = [os.path.join(path, d) for d in dirs]
    for d in dirs:
        os.makedirs(d)

    train_script = r"""
import ever as er
    


def train(trainer_name):
    trainer = er.trainer.get_trainer(trainer_name)()
    trainer.run()
        """
    with open(os.path.join(path, 'train.py'), 'w') as f:
        f.write(train_script)

    print('created project in {}'.format(path))

