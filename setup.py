from setuptools import find_packages, setup
from os import path


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "ever", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


install_requires = [
    'numpy',
    'prettytable',
    'Pillow',
    'albumentations>=0.4.2',
    'tensorboard>=1.14',
    'tifffile',
    'scikit-image',
    'scipy',
    'matplotlib',
    'tqdm',
    'pandas'
]
setup(
    name='ever_beta',
    version=get_version(),
    description='A Library for Earth Vision Researcher',
    long_description_content_type='text/plain',
    keywords='Remote Sensing, '
             'Earth Vision, '
             'Deep Learning, '
             'Object Detection, '
             'Semantic Segmentation, '
             'Image Classification '
             'Change Detection',
    packages=find_packages(exclude=['projects', 'tools']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    url='https://github.com/Z-Zheng/ever.git',
    author='Zhuo Zheng',
    author_email='zhengzhuo@whu.edu.cn',
    license='',
    setup_requires=[],
    tests_require=[],
    install_requires=install_requires,
    zip_safe=False
)
