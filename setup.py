from setuptools import find_packages
from distutils.core import setup

setup(
    name='engineai_training',
    version='1.0.0',
    author='engineai',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='rl@engineai.com.cn',
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'tensorboard',
                      'setuptools==59.5.0',
                      'matplotlib']
)