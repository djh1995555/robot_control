from setuptools import find_packages
from distutils.core import setup

setup(name='robot_control',
      version='1.0.0',
      author='djh',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='djh199512@163.com',
      description='Robot control with numerical and rl method',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
