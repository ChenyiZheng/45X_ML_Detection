from setuptools import setup, find_packages

setup(
    name='MECH_45X_Detection',
    version=1.0,
    packages=find_packages(),
    url='https://github.com/ChenyiZheng/45X_ML_Detection',
    author='Henry Situ, Chenyi Zheng, MECH 45X Team 10',
    description='The detection project used to detect fire, smoke and heat sources from a thermal and visual feed.',
    install_requires=[
        'https://github.com/ultralytics/yolov5',
    ],
)