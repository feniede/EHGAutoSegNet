from setuptools import setup, find_packages

setup(
    name='EHGAutoSegNet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.3',
        'matplotlib==3.8.2',
        'scipy==1.12.0',
        'tensorflow==2.15.0',
        'scikit-learn==1.4.0',
        # Add other dependencies as needed
    ],
    author='Félix Nieto del Amor',
    author_email='feniede@gmail.com',
    description='Automatic Semantic Segmentation of EHG Recordings by Deep Learning: an Approach to a Screening Tool for Use in Clinical Practice',
    url='https://github.com/feniede/EHGAutoSegNet',
)