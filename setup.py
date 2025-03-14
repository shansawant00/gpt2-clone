from setuptools import setup, find_packages

setup(
    name='gpt2-clone',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A clone of the GPT-2 model for text generation.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'transformers>=4.0.0',
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)