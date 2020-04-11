from setuptools import setup

setup(
    name='torch_kernel',
    version='1.0.0',
    description=('Kernel functions for pytorch'),
    author='Aria Ghora Prabono',
    author_email='hello@ghora.net',
    url='https://github.com/ariaghora/torch_kernel',
    license='MIT',
    packages=['torch_kernel'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'],
    )