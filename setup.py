from setuptools import setup

long_description = open('README.md').read()

setup(
    name="SpenderQ",
    description="Spectrum autoencoder for Quasar Reconstruction",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.0.0",
    license="MIT",
    author="ChangHoon Hahn",
    author_email="changhoon.hahn@princeton.edu",
    url="https://github.com/changhoonhahn/SpenderQ",
    packages=["spenderq"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    keywords = ['spectroscopy','autoencoder','quasars'],
    install_requires=["torch", "numpy", "accelerate", "torchinterp1d", "astropy", "humanize", "psutil", "GPUtil", "nflows"]
)
