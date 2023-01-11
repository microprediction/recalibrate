import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="recalibrate",
    version="0.0.3",
    description="Because everybody gets probabilities wrong",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/recalibrate",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["recalibrate","recalibrate.inclusion"],
    test_suite='pytest',
    tests_require=['pytest','winning>=0.4.9','scikit-learn'],
    include_package_data=True,
    install_requires=['pandas','nevergrad','humpday','scikit-learn'],
    entry_points={
        "console_scripts": [
            "recalibrate=recalibrate.__main__:main",
        ]
    },
)
