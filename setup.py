from setuptools import setup

setup(
    name = 'quantsailor',
    version = '0.0.1',
    author = "quantsailor",
    email = "quantsailor@gmail.com",
    license = "GPL",
    packages = ['quantsailor'],
    py_modules = ['backtest_engine, dataloader, strategy']
)