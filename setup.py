from setuptools import setup

setup(
    name = 'backtest_engine',
    version = '0.0.1',
    author = "quantsailor",
    email = "quantsailor@gmail.com",
    license = "GPL",
    packages = ['backtest_engine'],
    py_modules = ['backtest_engine, dataloader, strategy']
)