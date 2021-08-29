from setuptools import setup

setup(
    name = 'quantnote',
    version = '0.0.2',
    author = "quantnote",
    email = "quantsailor@gmail.com",
    license = "GPL",
    packages = ['quantnote'],
    py_modules = ['backtest_engine, dataloader, strategy']
)