from setuptools import find_packages, setup

setup_requires = []

install_requires = ["scikit-motionplan"]

setup(
    name="frmax_ros",
    version="0.0.1",
    description="frmax_ros",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(include=["frmax_ros", "frmax_ros.*"]),
)
