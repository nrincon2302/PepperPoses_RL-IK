from setuptools import setup, find_packages

setup(
    name="pepper_rl_ik",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "matplotlib",
        "pandas",
        "scipy",
    ],
)
