from setuptools import setup, find_packages

with open("version.txt") as f:
    version = f.read().strip()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="llmops",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    description="LLMOps Manager for deploying and managing LLMs across cloud providers",
    author="LLMOps Team",
)
