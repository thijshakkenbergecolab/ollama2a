from setuptools import setup, find_packages
import os


# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A Python library for running Ollama agents with server management"


setup(
    name="ollama2a",
    version="0.1.0",
    author="Thijs Hakkenberg",
    author_email="thijs.hakkenberg@ecolab.com",
    description="A Python library for running Ollama agents with automated server management",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thijshakkenbergecolab/ollama2a",  # Replace with your repo URL
    project_urls={
        "Bug Reports": "https://github.com/thijshakkenbergecolab/ollama2a/issues",
        "Source": "https://github.com/thijshakkenbergecolab/ollama2a",
        "Documentation": "https://github.com/thijshakkenbergecolab/ollama2a#readme",
    },
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: agents",
        "Topic :: a2a",
        "Topic :: a2a-protocol",
        "Topic :: a2a-server",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic-ai>=0.0.14",
        "ollama>=0.5.1,<1.0.0",
        "fasta2a>=0.3.5,<1.0.0",
        "psutil>=5.8.0,<8.0.0",
        "a2a-sdk",
        "openai>=1.0.0,<2.0.0",
        "httpx>=0.24.0,<1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
    keywords="ollama ai agents llm pydantic-ai a2a",
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "ollama2a": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
