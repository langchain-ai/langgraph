"""Setup script for the langgraph-cli-install package."""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="langgraph-cli-install",
        version="0.1.0",
        description="Simple installer for langgraph-cli",
        author="",
        author_email="",
        license="MIT",
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            "console_scripts": [
                "langgraph-cli-install=langgraph_cli_install.main:main",
            ],
        },
        python_requires=">=3.8",
        install_requires=[
            "uv>=0.1.24",
            "packaging>=23.0",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )