#!/usr/bin/env python3

"""
Setup script for the comprehensive quantitative trading framework.

This package provides both Python and Rust trading engines with seamless integration.
"""

from setuptools import setup, find_packages
from setuptools_rust import RustExtension
import os

# Read README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="trading-framework",
    version="1.0.0",
    author="Trading Framework Team",
    author_email="team@trading-framework.com",
    description="High-performance quantitative trading framework with Python and Rust engines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jadenfix/trading_research",

    packages=find_packages(where=".", exclude=["rust_engine", "cpp_engine"]),
    package_dir={"": "."},

    rust_extensions=[
        RustExtension(
            "rust_trading_engine",
            path="rust_engine/Cargo.toml",
            features=["python"],
            debug=False,
        )
    ],

    install_requires=read_requirements('requirements.txt'),

    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "plotly>=4.0",
        ],
        "research": [
            "numpy>=1.21",
            "pandas>=1.3",
            "scipy>=1.7",
            "scikit-learn>=1.0",
            "statsmodels>=0.13",
            "arch>=5.0",
            "quantlib>=1.25",
            "ta-lib>=0.4",
        ],
        "ml": [
            "torch>=1.9",
            "tensorflow>=2.6",
            "xgboost>=1.4",
            "lightgbm>=3.2",
            "catboost>=0.26",
        ],
        "live": [
            "yfinance>=0.1.66",
            "alpha-vantage>=2.3",
            "polygon-api-client>=1.0",
            "ccxt>=3.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "trading-framework=trading_framework.cli:main",
            "tf-demo=trading_framework.demo:main",
            "tf-backtest=trading_framework.backtest:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    keywords=[
        "trading",
        "quantitative",
        "finance",
        "algorithmic",
        "high-frequency",
        "backtesting",
        "risk-management",
        "portfolio-optimization",
        "machine-learning",
        "technical-analysis",
        "rust",
        "performance",
    ],

    python_requires=">=3.8",
    zip_safe=False,

    project_urls={
        "Documentation": "https://trading-framework.readthedocs.io/",
        "Source": "https://github.com/jadenfix/trading_research",
        "Tracker": "https://github.com/jadenfix/trading_research/issues",
        "Changelog": "https://github.com/jadenfix/trading_research/blob/main/CHANGELOG.md",
    },
)
