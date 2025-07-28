#!/usr/bin/env python3
"""
Test script to verify that packages from your specific flake.nix are properly accessible in PyCharm.
This script is tailored to test pandas, scipy, numpy, langchain, langchain-community,
langchain-ollama, and chromadb.
"""

import sys
import platform
from datetime import datetime


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)


def test_basic_environment():
    """Test basic Python environment information."""
    print_section("BASIC ENVIRONMENT INFO")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Current time: {datetime.now()}")


def test_data_science_packages():
    """Test the data science packages from your flake.nix."""
    print_section("DATA SCIENCE PACKAGES")

    # Test pandas
    try:
        print("Testing pandas...", end=" ")
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")

        # Quick pandas functionality test
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"  Sample DataFrame:\n{df.head()}")
    except ImportError:
        print("❌ pandas is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Test numpy
    try:
        print("\nTesting numpy...", end=" ")
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")

        # Quick numpy functionality test
        arr = np.array([1, 2, 3, 4, 5])
        print(f"  Sample array: {arr}")
        print(f"  Array mean: {np.mean(arr)}")
    except ImportError:
        print("❌ numpy is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Test scipy
    try:
        print("\nTesting scipy...", end=" ")
        import scipy
        print(f"✓ SciPy version: {scipy.__version__}")

        # Quick scipy functionality test
        from scipy import stats
        print(f"  Sample stats calculation: {stats.norm.cdf(0)}")
    except ImportError:
        print("❌ scipy is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")


def test_langchain_packages():
    """Test the LangChain packages from your flake.nix."""
    print_section("LANGCHAIN PACKAGES")

    # Test langchain
    try:
        print("Testing langchain...", end=" ")
        import langchain
        print(f"✓ LangChain version: {langchain.__version__}")
    except ImportError:
        print("❌ langchain is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Test langchain-community
    try:
        print("\nTesting langchain-community...", end=" ")
        import langchain_community
        print(f"✓ LangChain Community version: {langchain_community.__version__}")
    except ImportError:
        print("❌ langchain-community is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Test langchain-ollama
    try:
        print("\nTesting langchain-ollama...", end=" ")
        import langchain_ollama
        print(f"✓ LangChain Ollama version: {langchain_ollama.__version__}")
    except ImportError:
        print("❌ langchain-ollama is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")


def test_chromadb():
    """Test ChromaDB from your flake.nix."""
    print_section("CHROMADB")

    try:
        print("Testing chromadb...", end=" ")
        import chromadb
        print(f"✓ ChromaDB version: {chromadb.__version__}")

        # Basic ChromaDB functionality test
        print("  Testing basic ChromaDB functionality...")
        try:
            client = chromadb.Client()
            print("  ✓ Successfully created ChromaDB client")
        except Exception as e:
            print(f"  ⚠️ Could not create ChromaDB client: {e}")
    except ImportError:
        print("❌ chromadb is not installed")
    except Exception as e:
        print(f"⚠️ Error: {e}")


def test_sys_path():
    """Print sys.path to see where Python is looking for packages."""
    print_section("PYTHON PATH")
    print("Python is looking for packages in these locations:")
    for i, path in enumerate(sys.path, 1):
        print(f"{i}. {path}")


def test_virtual_env():
    """Check if we're running in a virtual environment."""
    print_section("VIRTUAL ENVIRONMENT")

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✓ Running in a virtual environment")
        print(f"  Virtual env path: {sys.prefix}")
    else:
        print("❌ Not running in a virtual environment")


def main():
    """Run all tests."""
    print_section("NIX FLAKE ENVIRONMENT TEST")
    print("Testing if PyCharm is correctly configured with your Nix environment")
    print(f"Test run at: {datetime.now()}")

    test_basic_environment()
    test_virtual_env()
    test_data_science_packages()
    test_langchain_packages()
    test_chromadb()
    test_sys_path()

    print_section("TEST COMPLETE")
    print("If you see your expected packages above, your setup is working!")
    print("If packages are missing, check your flake.nix and PyCharm configuration.")
    print("\nYour flake.nix includes:")
    print("- pandas, scipy, numpy")
    print("- langchain, langchain-community, langchain-ollama")
    print("- chromadb")
    print("\nMake sure all these packages show as installed (✓) above.")


if __name__ == "__main__":
    main()