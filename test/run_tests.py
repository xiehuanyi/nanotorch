import unittest
import os
import sys

if __name__ == "__main__":
    # Add project root (nano-torch) to path so nanotorch package is importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    print(f"Running tests from {os.path.dirname(os.path.abspath(__file__))}")
    
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)
