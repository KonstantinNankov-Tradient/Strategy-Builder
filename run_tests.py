#!/usr/bin/env python3
"""
Test runner for Strategy Builder system.

Usage:
    python3 run_tests.py           # Run all tests
    python3 run_tests.py -v        # Run with verbose output
"""

import sys
import unittest
import logging
import time

# Suppress INFO logs during testing unless verbose
if '-v' not in sys.argv:
    logging.basicConfig(level=logging.WARNING)


class CustomTestResult(unittest.TextTestResult):
    """Custom test result class that displays test descriptions with ticks."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_descriptions = []

    def startTest(self, test):
        """Called when a test starts."""
        super().startTest(test)
        # Store test info for later display
        test_method = getattr(test, test._testMethodName)
        doc = test_method.__doc__ or test._testMethodName
        # Clean up docstring
        if doc:
            doc = doc.strip().replace('\n', ' ').replace('  ', ' ')
        self.current_test_info = (test._testMethodName, doc, test.__class__.__name__)

    def addSuccess(self, test):
        """Called when a test passes."""
        super().addSuccess(test)
        if hasattr(self, 'current_test_info'):
            name, doc, class_name = self.current_test_info
            print(f"  ✅ {class_name}.{name}: {doc}")

    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        if hasattr(self, 'current_test_info'):
            name, doc, class_name = self.current_test_info
            print(f"  ❌ {class_name}.{name}: {doc}")

    def addError(self, test, err):
        """Called when a test has an error."""
        super().addError(test, err)
        if hasattr(self, 'current_test_info'):
            name, doc, class_name = self.current_test_info
            print(f"  ⚠️  {class_name}.{name}: {doc}")

    def addSkip(self, test, reason):
        """Called when a test is skipped."""
        super().addSkip(test, reason)
        if hasattr(self, 'current_test_info'):
            name, doc, class_name = self.current_test_info
            print(f"  ⏭️  {class_name}.{name}: {doc} (skipped: {reason})")


class CustomTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses our custom result class."""

    def __init__(self, stream=None, descriptions=True, verbosity=1,
                 failfast=False, buffer=False, resultclass=None, warnings=None):
        super().__init__(stream, descriptions, verbosity, failfast, buffer,
                        CustomTestResult, warnings)

    def run(self, test):
        """Run the given test case or test suite."""
        result = super().run(test)

        # Print summary
        print("\n" + "-"*50)
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Skipped: {len(result.skipped)}")

        return result


def run_all_tests():
    """Run all tests in the tests directory."""
    import os

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests')

    # Change to script directory for proper test discovery
    original_dir = os.getcwd()
    os.chdir(script_dir)

    try:
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')

        # Use custom runner for better output
        runner = CustomTestRunner(verbosity=0)  # Set to 0 to suppress default output
        result = runner.run(suite)

        # Return exit code based on test results
        return 0 if result.wasSuccessful() else 1
    finally:
        # Change back to original directory
        os.chdir(original_dir)


def main():
    """Main entry point."""
    print("\n" + "="*50)
    print("  STRATEGY BUILDER TEST SUITE")
    print("="*50 + "\n")

    start_time = time.time()
    exit_code = run_all_tests()
    elapsed = time.time() - start_time

    print("-"*50)
    print(f"  Time elapsed: {elapsed:.2f}s")

    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check output above.")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())