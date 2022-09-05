#/bin/python3

import argparse
import re
import os
import shlex
import sys

from collections import OrderedDict
from subprocess import check_output

# Set arguments
parser = argparse.ArgumentParser(description='Test driver.')
parser.add_argument('--command', dest='command', type=str, required=True,
                    help='the command that will be run')
parser.add_argument('--expected', dest='expected', type=str, required=True,
                    help='the expected error')

# Parse arguents
args = parser.parse_args()

# Run the test
test_environment = dict(os.environ)
test_environment["LSAN_OPTIONS"] = "detect_leaks=0"

output = check_output(shlex.split(args.command),env=test_environment).decode('ascii', 'ignore')

print()
print(" -------------- TEST OUTPUT --------------")
print(output)
print(" -----------------------------------------")

# Get expected values
expected = args.expected.split(";")

expected_iteration = expected[0]

expected_results = OrderedDict()
expected_results['final_error'] = expected[0]

# Parse the output
for line in iter(output.splitlines()):
    line = line.strip()
    if "Final error:" in line:
        results_string = line.split("Final error:")

        results = OrderedDict()
        results['final_error'] = results_string[1].strip()
        break

# Check the values
print()
print(" -------------- TEST RESULT --------------")

status = 0
for key in results.keys():
    value = float(results[key])

    expected_result = expected_results[key]
    if "@" in expected_result:
        expected_value = float(expected_result.split("@")[0])
        tolerance      = float(expected_result.split("@")[1])

        test_passed = (abs(value - expected_value) < tolerance)
    else:
        expected_value = float(expected_result)

        test_passed = (value == expected_value)

    print(" Checking '%s' variable:" % (key))
    print("    Value          : ", value)
    print("    Expected value : ", expected_value)
    if test_passed:
        print("    Check status   : PASSED")
    else:
        print("    Check status   : FAILED")
        status = 1

print(" -----------------------------------------")

if status == 0:
    print("            TEST PASSED")
else:
    print("            TEST FAILED")
    print(" -----------------------------------------")
    print(" Updated expected results: {0};{1};{2};{3}".format(results['iteration'], results['time'], results['residual_L2'], results['residual_Linf']))

print(" -----------------------------------------")


sys.exit(status)
