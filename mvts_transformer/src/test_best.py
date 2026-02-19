from utils.utils import get_command_of_best_loss, append_mean_std
import argparse
import subprocess
import shlex
import os
import re
import sys

parser = argparse.ArgumentParser(description="Find comment for row with best (smallest) loss in an .xls file.")
parser.add_argument("--records_file", type=str, help="Path to the .xls file")
parser.add_argument("--specific_seeds", type=int, nargs='*', default=[0, 1, 2])
parser.add_argument("--override_patience", type=int, default=-1)

args = parser.parse_args()

# Parse records file, pull out command with best validation loss
command = get_command_of_best_loss(args.records_file)
print("BEST VAL - COMMAND", command)

# Old command strings contained a comment followed by COMMAND: before the real command.
# Remove the stuff before/including COMMAND:
if "COMMAND: " in command: 
    command = command.split("COMMAND: ")[1]

# Change command to validate on TEST set
command = re.sub(r"--val_ratio\s+\d*\.?\d+", "--val_pattern TEST", command)
command = command.replace("--val_temporal_split", "")
command = command.replace("--val_sequential_split", "")

# Change the output file
test_excel = args.records_file.replace(".xls", "_TESTBEST.xls")
i = 2
while os.path.exists(test_excel):  # If file exists, make a new one
    test_excel = args.records_file.replace(".xls", f"_TESTBEST{i}.xls")
    i += 1
command = command.replace(args.records_file, test_excel)

# Change patience if desired
if args.override_patience >= 0:
    print("Override patience")
    command = re.sub('--patience\s+\d+', f'--patience {args.override_patience}', command)

# Run best command with multiple seeds
for SEED in args.specific_seeds:
    seed_command = re.sub('--seed\s+\d+', f'--seed {SEED}', command)

    # Modify note to reflect this is the final run
    seed_command = re.sub(r'(--name\s+)(\S+)', rf'\1\2_TESTSEED={SEED}', seed_command)

    command_args = shlex.split(seed_command)
    command_args.insert(0, "python")
    print("ABOUT TO RUN", command_args)
    sys.stdout.flush()
    result = subprocess.run(command_args, capture_output=False, text=True)

append_mean_std(test_excel)