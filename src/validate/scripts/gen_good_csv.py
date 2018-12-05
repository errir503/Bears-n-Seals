# Given a csv that's been gone through with the validation_gui will generate
# a new csv with only true positive results and "maybe seals"
import sys
sys.path.append("../../")
from validate import Parser
file = sys.argv[1]
start = sys.argv[2]

p = Parser(file)
rows = p.get_objects()[start:]

for row in rows:
    if row.status == "SEAL" or row.status == "MAYBESEAL":
        with open("good.csv", 'a') as file:
            row_str = p.get_row_str(row)
            file.write(row_str)
