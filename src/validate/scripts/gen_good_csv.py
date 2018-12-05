# Given a csv that's been gone through with the validation_gui will generate
# a new csv with only true positive results and "maybe seals"
from validate import Parser

p = Parser('../new_file.csv')

rows = p.get_objects()
for row in rows:
    if row.status == "SEAL" or row.status == "MAYBESEAL":
        with open("good.csv", 'a') as file:
            row_str = p.get_row_str(row)
            file.write(row_str)