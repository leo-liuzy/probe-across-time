import os
import sys
import csv
import json
for test_type in os.listdir('.'):
    if not os.path.isdir(test_type):
        continue
    if test_type not in ["ReCoRD", "WSC"]:
        continue
    for scenario in os.listdir(test_type):
        if not os.path.isdir(f"{test_type}/{scenario}"):
            continue
        input_file = f"{test_type}/{scenario}/table.csv"
        with open(input_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            is_head = 0
            results = []
            new_header = []
            for row in csv_reader:
                if is_head == 0:
                    # assert len(row) in 3
                    # assert row[-1] == 'accuracy'
                    if len(row) == 5:
                        assert scenario in ["upperbounds", "lowerbounds"]
                        new_header = ["scenario", "test_dataset", "metric_name", "metric"]
                    if len(row) == 6:
                        new_header = ["checkpoint (in update steps)", "test_dataset", "metric_name", "metric"]
                    is_head += 1
                else:
                    assert len(new_header) != 0
                    if len(row) == 5:
                        results.append([row[0], row[2], row[-2], row[-1]])
                    if len(row) == 6:
                        results.append([row[1], row[3], row[-2], row[-1]])
        output_file = f"{test_type}/{scenario}.csv"
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(new_header)
            writer.writerows(results)