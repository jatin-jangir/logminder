import csv
from setup import *
from tqdm import tqdm
import re
from Preprocessing import Preprocess
import pickle

def gen_tests(log_file_path: Path, label_file_path: Path):
    """Function to genearate seperate test log files according to different anomalous block ratios.

    Args:
        log_file_path (Path): Path of the original test log file.
        label_file_path (Path): Path of the original test label file.
    """

    blockid2label = {} # A Dictonary which maps block id to its label.
    normal_ids = [] # A list which stores all the block ids with normal label.
    anomaly_ids = [] # A list which stores all the block ids with anomaly label.

    with open(label_file_path, 'r') as file:
        csv_reader = csv.reader(file) # Reads the csv file.
        for row in csv_reader:
            # In the CSV file, each row represents one block. 
            # First column gives the block id.
            # Second gives the label.
            block_id = row[0]
            label = row[1]
            blockid2label[block_id] = label
            if label == 'Normal':
                normal_ids.append(block_id)
            elif label == 'Anomaly':
                anomaly_ids.append(block_id)

        # The first row in the csv file is the headers, which we have to remove.
        del(blockid2label['BlockId'])

    # Creates a dictionary maping of block id to a list of tuples with (line_num, line) 
    # for lines of the same block.

    pattern = r'blk_\-?[0-9]*' # Regex pattern to match for block id.
    # A dictionary maping of normal block ids to lines.
    normal = {id:[] for id in normal_ids}  
    # A dictionary maping of anomaly block ids to lines.
    anomaly = {id:[] for id in anomaly_ids}

    # Reading the test log file.
    line_num = 0
    with open(log_file_path, 'r', encoding='utf-8') as read_file:
        for line in tqdm(read_file.readlines()):
            line = line.strip()
            match = re.findall(pattern, line)
            line_num += 1
            if blockid2label[match[0]] == 'Normal':
                normal[match[0]].append((line_num, line))
            elif blockid2label[match[0]] == 'Anomaly':
                anomaly[match[0]].append((line_num, line))

    # Printing statistics.
    print(f"Total Number of blocks: {len(blockid2label)}.")
    print(f"Number of Normal blocks: {len(normal_ids)}.")
    print(f"Number of Anomalous blocks: {len(anomaly_ids)}.")
    print(f"Ratio of Normal blocks: {len(normal_ids)/len(blockid2label):.2f}")
    print(f"Ratio of Anomalous blocks: {len(anomaly_ids)/len(blockid2label):.2f}")
    print()

    anomaly_count = len(anomaly_ids)
    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # The ratios of anomalous blocks.
    normal_counts = [] # A list of number of normal blocks required to satisfy the above ratios.

    # Calculating the number of normal blocks required.
    for ratio in ratios:
        normal_count = anomaly_count * (1-ratio)/ratio
        normal_counts.append(round(normal_count))

    # Printing the normal counts.
    print(f"We are creating test sets with the following anomalous block ratios: {ratios}.")
    print(f"The number of normal blocks required for the corressponding ratios are: {normal_counts}.")
    
    for ratio, normal_count in zip(ratios, normal_counts):
        # Picks the normal block ids.
        keys =  list(normal.keys())
        normal_keys = keys[:normal_count + 1]
        
        # Picks the anomaly block ids.
        anomaly_keys =  list(anomaly.keys())

        test_set = {} # A dictionary maping of line numbers to log lines for the test set.

        # Creating a test set by combining the anomaly_keys and normal_keys.
        # Creating a label file for the test set.
        label_path = ratioed_test_log_file_path /f'{ratio}'/ f'test_label_{ratio}.csv'
        log_path = ratioed_test_log_file_path /f'{ratio}'/ f'test_{ratio}.log'

        with open(label_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['BlockId', 'Label'])

            for key in normal_keys:
                for line in normal[key]:
                    test_set[line[0]] = line[1]
                writer.writerow([key, 'Normal'])

            for key in anomaly_keys:
                for line in anomaly[key]:
                    test_set[line[0]] = line[1]
                writer.writerow([key, 'Anomaly'])

        # Sorts the sets by line number.
        test_set = dict(sorted(test_set.items()))

        # Converts the dictionary to list.
        test_set = [value for value in test_set.values()]

        # Writing the log file.
        with open(log_path, 'w', encoding='utf-8') as file:
            for line in test_set:
                file.write(line + '\n')  # Adding a newline after each line

        test_preprocessor = Preprocess(log_path, label_path, ratioed_test_log_file_path/f'{ratio}', templates)
        test_preprocessor.preprocess()

        #Splitting the test dataset into inputs and outputs.
        X_test, Y_test = test_preprocessor.get_train_and_val(val = False)

        # Saving the sets.
        X_test_path = ratioed_test_log_file_path /f'{ratio}'/ f'X_test_{ratio}.pickle'
        Y_test_path = ratioed_test_log_file_path /f'{ratio}'/ f'Y_test_{ratio}.pickle'

        with open(X_test_path, 'wb') as file:
            pickle.dump(X_test, file)

        with open(Y_test_path, 'wb') as file:
            pickle.dump(Y_test, file)

if __name__ == '__main__':
    # Generating the ratioed test files.
    gen_tests(test_log_file_path, test_label_file_path)