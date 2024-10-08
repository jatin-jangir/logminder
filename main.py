from Preprocessing import Preprocess
from GRU import GRUModel
from setup import *
import os
import re
from tqdm import tqdm
import csv
import pickle

def split_log_file(log_file_path:str, label_file_path:str ,
                   train_log_file_path:str, train_label_file_path:str, 
                   test_log_file_path:str, test_label_file_path:str, ratio:float) -> None:
    """Splits the log file into train and test log files according to the given ratio.
        Writes the corresponding log and label files for train and test sets.

    Args:
        log_file_path (str): Path to the log file.
        label_file_path (str): Path to the labels file. 
        train_log_file_path (str): Path to which train log file should be written to.
        train_label_file_path (str): Path to which train label file should be written to. 
        test_log_file_path (str): Path to which test log file should be written to.
        test_label_file_path (str): Path to which test label file should be written to.
        ratio (float): Training split ratio. This proportion will be allocated to the training file.
    """

    # Checking if the files already exists.
    if (os.path.exists(train_log_file_path) and os.path.getsize(train_log_file_path)!=0 and
        os.path.exists(test_log_file_path) and os.path.getsize(test_log_file_path)!=0):
        print("The train and test log files already exits.")
        return
    
    blockid2label = {} # A Dictonary which maps block id to its label.
    normal_ids = [] # A list which stores all the block ids with normal label.
    anomaly_ids = [] # A list which stores all the block ids with anomaly label.
    
    # Loads the label file and creates a dictionary mapping of block id to label.
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

    # Picks the normal block ids for train and test logs.
    keys =  list(normal.keys())
    train_normal_keys = keys[:round(train_test_split_ratio * len(keys))]
    test_normal_keys = keys[round(train_test_split_ratio * len(keys)):]

    # Picks the anomaly block ids for train and test logs.
    keys =  list(anomaly.keys())
    train_anomaly_keys = keys[:round(train_test_split_ratio * len(keys))]
    test_anomaly_keys = keys[round(train_test_split_ratio * len(keys)):]

    train_set = {} # A dictionary maping of line numbers to log lines for the train set.
    test_set = {} # A dictionary maping of line numbers to log lines for the test set.


    # Creating a train set by combining the train_anomaly_keys and train_normal_keys.
    # Creating a label file for the train set.
    print(f"The label file for the train log file will be saved at the following path {train_label_file_path}.")
    with open(train_label_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['BlockId', 'Label'])

        for key in train_normal_keys:
            for line in normal[key]:
                train_set[line[0]] = line[1]
            writer.writerow([key, 'Normal'])

        for key in train_anomaly_keys:
            for line in anomaly[key]:
                train_set[line[0]] = line[1]
            writer.writerow([key, 'Anomaly'])

    # Creating a test set by combining the test_anomaly_keys and test_normal_keys.
    # Creating a label file for the test set.
    print(f"The label file for the test log file will be saved at the following path {test_label_file_path}.")
    with open(test_label_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['BlockId', 'Label'])

        for key in test_normal_keys:
            for line in normal[key]:
                test_set[line[0]] = line[1]
            writer.writerow([key, 'Normal'])

        for key in test_anomaly_keys:
            for line in anomaly[key]:
                test_set[line[0]] = line[1]
            writer.writerow([key, 'Anomaly'])

    # Sorts the sets by line number.
    train_set = dict(sorted(train_set.items()))
    test_set = dict(sorted(test_set.items()))

    # Converts the dictionary to list.
    train_set = [value for value in train_set.values()]
    test_set = [value for value in test_set.values()]

    # Writes the train and test log files.
    print(f"The train log file will be saved at the following path {train_log_file_path}.")
    with open(train_log_file_path, 'w', encoding='utf-8') as file:
        for line in train_set:
            file.write(line + '\n')  # Adding a newline after each line

    print(f"The test log file will be saved at the following path {test_log_file_path}.")
    with open(test_log_file_path, 'w', encoding='utf-8') as file:
        for line in test_set:
            file.write(line + '\n')  # Adding a newline after each line

if __name__ == '__main__':
    ################## Splitting the original log file.#####################
    print(f"Splitting the log file into train and test log files with the ratio: {train_test_split_ratio} for train log file.")
    split_log_file(log_file_path, label_file_path,
                    train_log_file_path, train_label_file_path,
                    test_log_file_path, test_label_file_path,train_test_split_ratio)
    
    ################## Preprocessing the train set.#########################
    print("Starting the preprocessing of the train dataset log file.")
    train_preprocessor = Preprocess(train_log_file_path, train_label_file_path, train_directory, templates)
    train_preprocessor.preprocess()
    
    print("Splitting the train dataset into train and val sets.")
    X_train, Y_train,  X_val, Y_val = train_preprocessor.get_train_and_val(train_val_split_ratio, val = True)

    # Saving the sets.
    with open(os.path.join(train_directory, X_train_file_name), 'wb') as file:
        pickle.dump(X_train, file)

    with open(os.path.join(train_directory, Y_train_file_name), 'wb') as file:
        pickle.dump(Y_train, file)

    with open(os.path.join(train_directory, X_val_file_name), 'wb') as file:
        pickle.dump(X_val, file)

    with open(os.path.join(train_directory, Y_val_file_name), 'wb') as file:
        pickle.dump(Y_val, file)

    ################## Preprocessing the test set.##########################
    print("Starting the preprocessing of the test dataset log file.")
    test_preprocessor = Preprocess(test_log_file_path, test_label_file_path, test_directory, templates)
    test_preprocessor.preprocess()

    print("Splitting the test dataset into inputs and outputs.")
    X_test, Y_test = test_preprocessor.get_train_and_val(val = False)

    # Saving the sets.
    with open(os.path.join(test_directory, X_test_file_name), 'wb') as file:
        pickle.dump(X_test, file)

    with open(os.path.join(test_directory, Y_test_file_name), 'wb') as file:
        pickle.dump(Y_test, file)


    
  
