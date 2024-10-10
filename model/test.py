import csv
import re
import pickle
import numpy as np
import tensorflow as tf

def read_labels(label_file_path: str) -> dict:
    """Reads the labels file and creates a dictionary maping of blockid to label.
    
    Args:
        label_file_path (str): Path to the label file.

    Returns:
        dict: Dictionary maping of blockid to label.
    """
    blockid2label = {} # Dictionary maping of blockid to label.

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

    # The first row in the csv file is the headers, which we have to remove.
    del(blockid2label['BlockId'])
    return blockid2label

def create_template_seq (log_lines: list) -> dict:
    """Creates a dictionary maping of blockid to a list of template ids.

    Args:
        log_lines (list): A list of log lines.

    Returns:
        dict: A dictionary maping of blockid to a list of template ids.
    """
    blockid_pattern = r'blk_\-?[0-9]*' # Regex pattern to match for block id.
    # A list of regex expressions to match each template.
    templates = [r'.*?Receiving block.*?src:.*?dest:.*',
             r'.*?Received block.*?of size.*?from.*',
             r'.*?PacketResponder.*?for block.*?terminating.*',
             r'.*?BLOCK\* NameSystem.*?addStoredBlock: blockMap updated:.*?is added to.*?size.*',
             r'.*?BLOCK\* NameSystem.*?allocateBlock:.*',
             r'.*?Verification succeeded for.*',
             r'.*?Adding an already existing block.*',
             r'.*?Served block.*?to.*',
             r'.*?Got exception while serving.*?to.*',
             r'.*?Received block.*?src:.*?dest:.*?of size.*',
             r'.*?writeBlock.*?received exception.*',
             r'.*?PacketResponder.*?for block.*?Interrupted.*',
             r'.*?PacketResponder.*?Exception.*',
             r'.*?:Exception writing block.*?to mirror.*',
             r'.*?Receiving empty packet for block.*',
             r'.*?Exception in receiveBlock for block.*',
             r'.*?Changing block file offset of block.*?from.*?to.*?meta file offset to.*',
             r'.*?:Transmitted block.*?to.*',
             r'.*?:Failed to transfer.*?to.*?got.*',
             r'.*?Starting thread to transfer block.*?to.*',
             r'.*?Reopen Block.*',
             r'.*?Unexpected error trying to delete block.*?BlockInfo not found in volumeMap.*',
             r'.*?Deleting block.*?file.*',
             r'.*?BLOCK\* NameSystem.*?delete:.*?is added to invalidSet of.*',
             r'.*?BLOCK\* Removing block.*?from neededReplications as it does not belong to any file.*',
             r'.*?BLOCK\* ask.*?to replicate.*?to.*',
             r'.*?BLOCK\* NameSystem.*?addStoredBlock: Redundant addStoredBlock request received for.*?on.*?size.*',
             r'.*?BLOCK\* NameSystem.*?addStoredBlock: addStoredBlock request received for.*?on.*?size.*?But it does not belong to any file.*',
             r'.*?PendingReplicationMonitor timed out block.*']
    blockid2temp_seq = {} # A dictionary maping of blockid to a list of template ids.

    for line in log_lines:
        line = line.strip()
        # Looping through each of the templates.
        for i, template in enumerate(templates):
            # Searching for the template in the current line.
            if re.search(template, line):
                # Searching for the block id in the line.
                blockid = re.search(blockid_pattern, line).group()
                # Updates the blockid2temp_seq dictionary.
                if blockid in blockid2temp_seq:
                    blockid2temp_seq[blockid].append(i)
                else:
                    blockid2temp_seq[blockid] = [i]
                break
    
    return blockid2temp_seq

def preprocess(log_lines: list, label_file_path: str, template_embeddings_file_path: str, label2id: dict) -> tuple:
    """Preprocess the log lines to create dataset_X and dataset_Y for model prediction.

    Args:
        log_lines (list): A list of log lines.
        label_file_path (str): Path to the labels file.
        template_embeddings_file_path (str): Path to the template embeddings file.
        label2id (dict): A dictionary maping of label to id.

    Returns:
        tuple: (datset_X, dataset_Y)
    """
    #blockid2label = read_labels(label_file_path) # Gets a dictionary maping of blockid2labels.
    blockid2temp_seq = create_template_seq(log_lines) # Gets a dictionary maping of blockid to a list of templates ids.

    # Reading the template embeddings.
    with open(template_embeddings_file_path, 'rb') as file:
        template_embs = pickle.load(file)

    dataset_X = [] # The list for vector representations of the inputs.
    #dataset_Y = [] # The list for labels.
    vocab_size = len(template_embs[0]) # Vocab size is the dimension of each template embedding.

    # Looping through the blockids in the list.
    for blockid in blockid2temp_seq:
        # Appeding the label to the list.
        #dataset_Y.append(label2id[blockid2label[blockid]])
        vector_rep = np.zeros(vocab_size) # Initializing the representation with all zeroes.
        # Looping through each template id in the sequence.
        for template_id in blockid2temp_seq[blockid]:
            vector_rep += template_embs[template_id] # Adds up the vector representaion for each template.
        
        dataset_X.append(vector_rep) # Appends the representation to the list. 
    dataset_X = np.array(dataset_X) # Converts to an NP array.
    return dataset_X#, dataset_Y

def predict(log_lines: list) -> tuple:
    """Predicts the number of normal and anomalous blocks from the given list of log lines.

    Args:
        log_lines (list): A list of log lines.

    Returns:
        tuple: (num_normal, num_anomaly)
    """
    label_file_path = r'test_label.csv' # Path to labels file.
    template_embeddings_file_path = r'template_embs.pickle' # Path to the template embeddings file.
    label2id = {"Normal" : 0,
                "Anomaly" : 1} # A dictionary maping of label to id.
    model_file_path = r'model.keras' # Path to the trained ML model.
    
    # Preprocessing the log lines to generate datset_X and datset_Y.
    dataset_X = preprocess(log_lines, label_file_path, template_embeddings_file_path, label2id)
    # Loading the trained model.
    model = tf.keras.models.load_model(model_file_path)
   
    # Making predictions.
    Y_pred_probs = model.predict(dataset_X)  # Getting probabilities for each class.
    Y_pred = (Y_pred_probs > 0.5).astype(int) # Converting probability to class labels.

    # Counting normal and anomaly.
    num_normal = np.sum(Y_pred == 0)
    num_anomaly = np.sum(Y_pred == 1)

    return num_normal, num_anomaly  

if __name__ == '__main__':
    log_lines = []
    file_path = r'Dataset\test\test_subfile_57.log'
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            log_lines.append(line)

    num_normal, num_anomaly  = predict(log_lines)
    print("The number of anomalous blocks in the given log file is ", num_anomaly)
    print("The number of normal blocks in the given log file ", num_normal)