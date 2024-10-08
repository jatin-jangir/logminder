import csv
from tqdm import tqdm
import re
import os
import pickle
from setup import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocess:
    """
    Class for preprocessing the log dataset. After preprocessing it will split the dataset into train and val datasets.
    """
    def __init__(self, log_file_path:str, label_file_path:str, directory:str, templates:list ) -> None:
        """Initialises the object of Preprocess class.

        Args:
            log_file_path (str): Path to the log file.
            label_file_path (str): Path to the label file.
            directory (str) : Path to the directory to store computed files to avoid recomputation.
            templates (list): A list of regex to match the templates of the log file. 
        """
        self.log_file_path = log_file_path # Path of the log file.
        self.label_file_path = label_file_path # Path of the label file.
        self.directory = directory # Path to the directory to store computed files toavoid recomputation.
        self.label2id = {"Normal" : 0,
                         "Anomaly" : 1}
        self.blockid2label = {} # A dictionary to map the block ids to the labels.
        self.templates = templates # A list of regex expressions to match each template present in the log file.
        self.blockid2temp_seq = {} # A dictionary maping of a block id to its corresponding sequence of template ids.
        self.template_embs = [] # A list of semantic embeddings of templates.
        self.vocab_size = vocabsize

    def read_labels(self) -> None:
        """ 
        Reads the labels file and updates the block2label dictionary.
        """
        # Loads the label file and creates a dictionary mapping of block id to label.
        with open(self.label_file_path, 'r') as file:
            csv_reader = csv.reader(file) # Reads the csv file.

            for row in csv_reader:
                # In the CSV file, each row represents one block. 
                # First column gives the block id.
                # Second gives the label.
                block_id = row[0]
                label = row[1]
                self.blockid2label[block_id] = label

            # The first row in the csv file is the headers, which we have to remove.
            del(self.blockid2label['BlockId'])

    def parse(self) -> None:
        """
        Parses the log file to create a dictionary maping of block id to template sequecences for each block id.
        Updates the blockid2temp_seq dictionary.
        """
        # Path to store the maping after computation,
        blockid2temp_seq_path = os.path.join(self.directory,blockid2temp_seq_file_name)
        
        # Checking if the blockid2temp_seq file already exits.
        if os.path.exists(blockid2temp_seq_path) and os.path.getsize(blockid2temp_seq_path) !=0:
            print("The files from previous parsing already exists. Loading now.")
            with open(blockid2temp_seq_path, 'rb') as file:
                self.blockid2temp_seq = pickle.load(file)
            return
        
        # Create the maping if the blockid2temp_seq  file doesnt exist.
        blockid_pattern = r'blk_\-?[0-9]*' # Regex pattern to match for block id.
        with open(self.log_file_path, 'r') as read_file:
            for line in tqdm(read_file.readlines()):
                line = line.strip()
                # Looping through each of the cleaned templates.
                for i, template in enumerate(self.templates):
                    # Searching for the template in the current line.
                    if re.search(template, line):
                        # Searching for the block id in the line.
                        blockid = re.search(blockid_pattern, line).group()
                        # Updates the blockid2temp_seq dictionary.
                        if blockid in self.blockid2temp_seq:
                            self.blockid2temp_seq[blockid].append(i)
                        else:
                            self.blockid2temp_seq[blockid] = [i]
                        break

        # Saving the maping.
        print("The parsing result is being written into the following files: ", blockid2temp_seq_path)
        with open(blockid2temp_seq_path, 'wb') as file:
            pickle.dump(self.blockid2temp_seq, file)

    def create_template_embeddings(self) -> None:
        """
        Creates the semantic vector embeddings for the templates.
        """
        # Checking if the template embeddings already exists.
        template_embs_file_path = os.path.join(self.directory, template_embs_file_name)
        if os.path.exists(template_embs_file_path) and os.path.getsize(template_embs_file_path) !=0:
            print("The vector embedings of the templates already exists. Loading them now.")
            with open(template_embs_file_path, 'rb') as file:
                self.template_embs = pickle.load(file)
            return

        # Loading the word2vec representations.
        word2vec = {} # A dictionary to store to the word2vec representations.
        print("Loading the word2vec file.")
        with open(word2vec_file_path, 'r', encoding='utf-8') as read_file:
            for line in tqdm(read_file.readlines()):
                tokens = line.strip().split()
                word = tokens[0] 
                embed = np.asarray(tokens[1:], dtype=np.float64)
                word2vec[word] = embed

        # Cleaning the templates to remove special characters.
        cleaned_templates = [] # A list to store the cleaned templates.
        for template in self.templates:
            # Removes anything that is not alphanum nor space.
            template = re.sub(r'[^\w\s]', ' ', template)
            # Removes extra space.
            template = re.sub(r'\s+', ' ', template).strip()
            cleaned_templates.append(template)

        # Creating a TF-IDF vector represetnation for the cleaned templates.
        print("Creating the TF-IDF matrix.")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_templates)
        # Converting the sparse represntation to a dense one.
        tfidf_matrix = tfidf_matrix.toarray()

        # Creating a dictionary maping of word to its index in the coloumns of tf-idf matrix.
        features = list(vectorizer.get_feature_names_out())
        features = dict(zip(features, range(len(features))))

        print("Creating the vector embeddings for each template.")
        # Looping through the cleaned templates to create its semantic vector embeddings.
        for i, template in enumerate(cleaned_templates):
            template_emb = np.zeros(self.vocab_size) # Initializing with all zeroes.
            for word in template.split(): # Taking each word in the template by itself.
                index = features[word.lower()] # Getting the index of the word in the matrix.
                if not word.isupper(): 
                    # If the word is not all uppercase, add a space after each capital letter,
                    # except the first letter. This is done to convert any snake case words to seperate words.
                    word = re.sub(r'(?<!^)(?=[A-Z])', ' ', word).split()
                
                semantic_emb = np.zeros(self.vocab_size) # Initializing with all zeroes.
                # word will be further split by the above 'if' case if word was written using snake case.
                # Looping through each sub word in word.
                for w in word: 
                    if w in word2vec:
                        semantic_emb += word2vec[w] # Getting the semantic embeddings of each of the subwords.
                semantic_emb /= len(word)
                # Creating the semantic embedding of the 'i'th template wby combing the TF-IDF vector and the
                # semantic embeddings of all the words.
                template_emb += tfidf_matrix[i][index] * semantic_emb 

            self.template_embs.append(template_emb)
        self.template_embs = np.array(self.template_embs)

        # Writing the embeddings to a file.
        print(f"Writing the embedings to the following path: {template_embs_file_path}.")
        with open(template_embs_file_path ,'wb') as file:
            pickle.dump(self.template_embs, file)

    def preprocess(self) -> None:
        # Loading the labels of the blocks.
        print("Loading the label of the blocks.")
        self.read_labels()

        # Creating a dictionary maping of block id to template sequence for each block id.
        print("Parsing the log file to create a dictionary maping of block id to template sequence.")
        self.parse() 

        # Creating a semantic vector representation for the templates.
        print("Creating the semantic vector embeddings for the templates.")
        self.create_template_embeddings()

    def get_train_and_val(self, train_val_split_ratio = 1.0 , val = True) -> None:
        """Uses the vector embeding of the templates to create a dataset and then, if the 'val' argument is True
        splits it into train and val sets depending on the 'train_val_split_ratio'

        Args:
            train_val_split_ratio (float, optional): Ratio with which to split the dataset set. This value corresponds to the 
            proportion of the train set. Defaults to 1.0
            val (bool, optional): Wheater or not to create a val set. Defaults to True.
        """
        
        dataset_X = [] # The list for vector representations of the inputs.
        dataset_Y = [] # The list for labels.

        # Looping through the blockids in the list.
        for blockid in self.blockid2temp_seq:
            # Appeding the label to the list.
            dataset_Y.append(self.label2id[self.blockid2label[blockid]])

            vector_rep = np.zeros(self.vocab_size) # Initializing the representation with all zeroes.
            # Looping through each template id in the sequence.
            for template_id in self.blockid2temp_seq[blockid]:
                vector_rep += self.template_embs[template_id] # Adds up the vector representaion for each template.
           
            dataset_X.append(vector_rep) # Appends the representation to the list. 

        dataset_X = np.array(dataset_X) # Converts to an NP array.
        if val: # Splits to train and val sets, if specified.
            X_train = dataset_X[:round(train_val_split_ratio * len(dataset_X))]
            X_val = dataset_X[round(train_val_split_ratio * len(dataset_X)):]
            
            Y_train = dataset_Y[:round(train_val_split_ratio * len(dataset_Y))]
            Y_val = dataset_Y[round(train_val_split_ratio * len(dataset_Y)):]

            return X_train, Y_train, X_val, Y_val
        
        return dataset_X,dataset_Y



        



