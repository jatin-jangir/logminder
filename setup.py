## Paths
log_file_path = r'Dataset\HDFS.log'
label_file_path = r'Dataset\label.csv'
train_log_file_path = r'Dataset\train\train.log'
train_label_file_path = r'Dataset\train\train_label.csv'
test_log_file_path = r'Dataset\test\test.log'
test_label_file_path = r'Dataset\test\test_label.csv'
train_directory = r'Dataset\train'
test_directory = r'Dataset\test'
model_file_path = r'model.keras'

## Values
train_test_split_ratio = 0.8
train_val_split_ratio  = 0.75
vocabsize = 300

## File Names
blockid2temp_seq_file_name = r'blockid2temp_seq.pickle'
template_embs_file_name = r'template_embs.pickle'
word2vec_file_path = r'Dataset\glove.6B.300d.txt'
X_train_file_name = r'X_train.pickle'
Y_train_file_name = r'Y_train.pickle'
X_val_file_name = r'X_val.pickle'
Y_val_file_name = r'Y_val.pickle'
X_test_file_name = r'X_test.pickle'
Y_test_file_name = r'Y_test.pickle'

## Templates
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