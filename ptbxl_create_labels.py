import pandas as pd


path_to_original_csv = r'path/to/ptbxl_database/ptbxl_database.csv'
df = pd.read_csv(path_to_original_csv, index_col = 'ecg_id')

labels = []
for i in range(df.scp_codes.shape[0]):     
            if "AFIB" in df.scp_codes.iloc[i]:
                List = ['0', '0', '0', '0', '1', '0']
                labels.append(List)
            elif "STACH" in df.scp_codes.iloc[i]:
                List = ['0', '0', '0', '0', '0', '1']
                labels.append(List)
            elif "SBRAD" in df.scp_codes.iloc[i]:
                List = ['0', '0', '0', '1', '0', '0']
                labels.append(List)
            elif "1AVB" in df.scp_codes.iloc[i]:
                List = ['1', '0', '0', '0', '0', '0']
                labels.append(List)
            elif "CRBBB" in df.scp_codes.iloc[i]:
                List = ['0', '1', '0', '0', '0', '0']
                labels.append(List)
            elif "CLBBB" in df.scp_codes.iloc[i]:
                List = ['0', '0', '1', '0', '0', '0']
                labels.append(List)
            else:
                List = ['0', '0', '0', '0', '0', '0']
                labels.append(List)


header = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
classes = pd.DataFrame(labels, columns = header)

# Create csv with labels of training set
train_df = classes[0:20000]
train_df.to_csv(r'path/to/store/training/labels/ptbxl_train_labels_20k.csv', index = False)

# Create csv with labels of test set 
test_df = classes[20000:]
test_df.to_csv(r'path/to/store/test/labels/ptbxl_test_labels_1837.csv', index = False)


