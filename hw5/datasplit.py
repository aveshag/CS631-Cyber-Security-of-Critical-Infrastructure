import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



# attach the column names to the dataset
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

print("loading data...")


df = pd.read_csv("kdddata.csv", header=None, names = col_names)


# Transform categorical features into numbers using LabelEncoder()

print("encoding start...")

label_encoder = preprocessing.LabelEncoder()
df['protocol_type']= label_encoder.fit_transform(df['protocol_type'])
df['service']=label_encoder.fit_transform(df['service'])
df['flag']= label_encoder.fit_transform(df['flag'])

print("encoding end.")

labeldf=df['label']

#0:normal
#1:dos
#2:probe
#3:r2l
#4:u2r

newlabeldf=labeldf.replace({ 'normal.' : 0, 'neptune.' : 1 ,'back.': 1, 'land.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1, 'ipsweep.' : 2,'nmap.' : 2,'portsweep.' : 2,'satan.' : 2, 'ftp_write.': 3,'guess_passwd.': 3,'imap.': 3,'multihop.': 3,'phf.': 3,'spy.': 3,'warezclient.': 3,'warezmaster.': 3, 
'buffer_overflow.': 4,'loadmodule.': 4,'perl.': 4,'rootkit.': 4})

df['label'] = newlabeldf

print("spliting...")

train, test = train_test_split(df, test_size=0.25, random_state=57)
                        
df = pd.DataFrame(train)                                 #training data file
df.to_csv("train.csv",index=False)
                      
df_test = pd.DataFrame(test)                                  #test data file
df_test.to_csv("test.csv",index=False)

print("Done!")
