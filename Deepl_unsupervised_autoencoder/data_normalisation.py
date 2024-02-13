from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def data_normalisation(raw_data, normalisation = 1):
    labels = raw_data[:, 0]
    train_data, test_data, train_labels, test_labels = train_test_split(raw_data, labels, test_size=0.2, random_state=21)
    
    #train_data[2]
    
    
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    
    
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    
    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]
    
    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]
    
    print("train_data :",len(train_data))
    print("test_data :",len(test_data))
    print("normal train_data :",len(normal_train_data))
    print("normal_test_data :",len(normal_test_data))
    print("anormalous train_data :",len(anomalous_train_data))
    print("anormalous test_data :",len(anomalous_test_data))
    
    normal_train_data_labels = normal_train_data[:, 0]
    normal_test_data_labels = normal_test_data[:, 0]
    
    anomalous_train_data_labels = anomalous_train_data[:, 0] 
    anomalous_test_data_labels = anomalous_test_data[:, 0]
    
    
    normal_train_data_wl = normal_train_data[:, 1:]
    normal_test_data_wl = normal_test_data[:, 1:]
    
    anomalous_train_data_wl = anomalous_train_data[:, 1:]
    anomalous_test_data_wl = anomalous_test_data[:, 1:]
    
    min_val_normal_train = tf.reduce_min(normal_train_data_wl)
    max_val_normal_train = tf.reduce_max(normal_train_data_wl)
    
    
    min_val_normal_test = tf.reduce_min(normal_test_data_wl)
    max_val_normal_test = tf.reduce_max(normal_test_data_wl)
    
    min_val_anormalous_train = tf.reduce_min(anomalous_train_data_wl)
    max_val_anormalous_train = tf.reduce_max(anomalous_train_data_wl)
    
    min_val_anormalous_test = tf.reduce_min(anomalous_test_data_wl)
    max_val_anormalous_test = tf.reduce_max(anomalous_test_data_wl)
    
    if normalisation == 0 :
    
        print('no normalisation ')
        normal_train_data_wl_normalize = tf.constant(normal_train_data_wl, dtype=tf.float32)
        normal_test_data_wl_normalize = tf.constant(normal_test_data_wl, dtype=tf.float32)
        
        anomalous_train_data_wl_normalize = tf.constant(anomalous_train_data_wl, dtype=tf.float32)
        anomalous_test_data_wl_normalize = tf.constant(anomalous_test_data_wl, dtype=tf.float32)
    
    if normalisation == 1 :
        print('normalisation StandardScaler ')

        scaler = StandardScaler()
        
        normal_train_data_wl_normalize = scaler.fit_transform(normal_train_data_wl)
        normal_test_data_wl_normalize = scaler.fit_transform(normal_test_data_wl) 
        anomalous_train_data_wl_normalize =  scaler.fit_transform(anomalous_train_data_wl)
        anomalous_test_data_wl_normalize =  scaler.fit_transform(anomalous_test_data_wl)
        
        
        normal_train_data_wl_normalize = tf.constant(normal_train_data_wl_normalize, dtype=tf.float32)
        normal_test_data_wl_normalize = tf.constant(normal_test_data_wl_normalize, dtype=tf.float32)
        anomalous_train_data_wl_normalize = tf.constant(anomalous_train_data_wl_normalize, dtype=tf.float32)
        anomalous_test_data_wl_normalize = tf.constant(anomalous_test_data_wl_normalize, dtype=tf.float32)
        
    if normalisation == 2:
        print('normalisation MinMaxScaler')

        scaler = MinMaxScaler()
        
        normal_train_data_wl_normalize = scaler.fit_transform(normal_train_data_wl)
        normal_test_data_wl_normalize = scaler.fit_transform(normal_test_data_wl)
        anomalous_train_data_wl_normalize =  scaler.fit_transform(anomalous_train_data_wl)
        anomalous_test_data_wl_normalize =  scaler.fit_transform(anomalous_test_data_wl)
        
        normal_train_data_wl_normalize = tf.constant(normal_train_data_wl_normalize, dtype=tf.float32)
        normal_test_data_wl_normalize = tf.constant(normal_test_data_wl_normalize, dtype=tf.float32)
        anomalous_train_data_wl_normalize = tf.constant(anomalous_train_data_wl_normalize, dtype=tf.float32)
        anomalous_test_data_wl_normalize = tf.constant(anomalous_test_data_wl_normalize, dtype=tf.float32)
        
    if normalisation == 3:
        
        print('simple normalisation ')
        
        normal_train_data_wl_normalize = (normal_train_data_wl - min_val_normal_train)/ (max_val_normal_train - min_val_normal_train)
        normal_test_data_wl_normalize = (normal_test_data_wl - min_val_normal_train)/ (max_val_normal_test- min_val_normal_test) 
        anomalous_train_data_wl_normalize = (anomalous_train_data_wl - min_val_normal_train)/ (max_val_anormalous_train-          min_val_anormalous_train)
        anomalous_test_data_wl_normalize = (anomalous_test_data_wl - min_val_normal_train)/ (max_val_anormalous_test- min_val_anormalous_test)
    
        normal_train_data_wl_normalize = tf.constant(normal_train_data_wl_normalize, dtype=tf.float32)
        normal_test_data_wl_normalize = tf.constant(normal_test_data_wl_normalize, dtype=tf.float32)
        anomalous_train_data_wl_normalize = tf.constant(anomalous_train_data_wl_normalize, dtype=tf.float32)
        anomalous_test_data_wl_normalize = tf.constant(anomalous_test_data_wl_normalize, dtype=tf.float32)
        
        
    return normal_train_data_wl_normalize, normal_test_data_wl_normalize, anomalous_train_data_wl_normalize, anomalous_test_data_wl_normalize,normal_test_data_labels

    
