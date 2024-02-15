import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def data_preperation(feature_analysis=False):
    
    # Download the dataset
    dataframe_1 = pd.read_csv('./data/UNSW_NB15_training-set.csv')
    dataframe_2 = pd.read_csv('./data/UNSW_NB15_testing-set.csv')
    dataframe = pd.concat([dataframe_1, dataframe_2])
    dataframe.head(10)
    print()
    dataframe.shape
    print()
    print('dataframe info')
    dataframe.info()
    
    dataframe.drop(dataframe.columns[dataframe.columns.str.contains('id', case=False)], axis=1, inplace=True)
    print()
    print('dataframe info')
    dataframe.info()
    
    # Überprüfen Sie auf gemischte Typen in den Spalten
    mixed_types = dataframe.apply(lambda x: any(isinstance(val, str) for val in x))
    print()
    # Zeigen Sie die gemischten Typen und ihre Werte an
    print("Gemischte Typen in den Spalten:")
    for column, is_mixed_type in mixed_types.items():
        if is_mixed_type:
            mixed_values = dataframe.loc[dataframe[column].apply(lambda x: isinstance(x, str)), column].unique()
            print(f"Spalte {column} enthält gemischte Typen mit den Werten: {mixed_values}")
    
    sns.histplot(x = dataframe.label)
    
    if feature_analysis == False :
        label_encoder = LabelEncoder()
        dataframe['proto'] = label_encoder.fit_transform(dataframe['proto'])
        dataframe['service'] = label_encoder.fit_transform(dataframe['service'])
        dataframe['state'] = label_encoder.fit_transform(dataframe['state'])
        dataframe['attack_cat'] = label_encoder.fit_transform(dataframe['attack_cat'])
        
        print()
        dataframe.info()
        dataframe = dataframe.astype(float)
        dataframe['label'] = dataframe['label'].astype(int)
        
        label_column = dataframe['label']
        dataframe = dataframe.drop(columns=['label'])
        dataframe.insert(0, 'label', label_column)
        print()
        print('dataframe info')
        dataframe.info()
        #raw_data = encoded_dataframe.values # old data processing
        raw_data = dataframe.values
        return raw_data, dataframe

    if feature_analysis == True :
        nummerisch = dataframe.select_dtypes(include=['float64', 'int64', 'int16', 'int32'])
        corr_matrix = nummerisch.corr()
        plt.figure(figsize=(20, 20))
        
        sns.heatmap(corr_matrix[(corr_matrix >= 0.2) | (corr_matrix <= -0.2)], vmax=1.0, 
                    vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size": 6}, square=True)
        plt.title('Korrelationsmatrix')
        plt.show()
        
        df_num_corr = dataframe.corr(numeric_only=True)['label']
        df_num_corr = df_num_corr[df_num_corr >= 0]
        print(df_num_corr)

        
        golden_features_list = df_num_corr[abs(df_num_corr) > 0].sort_values(ascending=False)
        print("There are {} strongly correlated values with Target:\n{}".format(len(golden_features_list), golden_features_list))
        

        df_not_num = dataframe.select_dtypes(include = ['O'])
        print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))

        dtypes = dataframe.dtypes
        categorical_features = dtypes[dtypes == 'object'].index.tolist()

        for feature in df_not_num:
                sns.boxplot(x=feature, y='label', data=dataframe)
                plt.xticks(rotation=90)  
                plt.show()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        
        
        for i, feature in enumerate(df_not_num):
            row = i // 2  
            col = i % 2   
            sns.boxplot(x=feature, y='label', data=dataframe, ax=axes[row, col])
            axes[row, col].set_title(f'Boxplot für {feature}')
            axes[row, col].tick_params(axis='x', rotation=90)  
            
        # Platz zwischen den Plots hinzufügen
        plt.tight_layout()
        plt.show()
        
        dataframe.drop(dataframe.columns[dataframe.columns.str.contains('attack_cat', case=False)], axis=1, inplace=True)
        dataframe.info()
        
        num_categories_to_display = 4
        
        top_categories = dataframe['proto'].value_counts().nlargest(num_categories_to_display).index
        df_filtered = dataframe[dataframe['proto'].isin(top_categories)]
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='proto', y='label', data=df_filtered)
        plt.title('Top-Kategorien für Protokolle')
        #plt.xticks(rotation=90)  
        plt.show()

        pd.crosstab(index=dataframe['proto'], columns=dataframe['proto'])

        dtypes = dataframe.dtypes
        categorical_features = dtypes[dtypes == 'object'].index.tolist()

        nummerical_features = golden_features_list.index.tolist()
        
        df_selected_features = dataframe[nummerical_features + categorical_features]
        
        df_selected_features.info()

        values_to_encode_proto      = ['tcp', 'udp', 'arp']
        values_to_encode_service    = ['-', 'http', 'ftp', 'ftp-data', 'smtp']
        values_to_encode_state      = ['FIN', 'ACC']
        
        selected_rows_proto = df_selected_features[df_selected_features['proto'].isin(values_to_encode_proto)]
        
        
        # dType: Object to float64
        encoded_dataframe = pd.get_dummies(selected_rows_proto, columns=['proto'], prefix=["proto"], dtype=float)
        selected_rows_service = encoded_dataframe[encoded_dataframe['service'].isin(values_to_encode_service)]
        
        encoded_dataframe = pd.get_dummies(selected_rows_service, columns=['service'], prefix=["service"], dtype=float)
        selected_rows_state = encoded_dataframe[encoded_dataframe['state'].isin(values_to_encode_state)]
        
        encoded_dataframe = pd.get_dummies(selected_rows_state, columns=['state'], prefix=["state"], dtype=float)
        
        encoded_dataframe.sample(10)
        raw_data = encoded_dataframe.values
        return raw_data, encoded_dataframe
