import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from collections import Counter


skip_values = ['C0300', 'NO_ERROR', 'P0133']

def normalize_percentages_and_formating(column):
    if column.dtype == 'object' and column.str.contains('%').any():
        column = column.replace('%', '', regex=True)
    if column.dtype == 'object' and column.str.contains(',').any():
        column = column.replace(',', '.', regex=True)
    return column

def split_and_duplicate(row, column_name):
    value = row[column_name]
    if value in skip_values:
        return pd.DataFrame([row])

    values = [row[column_name][0:5], row[column_name][5:10], row[column_name][10:15]]

    other_columns = row.drop(column_name).to_dict()

    duplicated_rows = pd.DataFrame({
        column_name: values,
        **other_columns
    })
    return duplicated_rows


def clean_data(input_filename: str, output_filename: str):
    data = pd.read_csv(input_filename, low_memory=False)
    
    data = data.drop(columns=['TIMESTAMP', 'MARK', 'MODEL', 'CAR_YEAR', 'AUTOMATIC',  
                              'BAROMETRIC_PRESSURE(KPA)', 'FUEL_LEVEL', 'AMBIENT_AIR_TEMP', 
                              'MAF', 'LONG TERM FUEL TRIM BANK 2', 'FUEL_TYPE', 'FUEL_PRESSURE',
                              'SPEED', 'SHORT TERM FUEL TRIM BANK 2', 'SHORT TERM FUEL TRIM BANK 1',
                              'ENGINE_RUNTIME', 'DTC_NUMBER', 'EQUIV_RATIO', 'MIN', 'HOURS',
                              'DAYS_OF_WEEK', 'MONTHS', 'YEAR'])
    #'ENGINE_POWER', 'VEHICLE_ID', 'ENGINE_COOLANT_TEMP', 'ENGINE_LOAD',
    #'ENGINE_RPM', 'INTAKE_MANIFOLD_PRESSURE', 'AIR_INTAKE_TEMP',
    #'THROTTLE_POS', #'TROUBLE_CODES', 'TIMING_ADVANCE',
    
    data['TROUBLE_CODES'] = data['TROUBLE_CODES'].fillna('NO_ERROR')
    
    data = data.dropna()

    data['ENGINE_LOAD'] = normalize_percentages_and_formating(data['ENGINE_LOAD'])
    data['THROTTLE_POS'] = normalize_percentages_and_formating(data['THROTTLE_POS'])
    data['TIMING_ADVANCE'] = normalize_percentages_and_formating(data['TIMING_ADVANCE'])
    data['ENGINE_POWER'] = normalize_percentages_and_formating(data['ENGINE_POWER'])

    #new_data = pd.concat(data.apply(lambda row: split_and_duplicate(row, 'TROUBLE_CODES'), axis=1).tolist(), ignore_index=True)
    data.to_csv(output_filename, index=False)

def train_and_save(file_path, drop_columns, train_columns, class_column, model_name, num_of_classes, epoch):
    data = pd.read_csv(file_path, low_memory=False)
    data = data.drop(columns=drop_columns)

    X = data[train_columns]     #Features
    y = data[class_column]      #Target variable

    #Encode target labels
    if y.dtype == 'object':
        le = LabelEncoder()
        le.fit(y)
        print('\033[94m' + str(list(le.classes_)) + '\033[0m')
        y = le.transform(y)
        print('\033[93m' + str(list(le.inverse_transform([0, 1, 2]))) + '\033[0m')
        #y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train / X_train.max()
    X_test = X_test / X_test.max()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='elu', input_dim=X_train.shape[1]),   #Input layer
        tf.keras.layers.Dense(32, activation='elu'),                               #Hidden layer
        tf.keras.layers.Dense(16, activation='elu'),                               #Another hidden layer
        tf.keras.layers.Dense(num_of_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    y_train_onehot = to_categorical(y_train, num_classes=num_of_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_of_classes)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_name,  # Path where the best model will be saved
        monitor='val_accuracy',  # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode='max',  # Save when the monitored metric is maximized
        verbose=1  # Print a message when saving the model
    )

    model.fit(X_train, y_train_onehot, epochs=epoch, batch_size=32,
              validation_data=(X_test, y_test_onehot), callbacks=[checkpoint_callback])

    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test_onehot, y_pred))

    #model.save(model_name)
    print("Model saved")

def local_classify(model_name, csv_data, feature_columns, error_names, columns_to_normalize = []):
    model = tf.keras.models.load_model(model_name)
    df = pd.read_csv(csv_data)

    for column in columns_to_normalize:
        df[column] = normalize_percentages_and_formating(df[column])

    X = df[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)

    threshold = 0.5
    predicted_classes = (predictions > threshold).astype(int)

    predicted_class_names = [[error_names[i] for i, val in enumerate(row) if val == 1] for row in predicted_classes]

    flattened_classes = [item for sublist in predicted_class_names for item in sublist]
    class_counts = Counter(flattened_classes)
    total_classes = len(flattened_classes)
    class_percentages = {class_label: (count / total_classes) * 100 for class_label, count in class_counts.items()}

    for class_label, percentage in class_percentages.items():
        print(f"{class_label}: {percentage:.2f}%")

    # Writes predictions as csv file
    # df['predicted_class_names'] = ['; '.join(classes) for classes in predicted_class_names]
    # df.to_csv('updated_file.csv', index=False)