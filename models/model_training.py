import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

def normalize_percentages_and_formating(column):
    if column.dtype == 'object' and column.str.contains('%').any():
        column = column.replace('%', '', regex=True)
    if column.dtype == 'object' and column.str.contains(',').any():
        column = column.replace(',', '.', regex=True)
    return column


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

    data.to_csv(output_filename, index=False)

def train_and_save(file_path, drop_columns, train_columns, class_column, model_name):
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
        tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),   #Input layer
        tf.keras.layers.Dense(32, activation='relu'),                               #Hidden layer
        tf.keras.layers.Dense(16, activation='relu'),                               #Another hidden layer
        tf.keras.layers.Dense(14, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    y_train_onehot = to_categorical(y_train, num_classes=14)
    y_test_onehot = to_categorical(y_test, num_classes=14)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_name,  # Path where the best model will be saved
        monitor='val_accuracy',  # Metric to monitor (you can use 'accuracy' or 'val_accuracy'
        save_best_only=True,  # Save only the best model
        mode='max',  # Save when the monitored metric is maximized
        verbose=1  # Print a message when saving the model
)

    model.fit(X_train, y_train_onehot, epochs=100, batch_size=32, validation_data=(X_test, y_test_onehot), callbacks=[checkpoint_callback])

    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test_onehot, y_pred))

    #model.save(model_name)
    print("Model saved")


# clean_data('datasets/exp1_14drivers_14cars_dailyRoutes.csv',
#            'datasets/exp1_14drivers_14cars_dailyRoutes_CLEANED.csv')
# train_and_save('datasets/exp1_14drivers_14cars_dailyRoutes_CLEANED.csv',
#                ['VEHICLE_ID'],
#                ['ENGINE_POWER', 'ENGINE_COOLANT_TEMP', 'ENGINE_LOAD', 'ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE',
#                 'AIR_INTAKE_TEMP', 'THROTTLE_POS', 'TIMING_ADVANCE'],
#                 'TROUBLE_CODES',
#                 'models/trouble_code_classifier_tensorflow_100_epochs.keras')
