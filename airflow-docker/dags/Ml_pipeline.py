# dags/chest_xray_cnn_pipeline.py
import os
from airflow.decorators import task, dag
from datetime import datetime

# ----------------------------------------------------
# Paths & Config
# ----------------------------------------------------
BASE_DATA_DIR = "/opt/airflow/data"
TRAIN_DIR = f"{BASE_DATA_DIR}/train"
VAL_DIR = f"{BASE_DATA_DIR}/val"
TEST_DIR = f"{BASE_DATA_DIR}/test"

MODEL_DIR = "/opt/airflow/models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 2

# ----------------------------------------------------
# 1. Data Validation
# ----------------------------------------------------
@task
def check_data():
    for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset directory: {path}")

    print("Train, Validation, and Test data found.")
    return True

# ----------------------------------------------------
# 2. Data Preparation
# ----------------------------------------------------
@task
def prepare_data(_):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    return {
        "train_samples": train_data.samples,
        "val_samples": val_data.samples,
        "test_samples": test_data.samples,
    }

# ----------------------------------------------------
# 3. CNN Training
# ----------------------------------------------------
@task
def train_cnn(_):
    import mlflow
    import mlflow.tensorflow
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_experiment("cnn_chest_xray")
    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)
    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )


    with mlflow.start_run():
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            verbose=1,
        )

        # 🔹 Log params
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("img_size", IMG_SIZE)

        # 🔹 Log metrics
        mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

        # 🔹 Log model
        mlflow.tensorflow.log_model(model, artifact_path="model")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/cnn_chest_xray.h5"
        model.save(model_path)

    return model_path

#@task
#def train_cnn(_):
#    import tensorflow as tf
#    from tensorflow.keras.models import Sequential
#    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#    from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#    train_gen = ImageDataGenerator(rescale=1.0 / 255)
#    val_gen = ImageDataGenerator(rescale=1.0 / 255)
#
#    train_data = train_gen.flow_from_directory(
#        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
#    )
#    val_data = val_gen.flow_from_directory(
#        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
#    )
#
#    model = Sequential([
#        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
#        MaxPooling2D(2, 2),
#        Conv2D(64, (3, 3), activation="relu"),
#        MaxPooling2D(2, 2),
#        Flatten(),
#        Dense(128, activation="relu"),
#        Dropout(0.5),
#        Dense(1, activation="sigmoid"),
#    ])
#
#    model.compile(
#        optimizer="adam",
#        loss="binary_crossentropy",
#        metrics=["accuracy"],
#    )
#
#    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
#
#    os.makedirs(MODEL_DIR, exist_ok=True)
#    model_path = f"{MODEL_DIR}/cnn_chest_xray.h5"
#    model.save(model_path)
#
#    return model_path

# ----------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------
@task
def evaluate_cnn(model_path):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.models.load_model(model_path)

    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy}")
    return accuracy

# ----------------------------------------------------
# DAG Definition
# ----------------------------------------------------
@dag(
    dag_id="cnn_chest_xray_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["cnn", "medical-ai", "mlops"],
)
def cnn_pipeline_dag():
    data_ok = check_data()
    data_info = prepare_data(data_ok)
    model_path = train_cnn(data_info)
    evaluate_cnn(model_path)

cnn_pipeline_dag()
