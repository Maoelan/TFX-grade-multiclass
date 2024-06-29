
import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.executor import TrainerFnArgs

LABEL_KEY = "GradeClass"
FEATURE_KEY = [
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "GPA"
]

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=64, num_epochs=None) -> tf.data.Dataset:
    
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
        num_epochs=num_epochs
    )

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()

    return dataset


def get_model():
    input_features = []
    for key in FEATURE_KEY:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )

    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(512, activation="relu")(concatenate)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)

    deep = tf.keras.layers.Dense(256, activation="relu")(deep)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)

    deep = tf.keras.layers.Dense(128, activation="relu")(deep)
    deep = tf.keras.layers.BatchNormalization()(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)

    outputs = tf.keras.layers.Dense(5, activation="softmax")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0005),  # Adjust learning rate as needed
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=64, num_epochs=1)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=64, num_epochs=1)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

    plot_model(
        model,
        to_file="image/model_plot.png",
        show_shapes=True,
        show_layer_names=True
    )
