#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class LetterClassifier:
    """Runs TFLite CNN to classify individual letter images."""

    def __init__(self, model_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Gather input/output info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Your label set — MODIFY IF you include digits or punctuation
        self.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def predict_letter(self, img):
        """
        img: 200x150 grayscale (uint8 or float32)
        Returns: predicted character
        """

        # Normalize to float32 in [0,1]
        img = img.astype(np.float32) / 255.0

        # Add batch + channel dims: (1, 150, 200, 1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Set tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        # Run model
        self.interpreter.invoke()

        # Obtain output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Argmax → class index
        idx = int(np.argmax(output))

        # Convert to letter
        return self.labels[idx]
