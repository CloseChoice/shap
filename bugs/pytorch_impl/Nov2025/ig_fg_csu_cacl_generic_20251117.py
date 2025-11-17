import tensorflow as tf
import numpy as np
import shap

class InputGateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.fc_ii = tf.keras.layers.Dense(2, use_bias=True, activation=None)
        self.fc_hi = tf.keras.layers.Dense(2, use_bias=True, activation=None)
        self.inputs = None
        self.outputs = None

    def call(self, inputs):
        x, h = inputs
        self.inputs = list(inputs)
        x = self.fc_ii(x) + self.fc_hi(h)
        x = self.sigmoid(x)
        self.outputs = x
        return x

# Create model and set weights
model = InputGateModel()
# Input data
x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
h = np.array([[0.0, 0.1, 0.2]], dtype=np.float32)
x_base = np.array([[0.01, 0.02, 0.03]], dtype=np.float32)
h_base = np.array([[0.0, 0.001, 0.0]], dtype=np.float32)

_ = model((x, h))
weights_ii = np.array([[1., 1., 0.],
                       [0.0, 0.0, 0.0]], dtype=np.float32)
bias_ii = np.array([0.2, 0.0], dtype=np.float32)
weights_hi = np.array([[2., 1., 1.],
                       # [0.0, 0.0, 0.0]], dtype=np.float32)
                       [0.0, 0.0, 0.1]], dtype=np.float32) # this breaks the calculation

weights_hi_old = np.array([[2., 1., 1.],
                       [0.0, 0.0, 0.0]], dtype=np.float32)
bias_hi = np.array([0.32, 0.0], dtype=np.float32)

model.fc_ii.set_weights([weights_ii.T, bias_ii.T])  # No .T (no transpose)
model.fc_hi.set_weights([weights_hi.T, bias_hi.T])  # No .T (no transpose)


# SHAP Explainer
exp = shap.DeepExplainer(model, data=[x_base, h_base])
shap_values = exp.shap_values([x, h], check_additivity=False)

# Forward pass
output = model((x, h))
output_base = model((x_base, h_base))

# zii in vector notation
import pdb; pdb.set_trace()
Z_ii_v = (weights_ii * x).T / tf.transpose(tf.matmul(weights_ii, x.T))
Z_ii = (weights_ii * (x - x_base)) / tf.reduce_sum(tf.matmul(weights_ii, x.T) + tf.matmul(weights_hi, h.T) - tf.matmul(weights_hi, h_base.T) - tf.matmul(weights_ii, x_base.T), axis=0)
Z_hi = (weights_hi * (h - h_base)) / tf.reduce_sum(tf.matmul(weights_ii, x.T) + tf.matmul(weights_hi, h.T) - tf.matmul(weights_hi, h_base.T) - tf.matmul(weights_ii, x_base.T), axis=0)
normalized_outputs = (output - output_base) 
# r_x = np.matmul(normalized_outputs, Z_ii)
# r_h = np.matmul(normalized_outputs, Z_hi)
r_x = np.matmul(normalized_outputs, Z_ii)
r_h = np.matmul(normalized_outputs, Z_hi)

import pdb; pdb.set_trace()

# Wie kommt man hierauf? [0.32690227, 0.16345114, 0.16345114] -> hat auf jeden fall mit den weights zu tun
# Das hier sind die outputs in tf backpropagation:
# [0.817574501]
#  [0.768702626]]
assert np.allclose(r_x.squeeze(), shap_values[0].squeeze())
assert np.allclose(r_h.squeeze(), shap_values[1].squeeze())