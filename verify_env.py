import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import tensorflow as tf
import keras
import numpy as np

print("=" * 50)
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

# 1. Versions
print(f"\n[1] TensorFlow version : {tf.__version__}")
print(f"    Keras version       : {keras.__version__}")

# 2. tf.keras accessible
try:
    _ = tf.keras.layers.Dense(10)
    print("\n[2] tf.keras accessible : YES")
except Exception as e:
    print(f"\n[2] tf.keras accessible : NO — {e}")

# 3. preprocess_input import
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    print("[3] preprocess_input import : OK")
except Exception as e:
    print(f"[3] preprocess_input import : FAILED — {e}")

# 4. preprocess_input behavior check
#    Your model has built-in Rescaling + Normalization layers,
#    so preprocess_input is intentionally a no-op.
#    We pass raw [0,255] float32 — the model handles scaling internally.
dummy = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
out = preprocess_input(dummy.copy())
print(f"[4] preprocess_input output range: {out.min():.1f} to {out.max():.1f}")
print(f"    NOTE: This is a no-op by design — your model has built-in")
print(f"    Rescaling + Normalization layers. Pass raw [0,255] values.")

# 5. Model loading
print("\n[5] Loading best_modelnew.h5 ...")
try:
    model = tf.keras.models.load_model("best_modelnew.h5", compile=False)
    print(f"    Model loaded         : OK")
    print(f"    Input shape          : {model.input_shape}")
    print(f"    Output shape         : {model.output_shape}")
    print(f"    Total layers         : {len(model.layers)}")

    # Check first 3 layers
    print(f"\n    First 3 layers:")
    for layer in model.layers[:3]:
        print(f"      - {type(layer).__name__} | {layer.name}")

    # 6. Dummy prediction
    print("\n[6] Running dummy prediction ...")
    preds = model.predict(dummy, verbose=0)
    print(f"    Input shape          : {dummy.shape}")
    print(f"    Raw prediction vector: {preds}")
    print(f"    Predicted index      : {np.argmax(preds)}")
    print(f"    Sum of probabilities : {preds.sum():.6f} (should be ~1.0)")
    print(f"    Prediction valid     : {abs(preds.sum() - 1.0) < 0.001}")

except Exception as e:
    print(f"    FAILED: {e}")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
