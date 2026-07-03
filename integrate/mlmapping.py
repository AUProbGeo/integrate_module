"""
Simple ML Mapping Module

A simplified Python module for creating neural network mappings between input 
parameters (D_in) and output parameters (D_out) using TensorFlow/Keras.

Functions:
- create_model(D_in, D_out, model_name='model', hidden_layers=3, units=64, activation='relu', ...) -> model_file
- train_model(D_in, D_out, model_file, epochs=1000, batch_size=256, ...) -> history  
- hyperparameter_search(D_in, D_out, model_name='tuned_model', max_trials=10, overwrite=False, use_tensorboard=True, search_method='random', ...) -> best_model_file
- load_model(model_file) -> model_file
- predict(D_in, model_file, batch_size=None) -> D_out
- predict_fast(D_in, model_file, batch_size=None) -> D_out
"""

import os
import numpy as np
import h5py
import time
import pickle

tf = None
kt = None
StandardScaler = None
train_test_split = None


def _ensure_ml_deps():
    """Lazily import tensorflow/scikit-learn/keras_tuner on first use."""
    global tf, kt, StandardScaler, train_test_split
    if tf is not None:
        return
    try:
        import tensorflow as _tf
        import keras_tuner as _kt
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.preprocessing import StandardScaler as _ss
    except ImportError as e:
        raise ImportError(
            "mlmapping requires tensorflow, scikit-learn, and keras_tuner. "
            "Install them with: pip install integrate_module[ml]"
        ) from e
    tf, kt, train_test_split, StandardScaler = _tf, _kt, _tts, _ss


def _prepare_data(D_in, D_out, output_types='auto'):
    """Prepare input and output data for training."""
    _ensure_ml_deps()
    # Process inputs separately
    X_list = []
    input_scalers = []
    input_shapes = [arr.shape[1:] for arr in D_in]

    for i, arr in enumerate(D_in):
        X_reshaped = arr.reshape(arr.shape[0], -1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_list.append(X_scaled)
        input_scalers.append(scaler)

    # Handle output_types as list or single value
    if isinstance(output_types, (list, tuple)):
        types_list = list(output_types)
    else:
        types_list = [output_types] * len(D_out)

    # Process each output separately based on its type
    y_parts = []
    output_scalers = []
    output_shapes = [arr.shape[1:] for arr in D_out]

    for i, (arr, otype) in enumerate(zip(D_out, types_list)):
        y_part = arr.reshape(arr.shape[0], -1)

        if otype == 'classification':
            # For classification, don't scale labels
            unique_vals = np.unique(y_part)

            # Check if already one-hot encoded
            if (np.array_equal(unique_vals, [0, 1]) and
                y_part.shape[1] > 1 and
                np.allclose(np.sum(y_part, axis=1), 1.0)):
                # Already one-hot encoded, use as-is
                y_processed = y_part.copy()
            elif y_part.shape[1] == 1:
                # Single output column - convert to categorical (one-hot encoding)
                num_classes = len(unique_vals)
                y_processed = tf.keras.utils.to_categorical(y_part.flatten(), num_classes=num_classes)
            else:
                # Multiple output columns - assume already in correct format
                y_processed = y_part.copy()

            # No scaling for classification
            output_scalers.append(None)

        else:  # regression
            # For regression, scale outputs
            scaler = StandardScaler()
            y_processed = scaler.fit_transform(y_part)
            output_scalers.append(scaler)

        y_parts.append(y_processed)

    # For mixed types, keep outputs separate; for same types, concatenate
    unique_types = list(set(types_list))
    if len(unique_types) == 1:
        # All same type - concatenate as before
        y_scaled = np.concatenate(y_parts, axis=1)
    else:
        # Mixed types - keep separate for multi-output model
        y_scaled = y_parts  # List of arrays, one per output

    return X_list, y_scaled, input_scalers, output_scalers, input_shapes, output_shapes


def _build_network(input_dims, output_data, options):
    """Build neural network architecture."""
    _ensure_ml_deps()
    # Extract options with defaults
    hidden_layers = options.get('hidden_layers', 3)
    units = options.get('units', 64)
    activation = options.get('activation', 'relu')
    dropout_rate = options.get('dropout_rate', 0.0)
    use_batch_norm = options.get('use_batch_norm', True)
    output_types = options.get('output_types', ['regression'])
    clip_norm = options.get('clip_norm', None)
    i_layer_entrance = options.get('i_layer_entrance')

    if i_layer_entrance is None:
        i_layer_entrance = [0] * len(input_dims)

    if len(i_layer_entrance) != len(input_dims):
        raise ValueError(
            f"Length of i_layer_entrance ({len(i_layer_entrance)}) must match number of inputs ({len(input_dims)})"
        )

    if any(layer_idx < 0 for layer_idx in i_layer_entrance):
        raise ValueError("i_layer_entrance values must be non-negative layer indices.")

    max_entrance = max(i_layer_entrance) if i_layer_entrance else 0
    if max_entrance >= hidden_layers:
        raise ValueError(
            f"Entrance layer {max_entrance} exceeds number of hidden layers ({hidden_layers})"
        )
    
    # Create input layers
    inputs = []
    for i, dim in enumerate(input_dims):
        input_layer = tf.keras.layers.Input(shape=(dim,), name=f'input_{i}')
        inputs.append(input_layer)

    # Map each layer to the inputs that should enter there
    entrance_map = {}
    for idx, layer_idx in enumerate(i_layer_entrance):
        entrance_map.setdefault(layer_idx, []).append(inputs[idx])

    x = None
    for layer_idx in range(hidden_layers):
        current_inputs = []
        if x is not None:
            current_inputs.append(x)
        current_inputs.extend(entrance_map.get(layer_idx, []))

        if not current_inputs:
            raise ValueError(
                f"No inputs available for dense layer {layer_idx}. "
                "Ensure at least one input enters at or before this layer."
            )

        if len(current_inputs) == 1:
            layer_input = current_inputs[0]
        else:
            concat_name = 'input_concat' if layer_idx == 0 else f'concat_{layer_idx}'
            layer_input = tf.keras.layers.Concatenate(name=concat_name)(current_inputs)

        x = tf.keras.layers.Dense(units, activation=activation, name=f'dense_{layer_idx}')(layer_input)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{layer_idx}')(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f'bn_{layer_idx}')(x)
    
    # Configure optimizer
    if clip_norm is not None:
        optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)
    else:
        optimizer = tf.keras.optimizers.Adam()

    # Determine if mixed types (multiple outputs or multiple different types)
    unique_types = list(set(output_types))
    is_mixed = len(unique_types) > 1 and len(output_types) > 1

    if is_mixed:
        # Mixed types - create separate output heads with appropriate losses
        outputs = []
        losses = {}
        metrics = {}

        # Get output dimensions from data
        if isinstance(output_data, list):
            output_dims = [arr.shape[1] for arr in output_data]
        else:
            # Single concatenated output - need to split by output_shapes
            output_shapes = options.get('output_shapes', [(output_data.shape[1],)])
            output_dims = [int(np.prod(shape)) for shape in output_shapes]

        for i, (output_type, output_dim) in enumerate(zip(output_types, output_dims)):
            if output_type == 'classification':
                # Classification output head
                out = tf.keras.layers.Dense(output_dim, activation='softmax', name=f'output_{i}')(x)
                outputs.append(out)
                losses[f'output_{i}'] = 'categorical_crossentropy'
                metrics[f'output_{i}'] = ['accuracy']
            else:
                # Regression output head
                out = tf.keras.layers.Dense(output_dim, name=f'output_{i}')(x)
                outputs.append(out)
                losses[f'output_{i}'] = 'mse'
                metrics[f'output_{i}'] = ['mae']

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    else:
        # Single output type - use original logic
        overall_type = unique_types[0]

        if isinstance(output_data, list):
            output_dim = sum(arr.shape[1] for arr in output_data)
        else:
            output_dim = output_data.shape[1]

        if overall_type == 'classification':
            outputs = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            outputs = tf.keras.layers.Dense(output_dim, name='output')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def _save_model_metadata(model_file, model, input_scalers, output_scalers, 
                        input_shapes, output_shapes, options, history=None):
    """Save model weights and metadata to HDF5 file."""
    # Save model weights
    weights_file = f"{model_file}.weights.h5"
    model.save_weights(weights_file)
    
    # Save metadata
    metadata_file = f"{model_file}_metadata.h5"
    with h5py.File(metadata_file, 'w') as f:
        # Save scalers using pickle
        f.create_dataset('input_scalers', data=np.void(pickle.dumps(input_scalers)))
        f.create_dataset('output_scalers', data=np.void(pickle.dumps(output_scalers)))
        
        # Save shapes (serialize as pickled data)
        f.create_dataset('input_shapes', data=np.void(pickle.dumps(input_shapes)))
        f.create_dataset('output_shapes', data=np.void(pickle.dumps(output_shapes)))
        
        # Save options
        f.create_dataset('options', data=np.void(pickle.dumps(options)))
        
        # Save model architecture
        model_config = model.to_json()
        f.create_dataset('model_config', data=model_config.encode('utf-8'))
        
        # Save history if available
        if history:
            for key, values in history.items():
                f.create_dataset(f'history_{key}', data=values)
    
    return weights_file, metadata_file


def _load_model_metadata(model_file):
    """Load model weights and metadata from HDF5 files."""
    _ensure_ml_deps()
    weights_file = f"{model_file}.weights.h5"
    metadata_file = f"{model_file}_metadata.h5"
    
    # Load metadata
    with h5py.File(metadata_file, 'r') as f:
        input_scalers = pickle.loads(f['input_scalers'][()])
        output_scalers = pickle.loads(f['output_scalers'][()])
        input_shapes = pickle.loads(f['input_shapes'][()])
        output_shapes = pickle.loads(f['output_shapes'][()])
        options = pickle.loads(f['options'][()])
        model_config = f['model_config'][()].decode('utf-8')
        
        # Load history if available
        history = {}
        for key in f.keys():
            if key.startswith('history_'):
                hist_key = key.replace('history_', '')
                history[hist_key] = f[key][()]
    
    # Reconstruct model
    model = tf.keras.models.model_from_json(model_config)
    model.load_weights(weights_file)
    
    return model, input_scalers, output_scalers, input_shapes, output_shapes, options, history


def create_model(D_in, D_out, model_name='model', hidden_layers=4, units=128, 
                 activation='relu', dropout_rate=0.0, use_batch_norm=False, 
                 i_layer_entrance=None, output_type='auto', clip_norm=None, verbose=1):
    """
    Create and configure a neural network model.
    
    Args:
        D_in: List of numpy arrays (input parameters)
        D_out: List of numpy arrays (output parameters)
        model_name: Name for the model (default: 'model')
        hidden_layers: Number of hidden layers (default: 3)
        units: Units per hidden layer (default: 64)
        activation: Activation function (default: 'relu')
        dropout_rate: Dropout rate (default: 0.0)
        use_batch_norm: Use batch normalization (default: False)
        i_layer_entrance: List specifying the dense layer (0-indexed) where each input joins the network.
                          Defaults to all zeros, meaning every input is concatenated before the first layer.
        output_type: 'regression', 'classification', 'auto', or list of types for each output (default: 'auto')
        clip_norm: Gradient clipping norm (default: None, disabled)
        verbose: Verbosity level (default: 1)
    
    Returns:
        str: Model file path (without extension)
    """
    _ensure_ml_deps()

    if verbose:
        print(f"Creating model '{model_name}'...")
    
    if i_layer_entrance is not None and len(i_layer_entrance) != len(D_in):
        raise ValueError(
            f"Length of i_layer_entrance ({len(i_layer_entrance)}) must match number of inputs ({len(D_in)})"
        )
    
    # Always convert output_type to a list
    if isinstance(output_type, (list, tuple)):
        output_types = list(output_type)
        if len(output_types) != len(D_out):
            raise ValueError(f"Length of output_type list ({len(output_types)}) must match number of outputs ({len(D_out)})")
    else:
        # Single value - apply to all outputs
        output_types = [output_type] * len(D_out)

    # Auto-detect types where needed
    detected_types = []
    for i, otype in enumerate(output_types):
        if otype == 'auto':
            arr = D_out[i]
            y_raw = arr.reshape(arr.shape[0], -1)
            unique_vals = np.unique(y_raw)

            # Check for one-hot encoded classification data
            if (np.array_equal(unique_vals, [0, 1]) and
                y_raw.shape[1] > 1 and
                np.allclose(np.sum(y_raw, axis=1), 1.0)):
                detected_type = 'classification'
                if verbose:
                    print(f"Output {i}: Auto-detected one-hot classification ({y_raw.shape[1]} classes)")
            # Check for regular integer classification labels
            elif (len(unique_vals) <= 10 and
                  np.all(unique_vals >= 0) and
                  np.all(unique_vals == np.round(unique_vals))):
                detected_type = 'classification'
                if verbose:
                    print(f"Output {i}: Auto-detected classification ({len(unique_vals)} classes: {list(unique_vals)})")
            else:
                detected_type = 'regression'
                if verbose:
                    print(f"Output {i}: Auto-detected regression (range: {y_raw.min():.3f} to {y_raw.max():.3f})")

            detected_types.append(detected_type)
        else:
            detected_types.append(otype)
            if verbose:
                print(f"Output {i}: Using specified type '{otype}'")

    # Prepare data with detected types
    X_list, y, input_scalers, output_scalers, input_shapes, output_shapes = _prepare_data(D_in, D_out, detected_types)

    if verbose:
        print(f"Output types: {detected_types}")
        if isinstance(y, list):
            print(f"Input shapes: {[X.shape for X in X_list]}, Output shapes: {[arr.shape for arr in y]}")
        else:
            print(f"Input shapes: {[X.shape for X in X_list]}, Output shape: {y.shape}")
        if i_layer_entrance is not None:
            print(f"Input entrance layers: {i_layer_entrance}")
    
    # Create options dictionary for internal functions
    options = {
        'hidden_layers': hidden_layers,
        'units': units,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'use_batch_norm': use_batch_norm,
        'output_types': detected_types,  # Store as list
        'output_shapes': output_shapes,  # Store output shapes
        'clip_norm': clip_norm,
        'i_layer_entrance': i_layer_entrance
    }

    # Build model
    input_dims = [X.shape[1] for X in X_list]
    model = _build_network(input_dims, y, options)

    if verbose:
        print(f"Model compiled with:")
        print(f"  Loss function: {model.loss}")
        print(f"  Optimizer: {type(model.optimizer).__name__}")
        print(f"  Metrics: {[m.name if hasattr(m, 'name') else str(m) for m in model.metrics]}")
        print(f"  Output activation: {model.layers[-1].activation.__name__}")
        model.summary()

    # Save model and metadata
    model_file = model_name
    _save_model_metadata(model_file, model, input_scalers, output_scalers,
                        input_shapes, output_shapes, options)

    if verbose:
        print(f"Model '{model_name}' created and saved successfully!")
    
    return model_file


def train_model(D_in, D_out, model_file, epochs=400, batch_size=1025, 
                validation_split=0.1, patience=20, clip_norm=None, lr_factor=1.0,
                verbose=1, use_tensorboard=True):
    """
    Train a previously created model.
    
    Args:
        D_in: List of numpy arrays (input parameters)
        D_out: List of numpy arrays (output parameters)
        model_file: Path to model file (without extension)
        epochs: Number of training epochs (default: 1000)
        batch_size: Training batch size (default: 256)
        validation_split: Fraction for validation (default: 0.1)
        patience: Early stopping patience (default: 50)
        clip_norm: Gradient clipping norm (default: None, uses model's existing setting)
        lr_factor: Multiplicative factor applied when ReduceLROnPlateau triggers.
                   Values <1.0 enable learning-rate reduction; 1.0 disables it (default: 1.0)
        verbose: Verbosity level (default: 1)
        use_tensorboard: Enable TensorBoard (default: True)
    
    Returns:
        dict: Training history
    """
    _ensure_ml_deps()

    if verbose:
        print(f"Training model '{model_file}'...")
    
    # Load model
    model, input_scalers, output_scalers, input_shapes, output_shapes, model_options, _ = _load_model_metadata(model_file)
    
    # Update optimizer with gradient clipping if specified
    if clip_norm is not None:
        # Skip gradient clipping recompilation to avoid metric issues
        # User should specify clip_norm during model creation instead
        if verbose:
            print(f"Warning: Gradient clipping not applied during training. Use clip_norm during model creation instead.")

        # Update model options to save the new clip_norm setting
        model_options['clip_norm'] = clip_norm
    
    # Persist learning-rate reduction choice for downstream metadata queries
    model_options['lr_factor'] = lr_factor
    
    # Get output types from saved model options
    output_types = model_options.get('output_types', ['regression'])

    # Prepare data with the same types as when model was created
    X_list, y, _, _, _, _ = _prepare_data(D_in, D_out, output_types)

    # Split data
    n_samples = X_list[0].shape[0]
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(indices, test_size=validation_split, random_state=42)

    X_train_list = [X[train_indices] for X in X_list]
    X_val_list = [X[val_indices] for X in X_list]

    # Handle data splitting for mixed types
    if isinstance(y, list):
        # Mixed types - split each output separately
        y_train = [arr[train_indices] for arr in y]
        y_val = [arr[val_indices] for arr in y]
    else:
        # Single type - split normally
        y_train = y[train_indices]
        y_val = y[val_indices]
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    ]

    if lr_factor < 1.0:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=max(1, patience // 2),
            min_lr=1e-7
        )
        callbacks.append(reduce_lr)
    
    # Add TensorBoard callback
    if use_tensorboard:
        import datetime
        log_dir = f"tensorboard_logs/training/{model_file}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        if verbose:
            print(f"TensorBoard logs: {log_dir}")
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train_list, y_train,
        validation_data=(X_val_list, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
    
    # Save updated model with history
    _save_model_metadata(model_file, model, input_scalers, output_scalers, 
                        input_shapes, output_shapes, model_options, history.history)
    
    return history.history


def hyperparameter_search(D_in, D_out, model_name='tuned_model', max_trials=10,
                         epochs=100, batch_size=256, validation_split=0.1, patience=20,
                         output_type='auto', i_layer_entrance=None, verbose=1, overwrite=False,
                         use_tensorboard=True, search_method='random', search_clip_norm=False):
    """
    Find optimal hyperparameters using Keras Tuner.
    
    Args:
        D_in: List of numpy arrays (input parameters)
        D_out: List of numpy arrays (output parameters)
        model_name: Base name for models (default: 'tuned_model')
        max_trials: Maximum trials (default: 10)
        epochs: Epochs per trial (default: 100)
        batch_size: Batch size (default: 256)
        validation_split: Validation fraction (default: 0.1)
        patience: Early stopping patience (default: 20)
        output_type: 'regression', 'classification', 'auto', or list of types for each output (default: 'auto')
        i_layer_entrance: Optional list specifying the dense layer (0-indexed) where each input joins the network.
        verbose: Verbosity level (default: 1)
        overwrite: Remove existing tuning cache before starting (default: False)
        use_tensorboard: Enable TensorBoard logging for each trial (default: True)
        search_method: Search algorithm - 'random' or 'bayesian' (default: 'random')
        search_clip_norm: Include gradient clipping in hyperparameter search (default: False)
    
    Returns:
        dict: Comprehensive results containing:
            - best_model_file: Path to best model
            - best_hyperparameters: Best hyperparameters found
            - best_score: Best validation score achieved
            - best_duration: Training time for best model (seconds)
            - all_trials: List of all completed trials with details
            - total_trials: Number of completed trials
    """
    _ensure_ml_deps()

    if verbose:
        print(f"Hyperparameter search for '{model_name}'...")
    
    # Clean up existing tuning cache if overwrite is True
    if overwrite:
        import shutil
        tuning_dir = 'mlmapping_tuning'
        if os.path.exists(tuning_dir):
            if verbose:
                print(f"Removing existing tuning cache: {tuning_dir}")
            shutil.rmtree(tuning_dir)
    
    # Always convert output_type to a list
    if isinstance(output_type, (list, tuple)):
        output_types = list(output_type)
        if len(output_types) != len(D_out):
            raise ValueError(f"Length of output_type list ({len(output_types)}) must match number of outputs ({len(D_out)})")
    else:
        # Single value - apply to all outputs
        output_types = [output_type] * len(D_out)

    # Auto-detect types where needed
    detected_types = []
    for i, otype in enumerate(output_types):
        if otype == 'auto':
            arr = D_out[i]
            y_raw = arr.reshape(arr.shape[0], -1)
            unique_vals = np.unique(y_raw)

            # Check for one-hot encoded classification data
            if (np.array_equal(unique_vals, [0, 1]) and
                y_raw.shape[1] > 1 and
                np.allclose(np.sum(y_raw, axis=1), 1.0)):
                detected_type = 'classification'
                if verbose:
                    print(f"Output {i}: Auto-detected one-hot classification ({y_raw.shape[1]} classes)")
            # Check for regular integer classification labels
            elif (len(unique_vals) <= 10 and
                  np.all(unique_vals >= 0) and
                  np.all(unique_vals == np.round(unique_vals))):
                detected_type = 'classification'
                if verbose:
                    print(f"Output {i}: Auto-detected classification ({len(unique_vals)} classes: {list(unique_vals)})")
            else:
                detected_type = 'regression'
                if verbose:
                    print(f"Output {i}: Auto-detected regression (range: {y_raw.min():.3f} to {y_raw.max():.3f})")

            detected_types.append(detected_type)
        else:
            detected_types.append(otype)
            if verbose:
                print(f"Output {i}: Using specified type '{otype}'")

    # Prepare data with detected types
    X_list, y, input_scalers, output_scalers, input_shapes, output_shapes = _prepare_data(D_in, D_out, detected_types)
    
    def build_model(hp):
        search_options = {
            'hidden_layers': hp.Int('hidden_layers', 2, 5),
            'units': hp.Int('units', 32, 256, step=32),
            'activation': hp.Choice('activation', ['selu','relu', 'elu', 'gelu']),
            'dropout_rate': 0,
            'use_batch_norm': False,
            'output_types': detected_types,  # Use the detected types list
            'i_layer_entrance': i_layer_entrance
        }
        
        # Add gradient clipping to search if enabled
        if search_clip_norm:
            use_clipping = hp.Boolean('use_gradient_clipping')
            if use_clipping:
                search_options['clip_norm'] = hp.Choice('clip_norm_value', [0.5, 1.0, 2.0, 5.0])
            else:
                search_options['clip_norm'] = None
        else:
            search_options['clip_norm'] = None
        
        input_dims = [X.shape[1] for X in X_list]
        return _build_network(input_dims, y, search_options)
    
    # Split data
    n_samples = X_list[0].shape[0]
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(indices, test_size=validation_split, random_state=42)
    
    X_train_list = [X[train_indices] for X in X_list]
    X_val_list = [X[val_indices] for X in X_list]
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    # Setup callbacks
    search_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
    ]

    # Setup TensorBoard logging if requested
    if use_tensorboard:
        import datetime
        base_log_dir = f"tensorboard_logs/hyperparameter_tuning/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if verbose:
            print(f"TensorBoard logs for hyperparameter tuning: {base_log_dir}")
            print("To view: tensorboard --logdir tensorboard_logs")
        
        # Create TensorBoard callback for hyperparameter search
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=base_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        search_callbacks.append(tensorboard_callback)
    
    # Create tuner based on search method
    if search_method.lower() == 'bayesian':
        if verbose:
            print(f"Using Bayesian Optimization search with {max_trials} trials")
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_loss',
            max_trials=max_trials,
            directory='mlmapping_tuning',
            project_name=model_name
        )
    elif search_method.lower() == 'random':
        if verbose:
            print(f"Using Random Search with {max_trials} trials")
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=max_trials,
            directory='mlmapping_tuning',
            project_name=model_name
        )
    else:
        raise ValueError(f"Unknown search_method: {search_method}. Use 'random' or 'bayesian'.")
    
    # Search with or without TensorBoard
    tuner.search(X_train_list, y_train, epochs=epochs, batch_size=batch_size, 
                validation_data=(X_val_list, y_val), verbose=verbose, callbacks=search_callbacks)
    
    # Get all trial information
    all_trials = []
    for trial in tuner.oracle.trials.values():
        if trial.status == 'COMPLETED':
            # Handle timing information safely
            start_time = getattr(trial, 'start_time', None)
            end_time = getattr(trial, 'end_time', None)
            duration = None
            
            if start_time and end_time:
                try:
                    duration = (end_time - start_time).total_seconds()
                except:
                    duration = None
            
            trial_info = {
                'trial_id': trial.trial_id,
                'hyperparameters': dict(trial.hyperparameters.values),
                'score': trial.score,
                'metrics': dict(trial.metrics.metrics) if hasattr(trial.metrics, 'metrics') else {},
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }
            all_trials.append(trial_info)
    
    # Sort trials by score (best first)
    all_trials.sort(key=lambda x: x['score'])
    
    # Get best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = all_trials[0] if all_trials else None
    
    if verbose:
        print("Best hyperparameters:")
        for param, value in best_hps.values.items():
            print(f"  {param}: {value}")
        if best_trial and best_trial['duration']:
            print(f"  Training time: {best_trial['duration']:.1f}s")
    
    # Create descriptive name
    best_params = best_hps.values
    name_parts = [
        model_name,
        f"L{best_params['hidden_layers']}",
        f"U{best_params['units']}",
        f"{best_params['activation']}",
    ]
    
    if best_params.get('dropout_rate', 0) > 0:
        name_parts.append(f"drop{best_params['dropout_rate']:.1f}")
    
    if best_params.get('use_batch_norm', True):
        name_parts.append("BN")
        
    if best_params.get('clip_norm') is not None:
        name_parts.append(f"clip{best_params['clip_norm']}")

    if i_layer_entrance and any(layer_idx != 0 for layer_idx in i_layer_entrance):
        name_parts.append(f"entrance{''.join(str(idx) for idx in i_layer_entrance)}")
    
    best_model_file = "_".join(name_parts)
    
    # Save best model
    best_options = {
        'model_name': best_model_file,
        'output_types': detected_types,
        'i_layer_entrance': i_layer_entrance,
        **best_params
    }
    
    _save_model_metadata(best_model_file, best_model, input_scalers, output_scalers, 
                        input_shapes, output_shapes, best_options)
    
    if verbose:
        print(f"Best model saved as '{best_model_file}'")
        print(f"Total trials completed: {len(all_trials)}")
    
    # Return comprehensive results
    results = {
        'best_model_file': best_model_file,
        'best_hyperparameters': dict(best_hps.values),
        'best_score': best_trial['score'] if best_trial else None,
        'best_duration': best_trial['duration'] if best_trial else None,
        'all_trials': all_trials,
        'total_trials': len(all_trials)
    }
    
    return results


def load_model(model_file, device='auto'):
    """
    Load a saved model for CPU or GPU deployment.
    
    Args:
        model_file: Path to model file (without extension)
        device: 'cpu', 'gpu', or 'auto' (default: 'auto')
    
    Returns:
        str: Model file path for use with predict()
    """
    _ensure_ml_deps()
    if device == 'cpu':
        with tf.device('/CPU:0'):
            model, _, _, _, _, _, _ = _load_model_metadata(model_file)
    elif device == 'gpu':
        with tf.device('/GPU:0'):
            model, _, _, _, _, _, _ = _load_model_metadata(model_file)
    else:
        model, _, _, _, _, _, _ = _load_model_metadata(model_file)
    
    print(f"Model '{model_file}' loaded successfully on {device}")
    return model_file


def predict(D_in, model_file, batch_size=None):
    """
    Make predictions using a trained model.
    
    Args:
        D_in: List of numpy arrays (input parameters)
        model_file: Path to model file (without extension)
        batch_size: Batch size for prediction (default: None, uses model.predict default)
    
    Returns:
        list: Predictions in original D_out format
    """
    _ensure_ml_deps()
    # Load model and metadata
    model, input_scalers, output_scalers, input_shapes, output_shapes, _, _ = _load_model_metadata(model_file)
    
    # Prepare input data
    X_list = []
    for i, arr in enumerate(D_in):
        X_reshaped = arr.reshape(arr.shape[0], -1)
        X_scaled = input_scalers[i].transform(X_reshaped)
        X_list.append(X_scaled)
    
    # Make predictions
    if batch_size is not None:
        y_pred = model.predict(X_list, batch_size=batch_size)
    else:
        y_pred = model.predict(X_list)
    
    # Handle both single output (array) and multi-output (list) cases
    if isinstance(y_pred, list):
        # Multi-output model - y_pred is a list of arrays
        predictions = []
        for i, (pred_array, scaler) in enumerate(zip(y_pred, output_scalers)):
            if scaler is not None:
                # Regression output - apply inverse scaling
                pred_unscaled = scaler.inverse_transform(pred_array)
            else:
                # Classification output - no scaling
                pred_unscaled = pred_array.copy()

            # Reshape to original format
            shape = output_shapes[i]
            pred_reshaped = pred_unscaled.reshape(-1, *shape)
            predictions.append(pred_reshaped)

    else:
        # Single output model - y_pred is a single array
        y_pred_unscaled = y_pred.copy()
        start_idx = 0
        for i, scaler in enumerate(output_scalers):
            size = int(np.prod(output_shapes[i]))
            if scaler is not None:
                # Regression output - apply inverse scaling
                y_pred_unscaled[:, start_idx:start_idx+size] = scaler.inverse_transform(
                    y_pred[:, start_idx:start_idx+size]
                )
            # Classification outputs (scaler=None) remain unchanged
            start_idx += size

        # Reshape back to original format
        predictions = []
        start_idx = 0
        for shape in output_shapes:
            size = int(np.prod(shape))
            pred = y_pred_unscaled[:, start_idx:start_idx+size].reshape(-1, *shape)
            predictions.append(pred)
            start_idx += size
    
    return predictions


def predict_fast(D_in, model_file, batch_size=None):
    """
    Fast prediction using direct model call (bypasses some overhead).
    
    Args:
        D_in: List of numpy arrays (input parameters)
        model_file: Path to model file (without extension)
        batch_size: Batch size for prediction (default: None, processes all at once)
    
    Returns:
        list: Predictions in original D_out format
    """
    _ensure_ml_deps()
    # Load model and metadata
    model, input_scalers, output_scalers, _, output_shapes, _, _ = _load_model_metadata(model_file)
    
    # Prepare input data - more efficient without intermediate copying
    X_list = []
    for i, arr in enumerate(D_in):
        X_reshaped = arr.reshape(arr.shape[0], -1)
        X_scaled = input_scalers[i].transform(X_reshaped)
        X_list.append(X_scaled)
    
    # Direct model call (often faster than model.predict)
    if batch_size is not None and len(X_list[0]) > batch_size:
        # Process in batches for memory efficiency
        n_samples = len(X_list[0])
        y_pred_list = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = [X[start_idx:end_idx] for X in X_list]
            
            batch_pred = model(X_batch, training=False)
            if hasattr(batch_pred, 'numpy'):
                batch_pred = batch_pred.numpy()
            y_pred_list.append(batch_pred)
        
        y_pred = np.concatenate(y_pred_list, axis=0)
    else:
        # Process all at once
        y_pred = model(X_list, training=False)
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
    
    # Unscale outputs efficiently
    start_idx = 0
    for i, scaler in enumerate(output_scalers):
        size = int(np.prod(output_shapes[i]))
        y_pred[:, start_idx:start_idx+size] = scaler.inverse_transform(
            y_pred[:, start_idx:start_idx+size]
        )
        start_idx += size
    
    # Reshape back to original format
    predictions = []
    start_idx = 0
    for shape in output_shapes:
        size = int(np.prod(shape))
        pred = y_pred[:, start_idx:start_idx+size].reshape(-1, *shape)
        predictions.append(pred)
        start_idx += size
    
    return predictions


def load_model_once(model_file):
    """
    Load model and metadata once for repeated predictions.

    Returns a tuple (model, input_scalers, output_scalers, output_shapes)
    to be passed directly to predict_loaded().
    """
    _ensure_ml_deps()
    model, input_scalers, output_scalers, _, output_shapes, _, _ = _load_model_metadata(model_file)
    return model, input_scalers, output_scalers, output_shapes


def predict_loaded(D_in, model, input_scalers, output_scalers, output_shapes, batch_size=None):
    """
    Make predictions using a pre-loaded model (avoids reloading on every call).

    Args:
        D_in: List of numpy arrays (input parameters)
        model: Loaded Keras model (from load_model_once)
        input_scalers: Input scalers (from load_model_once)
        output_scalers: Output scalers (from load_model_once)
        output_shapes: Output shapes (from load_model_once)
        batch_size: Batch size (default: None, processes all at once)

    Returns:
        list: Predictions in original D_out format
    """
    _ensure_ml_deps()
    X_list = [input_scalers[i].transform(arr.reshape(arr.shape[0], -1))
              for i, arr in enumerate(D_in)]

    if batch_size is not None and len(X_list[0]) > batch_size:
        n_samples = len(X_list[0])
        chunks = [np.concatenate([model([X[s:min(s+batch_size, n_samples)] for X in X_list],
                                        training=False).numpy()
                                  if hasattr(model([X[s:min(s+batch_size, n_samples)] for X in X_list],
                                                   training=False), 'numpy')
                                  else model([X[s:min(s+batch_size, n_samples)] for X in X_list],
                                             training=False)
                                  ], axis=0)
                  for s in range(0, n_samples, batch_size)]
        y_pred = np.concatenate(chunks, axis=0)
    else:
        y_pred = model(X_list, training=False)
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()

    start_idx = 0
    for i, scaler in enumerate(output_scalers):
        size = int(np.prod(output_shapes[i]))
        if scaler is not None:
            y_pred[:, start_idx:start_idx+size] = scaler.inverse_transform(
                y_pred[:, start_idx:start_idx+size])
        start_idx += size

    predictions, start_idx = [], 0
    for shape in output_shapes:
        size = int(np.prod(shape))
        predictions.append(y_pred[:, start_idx:start_idx+size].reshape(-1, *shape))
        start_idx += size

    return predictions


def evaluate_model(D_in, D_out, model_file, verbose=1):
    """
    Evaluate a trained model on test data.
    
    Args:
        D_in: List of numpy arrays (input parameters)
        D_out: List of numpy arrays (target parameters)
        model_file: Path to model file (without extension)
        verbose: Verbosity level (default: 1)
    
    Returns:
        dict: Evaluation metrics
    """
    _ensure_ml_deps()
    # Load model
    model, input_scalers, output_scalers, _, _, _, _ = _load_model_metadata(model_file)
    
    # Prepare data
    X_list, y, _, _, _, _ = _prepare_data(D_in, D_out)
    
    # Evaluate
    results = model.evaluate(X_list, y, verbose=verbose)
    
    # Create results dictionary
    metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = results[i]
    
    return metrics


def get_model_info(model_file):
    """
    Get information about a saved model.
    
    Args:
        model_file: Path to model file (without extension)
    
    Returns:
        dict: Model information
    """
    _ensure_ml_deps()
    model, _, _, input_shapes, output_shapes, options, history = _load_model_metadata(model_file)
    
    info = {
        'model_file': model_file,
        'input_shapes': input_shapes,
        'output_shapes': output_shapes,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'options': options,
        'has_history': len(history) > 0
    }
    
    if len(history) > 0:
        loss_history = history.get('loss', [])
        val_loss_history = history.get('val_loss', [])
        info['final_loss'] = loss_history[-1] if len(loss_history) > 0 else None
        info['final_val_loss'] = val_loss_history[-1] if len(val_loss_history) > 0 else None
    
    return info
