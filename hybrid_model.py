import numpy as np
import tensorflow as tf
import os

class HybridEnsemble:
    def __init__(self, model_paths):
        """
        Initialize the ensemble with a list of model paths.
        
        Args:
            model_paths (list): List of paths to .h5 model files.
        """
        self.model_paths = []
        self.model_names = []
        for path in model_paths:
            if os.path.exists(path):
                self.model_paths.append(path)
                self.model_names.append(os.path.basename(os.path.dirname(path)))
            else:
                print(f"Warning: Model path {path} does not exist.")
        
        if not self.model_paths:
            raise ValueError("No valid model paths provided.")
        print(f"Ensemble initialized with {len(self.model_paths)} models: {self.model_names}")

    def predict(self, data_generator, weights=None):
        """
        Make predictions using the ensemble.
        
        Args:
            data_generator: Keras ImageDataGenerator or similar.
            weights (list, optional): List of weights for each model. 
                                      If None, uses equal weighting (soft voting).
        
        Returns:
            np.array: Weighted average of predictions.
        """
        if weights is None:
            weights = [1.0 / len(self.model_paths)] * len(self.model_paths)
        
        if len(weights) != len(self.model_paths):
            raise ValueError("Number of weights must match number of models.")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        print("Generating predictions from ensemble (Sequential Loading)...")
        all_preds = []
        
        for i, path in enumerate(self.model_paths):
            print(f"Processing model {i+1}/{len(self.model_paths)}: {self.model_names[i]}...")
            
            # Load model
            try:
                model = tf.keras.models.load_model(path)
                
                # Handle Input Resizing
                if hasattr(data_generator, 'target_size'):
                    gen_shape = data_generator.target_size
                    if model.input_shape and len(model.input_shape) == 4:
                        model_shape = model.input_shape[1:3]
                        if gen_shape != model_shape and model_shape != (None, None):
                            print(f"Resizing input from {gen_shape} to {model_shape} for {self.model_names[i]}")
                            inp = tf.keras.Input(shape=gen_shape + (3,))
                            resized = tf.keras.layers.Resizing(model_shape[0], model_shape[1])(inp)
                            out = model(resized)
                            model = tf.keras.Model(inputs=inp, outputs=out)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # If a model fails, we might need to handle it. 
                # For now, append zeros or skip? 
                # Skipping messes up weights. Let's raise for now.
                raise e

            # Predict
            data_generator.reset() 
            preds = model.predict(data_generator, verbose=1)
            all_preds.append(preds)
            
            # Cleanup
            del model
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
        all_preds = np.array(all_preds)
        
        # Weighted average
        weighted_preds = np.average(all_preds, axis=0, weights=weights)
        
        return weighted_preds

    def get_all_predictions(self, data_generator):
        """
        Get predictions from all models individually.
        Returns: np.array of shape (num_models, num_samples, num_classes)
        """
        print("Generating predictions from all models (Sequential Loading)...")
        all_preds = []
        
        for i, path in enumerate(self.model_paths):
            print(f"Processing model {i+1}/{len(self.model_paths)}: {self.model_names[i]}...")
            try:
                model = tf.keras.models.load_model(path)

                # Handle Input Resizing
                if hasattr(data_generator, 'target_size'):
                    gen_shape = data_generator.target_size
                    if model.input_shape and len(model.input_shape) == 4:
                        model_shape = model.input_shape[1:3]
                        if gen_shape != model_shape and model_shape != (None, None):
                            print(f"Resizing input from {gen_shape} to {model_shape} for {self.model_names[i]}")
                            inp = tf.keras.Input(shape=gen_shape + (3,))
                            resized = tf.keras.layers.Resizing(model_shape[0], model_shape[1])(inp)
                            out = model(resized)
                            model = tf.keras.Model(inputs=inp, outputs=out)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                raise e

            data_generator.reset() 
            preds = model.predict(data_generator, verbose=1)
            all_preds.append(preds)
            
            del model
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
        return np.array(all_preds)

        