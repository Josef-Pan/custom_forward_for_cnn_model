# Custom forward method for pretrained CNN model
Replace the most important method of a CNN model to save GPU memory usage
## Loading a CNN model and training data into GPU can easily consume GPU memory
- pytorch provides a mechanism called checkpoint
- We can use this to save GPU memory usage significantly
- Limitations: No dropout in checkpointed layers
## This example can demonstrate GPU usage with the new forward method of the CNN model
