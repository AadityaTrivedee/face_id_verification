Facial Recognition with Siamese Networks
========================================

This repository demonstrates the creation and deployment of a facial recognition system using TensorFlow and Keras. The system leverages a Siamese network for one-shot learning, enabling identity verification using image pairs. The code directly follows the guidelines given in the original research paper of [Siamese Neural Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Gkoch, Zemel, and Rsalakhu. 

Features
--------

-   **Data Preparation**: Set up directories for training and organize the dataset (anchor, positive, negative images).
-   **Model Architecture**: Create and train a Siamese network with custom distance metrics.
-   **Application Deployment**: Perform identity verification with preprocessed images.

* * * * *

Installation
------------

1.  Clone this repository:

    bash

    Copy code

    `git clone https://github.com/yourusername/face_id_verification.git`
    `cd face_id_verification`

2.  Install required dependencies:

    bash

    Copy code

    `pip install tensorflow keras opencv-python numpy matplotlib`

3.  Ensure GPU compatibility for TensorFlow (if applicable).

* * * * *

Dataset Preparation
-------------------

1.  **Download the LFW dataset**:

    -   Visit [LFW Dataset](https://vis-www.cs.umass.edu/lfw/) and download the `lfw.tgz` file, and extract it. (manually done/not available in code)
    -   Extract the contents and move all images into the `data/negative` folder.
2.  **Create Directory Structure**:

    -   Under the `data` folder, create the following directories:
        -   `anchor`
        -   `positive`
        -   `negative`
3.  **Capture Anchor and Positive Images**:

    -   Capture at least 300 images of the target identity for both `anchor` and `positive` folders. Ensure variations in lighting and angle for better results.

* * * * *

Training the Model
------------------

1.  Run the `main.ipynb` notebook:

    -   Prepares datasets and trains the Siamese network.
      
2.  Save the trained model:

    -   Save the model as `siamesenetv2.keras`.

* * * * *

Application Deployment
----------------------

1.  Create the following directory structure:

    -   `application_data`
        -   `input_images`
        -   `verification_images`
2.  Prepare Verification Data:

    -   Add one random image of the target to `input_images` (e.g., `input_image.jpg`). 
    -   Add 50 random images from the `positive` dataset to `verification_images`. (manually done/not available in code)
3.  Run the `irl_test.ipynb` notebook:

    -   Load the saved model and verify images.

* * * * *

How to Use
----------

1.  Load the trained model:

    python

    Copy code

    `siamese_network = tf.keras.models.load_model('siamesenetv2.keras', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})`

2.  Run the verification function:

    python

    Copy code

    `results, verified = verify(model=siamese_network, detection_treshold=0.5, verification_treshold=0.5)`
    `print(f"Verification result: {'Success' if verified else 'Failure'}")`

