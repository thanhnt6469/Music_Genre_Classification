# Music Genre Classification

A machine learning project for automatic music genre classification using deep learning techniques.

## Overview

This project implements a neural network-based classifier that can automatically identify the genre of a music file. The model is trained on a diverse dataset of music samples and can classify music into multiple genres including Rock, Jazz, Classical, Hip-Hop, and more.

## Features

- **Deep Learning Model**: Convolutional Neural Network (CNN) for audio feature extraction
- **Multiple Genres**: Support for various music genres
- **Audio Processing**: Mel-frequency cepstral coefficients (MFCC) feature extraction
- **Model Training**: Complete training pipeline with data augmentation
- **Testing**: Comprehensive testing with sample audio files

## Project Structure

```
Music_Genre_Classification/
├── Train_Music_Genre_Classifier.ipynb    # Training notebook
├── Test_Music_Genre.ipynb                # Testing notebook
├── Trained_model.h5                      # Trained model (HDF5 format)
├── Trained_model.keras                   # Trained model (Keras format)
├── training_hist.json                    # Training history
├── Test_Music/                           # Test audio files
│   ├── bensound-allthewayup-rock.mp3
│   ├── bensound-jazz-lefacteur.mp3
│   ├── classical-piano-9316.mp3
│   └── move-forward-hip-hop-165711.mp3
└── README.md                             # This file
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thanhnt6469/Music_Genre_Classification.git
cd Music_Genre_Classification
```

2. Install dependencies:
```bash
pip install tensorflow keras librosa numpy pandas matplotlib jupyter
```

## Usage

### Training the Model

1. Open `Train_Music_Genre_Classifier.ipynb` in Jupyter Notebook
2. Follow the notebook to train the model
3. The trained model will be saved as `Trained_model.h5` and `Trained_model.keras`

### Testing the Model

1. Open `Test_Music_Genre.ipynb` in Jupyter Notebook
2. Load the trained model
3. Test with audio files in the `Test_Music/` directory

## Model Architecture

The model uses a CNN architecture with:
- Input: MFCC features extracted from audio files
- Convolutional layers for feature extraction
- Dense layers for classification
- Dropout for regularization

## Results

The model achieves high accuracy in genre classification across multiple music genres. Detailed results and metrics can be found in the training notebook.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License. 