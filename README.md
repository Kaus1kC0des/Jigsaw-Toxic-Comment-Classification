# Jigsaw Toxic Comment Classification

This project aims to build a BERT-based model for classifying comments from the Jigsaw dataset into various toxicity categories. The model will be trained to identify toxic comments and help in moderating online discussions.

## Project Structure

```
jigsaw-toxic-comment-classification
├── data
│   ├── raw                # Raw dataset files
│   └── processed          # Processed dataset files
├── logs                   # Log files for training and evaluation
├── models
│   └── saved_models       # Saved model artifacts
├── notebooks              # Jupyter notebooks for EDA and evaluation
├── src
│   ├── __init__.py       # Marks the src directory as a package
│   ├── config.py         # Configuration settings
│   ├── data.py           # Data loading and preprocessing
│   ├── model.py          # Model architecture
│   ├── trainer.py        # Training loop and evaluation
│   ├── utils.py          # Utility functions
│   └── main.py           # Entry point for the project
├── requirements.txt       # Python dependencies
├── setup.py               # Packaging information
└── README.md              # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd jigsaw-toxic-comment-classification
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place the raw dataset files in the `data/raw` directory. The preprocessing script will handle the cleaning and preparation of the data.

2. **Training the Model**: Run the `main.py` script to start the training process. This script will load the data, initialize the model, and begin training.

   ```bash
   python src/main.py
   ```

3. **Evaluation**: After training, the model artifacts will be saved in the `models/saved_models` directory. You can use the Jupyter notebooks in the `notebooks` directory for exploratory data analysis and model evaluation.

## Dataset

The dataset used for this project is the Jigsaw Toxic Comment Classification dataset, which contains comments from various online platforms labeled for different types of toxicity.

## Logging

Training logs will be stored in the `logs` directory. These logs will include information about the training process, evaluation metrics, and any warnings or errors encountered.

## Contributing

Contributions to improve the model or the project structure are welcome. Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.