# Sentiment Analysis

This project uses a pre-trained BERT model for sentiment analysis. The model is trained to classify text into positive or negative sentiments.

## Project Structure

The project is organized as follows:
- **data/**: Contains data files used for training and prediction.
- **src/**: Contains the main code for model training and prediction.
- **notebooks/**: Jupyter Notebook files for experimentation.
- **output/**: Stores the final prediction results.
- **README.md**: Documentation file with project details.
- **requirements.txt**: Lists project dependencies.


## Download Trained Model and Checkpoints

The trained model and checkpoint files are too large to be stored directly in this repository. You can download them from the following links:

- [Download Best Trained Model (model.pth)](https://your-google-drive-link-for-model.com)
- [Download Checkpoints (ckpts/ folder)](https://your-google-drive-link-for-ckpts.com)

**Instructions**:
1. Download the model and checkpoint files from the links above.
2. Place the `model.pth` file in the `models/` directory.
3. Place the `ckpts/` folder with all checkpoint files in the root directory of the project, so it matches the structure above.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
