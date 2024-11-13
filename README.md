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

## Download Trained Model and Checkpoints

The trained model and checkpoint files are too large to be stored directly in this repository. You can download them from the following links:

- [Download Pre-trained BERT Model (models--bert-base-chinese.zip)](https://drive.google.com/drive/folders/1Wl7MjmCa8AY4Ki5GicfPewtummWtqFoc?usp=share_link)
- [Download Checkpoints (model-0.pth.zip to model-3.pth.zip)](https://drive.google.com/drive/folders/1Wl7MjmCa8AY4Ki5GicfPewtummWtqFoc?usp=share_link)

**Instructions**:
1. Download the files from the links above.
2. Unzip `models--bert-base-chinese.zip` and place the contents in the `models/` directory.
3. Unzip `model-0.pth.zip`, `model-1.pth.zip`, `model-2.pth.zip`, and `model-3.pth.zip`, and place the extracted `.pth` files in the `ckpts/` folder. Place the `ckpts/` folder with all checkpoint files in the root directory of the project, so it matches the structure above.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
