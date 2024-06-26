# Facebook Marketplace Recommendation Ranking System

## Description:
Key technologies used: Python, Pandas, PIL

Facebook Marketplace is a platform for buying and selling products on Facebook.

This is an implementation of the system behind the marketplace, which uses machine learning to recommend the most relevant listings based on a personalised search query.

## File Structure:
- **data_cleaning.py**: Python script containing the DataCleaning class
- **sandbox.ipynb**: Jupyter notebook that will highlight the proecess of extracting, cleaning and processing data
- **clean_image_data.py**: Python script that finds the average size of images in a folder and resizes them all to the same size in a new folder
- **resnet50_model.py**: Python script that adjusts a pre-trained image recognition model for this particular case, and finetunes it for the image dataset
- **encoder.pkl**: Pickle file containing the label encoder dictionary
- **training_data.csv**: CSV file containing image id's and their corresponding encoded labels

## Python Classes:
- **DataCleaning**: A class for cleaning extracted data
