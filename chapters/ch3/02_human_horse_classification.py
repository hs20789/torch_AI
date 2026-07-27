# %%
import urllib.request
import zipfile

# %%
url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(url, file_name)

with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(validation_dir)