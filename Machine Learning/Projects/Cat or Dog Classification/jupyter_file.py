{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2b93d32",
   "metadata": {},
   "outputs": [],
   "source": [

   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afd40cd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import and split directories of the dataset\n",
    "dic = 'cats_and_dogs.zip'\n",
    "with ZipFile(dic, 'r') as z:\n",
    "    z.extractall()\n",
    "PATH = 'cats_and_dogs'\n",
    "\n",
    "train_dir = os.path.join(PATH, 'train')\n",
    "val_dir = os.path.join(PATH, 'validation')\n",
    "test_dir = os.path.join(PATH, 'test')\n",
    "labels = ['Cat', 'Dog']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1cf8a36",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Create input function\n",
    "def input_format(x):\n",
    "    categories = ['cats', 'dogs'] \n",
    "    folder = os.path.join('cats_and_dogs', x)\n",
    "    data_set = []\n",
    "    \n",
    "    for ctg in categories:\n",
    "        path = folder\n",
    "        if x == 'train' or x == 'validation':\n",
    "            path = os.path.join(folder, ctg)\n",
    "            \n",
    "        for img in os.listdir(path):\n",
    "            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)\n",
    "            data_set.append(img_array)\n",
    "        \n",
    "    return data_set\n",
    "\n",
    "train_fn = input_format('train')\n",
    "test_fn = input_format('test')\n",
    "valid_fn = input_format('validation')\n",
    "print(train_fn)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c2415a0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f5275f0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
