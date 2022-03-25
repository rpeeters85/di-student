* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts may require large amounts of RAM (64GB+) and GPUs. It is advised to use a powerful workstation or server when experimenting with some of the larger files.

    The code has only been used and tested on Linux (CentOS) servers.

* **Building the conda environment**

    To build the exact conda environment used for the experiments, navigate to the project root folder where the file *environment.yml* is located and run ```conda env create -f environment.yml```
    
    Furthermore you need to install the project as a package. To do this, activate the environment with ```conda activate deeper-0322```, navigate to the root folder of the project, and run ```pip install -e .```

* **Downloading the raw data files**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the files into the correct locations.
    You can find the data at *data/raw/*

* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. *src/processing/preprocess/preprocess_corpus.py*
    2. *src/processing/preprocess/preprocess_ts_gs.py*
    3. *src/processing/preprocess/preprocess_deepmatcher_datasets.py*
    4. *src/processing/contrastive/prepare_data.py*
	5. *src/processing/contrastive/prepare_data_deepmatcher.py*

* **Running the Contrastive Pre-training and Cross-entropy Fine-tuning**

    Navigate to *src/contrastive/*
    
	You can find respective scripts for running the experiments of the paper in the subfolders *lspc/* *abtbuy/* and *amazongoogle/*. Note that you need to adjust the file path in these scripts for your system (replace ```your_path``` with ```path/to/repo```).
	
	* **Contrastive Pre-training**
	
		To run contrastive pre-training for the abtbuy dataset for example use 

		```bash abtbuy/run_pretraining_clean.sh MODEL BATCH_SIZE LEARNING_RATE TEMPERATURE (AUG)```

		You need to specify model, batch size, learning rate and temperature as arguments here. Optionally you can also apply data augmentation by passing an augmentation method as last argument.

		For the WDC Computers data you need to also supply the size of the training set, e.g. 

		```bash lspc/run_pretraining_roberta.sh MODEL BATCH_SIZE LEARNING_RATE TEMPERATURE TRAIN_SIZE (AUG)```
	
	* **Cross-entropy Fine-tuning**
	
		Finally, to use the pre-trained models for fine-tuning, run any of the fine-tuning scripts in the respective folders, e.g. 

		```bash abtbuy/run_finetune_siamese_frozen.sh MODEL BATCH_SIZE LEARNING_RATE TEMPERATURE (AUG)``` 

		Please note, that BATCH_SIZE refers to the batch size used in pre-training. The fine-tuning batch size is locked to 64 but can be adjusted in the bash scripts if needed.

		Analogously for fine-tuning WDC computers, add the train size: 

		```bash lspc/run_finetune_siamese_frozen_roberta.sh MODEL BATCH_SIZE LEARNING_RATE TEMPERATURE TRAIN_SIZE (AUG)```

	
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
