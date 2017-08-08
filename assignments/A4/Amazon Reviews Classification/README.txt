Amazon Reviews Classification folder

- data folder:
	- source data (Shoes.txt and glove.6B) 
	- data preprocessing jupyter notebook (data_processing_api.ipynb)
	- python script (data_processing_api.py)
	- preprocessed data
		- raw processed data (shoes_list_of_review_dicts.pkl), contains cleaned text, labels, etc
		- data for embedding (data_and_embedding100.npz)

- models folder: each subfolder contains trained model and checkpoint files for the particular architecture. 	
	- final trained models (model.h5)
	- checkpoint files during training 

- experiments folder: contains jupyter notebooks during development, only for experimental purpose 

- notebooks folder: contains refactored and cleaned notebooks for building different classfiers on the Amazon Reviews (shoes) dataset, fully documented and tested.  

- images folder: contains reference images for the notebooks.

- src folder: contains executable python scripts for building and test different classifiers, training results are saved in the models folder and benchmarks are in the following. 


**************************************************************************************************************

Training results: 


Classification Accuracies: 

Baselines: 
	- Majority class predictor: 0.65809
	- Multinomial logistic regression: 0.6580
	- SVM with embeddings: 0.68233

Deep learning models:
	- Convolution (2 conv layers): 0.9393 - 0.9443
	- Simple RNN (with gradient clipping): 0.7074 - 0.7116
	- LSTM: 0.8884
	- GRU: 0.8684 - 0.8792
	- Bidirectional LSTM: 0.9061
	- Convolution + Bidirectional LSTM: 0.9478
	- Word level attention (on bidirectional GRU): 0.9323 - 0.9404


Training Time:

on a CPU machine (Intel i5-5200, 2.20GHz x 4),
	- All the baselines are trained in a decent time, less than 10 min
	- Convolution: 505 s/epoch
	- Simple RNN: 219 s/epoch
	- LSTM: 
	- GRU:
	- Bidirectional LSTM: 
	- Convolution + Bidirectional LSTM: 2205 s/epoch
	- Word Attention model: 1554 s/epoch


on a GPU machine (GeForce GTX 1080/PCIe/SSE2),
- All the baselines are trained in a short time, less than 10 min
	- Convolution: 153 s/epoch
	- Simple RNN: 69 s/epoch
	- LSTM: 314 s/epoch
	- GRU: similar to LSTM 
	- Bidirectional LSTM: 580 s/epoch
	- Convolution + Bidirectional LSTM: 675 s/epoch
	- Word Attention model: 2033.2 s/epoch (very suspicious, may have to run the experiment again)
