Example code for 1. multinomial logistic regression and 2. skigram training algorithm, implemented with `Python` and `numpy` from scratch . 
These were used as educational examples in Brandeis university COSI114B course, *Fundamentals of Natural Language Processing II* in 2021 spring semester. 

Note that both code depend on `logres.py` module which is not uploaded here, because it was originally implemented as a solution code for a programming assignment of the course. 
The `logres.py` module should have 1) `load_data(dats_set, training=True)` method and 2) `featurize(document)` method, each does 1) construct a [design matrix](https://en.wikipedia.org/wiki/Design_matrix) of the `data_set` and build `class_dict` & `featrue_dict` with all token types when `training=True`, 2) convert a list of tokens into a list of integer encoding of tokens using the `feature_dict`, respectively. 

Data used to test algorithms are 
1. Binomial classification (`logres.py`): *movie reviews* dataset (https://www.kaggle.com/nltkdata/movie-review?select=movie_reviews)
1. Multinomial classification (`logres_multi.py`): *hotel reviews* dataset (https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)
1. Word vectors (`skipgram.py`): three books (*Emma*, *Sense and Sensibility*, and *Persuasion*) of Jane Austen, downloaded from the Gutenberg Project (https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)

Data for classification was trimmed for class balance, and split into `dev` and `train` set. 
