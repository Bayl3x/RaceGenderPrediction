# -*- coding: utf-8 -*-
"""
@author: Alex Zhao
"""

import gender_race_classification as model

model_type = 'race'
sample_size_per_group = 200
kfold_split = 3
training_epochs = 10
num_of_batches = 10
early_stop_patience = 3
model.build_model(model_type, sample_size_per_group, kfold_split, training_epochs, num_of_batches, early_stop_patience)

model_type = 'gender'
sample_size_per_group = 200
kfold_split = 3
training_epochs = 10
num_of_batches = 10
early_stop_patience = 3
model.build_model(model_type, sample_size_per_group, kfold_split, training_epochs, num_of_batches, early_stop_patience)