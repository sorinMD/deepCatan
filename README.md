# deepCatan

Code and data for paper: Mihai Dobre and Alex Lascarides, Combining a Mixture of Experts with Transfer Learning in Complex Games, Proceedings of the AAAI Spring Symposium: Learning from Observation of Humans, Stanford, USA.

This is an Eclipse Maven project. Before cloning install eclipse, the m2e plug-in http://www.eclipse.org/m2e/ and the git plug-in http://www.eclipse.org/egit/.

To run on GPU, replace "<nd4j.backend>nd4j-native</nd4j.backend>" line in pom.xml with the corresponding cuda backend, e.g.: "<nd4j.backend>nd4j-cuda-7.5</nd4j.backend>".

After building the project, run DeepCatan-0.0.1-SNAPSHOT-bin.jar and specify one of the test classes contained in the test package. These classes can be used to reproduce the experiments given in the paper as following:
- test.catan.baseline.FrequencyBased is the baseline model that counts the number of times an action was performed in the training dataset and selects the action with the highest count during evaluation.
- test.catan.mlp.CatanMlpTest trains one expert and evaluates it without crossvalidation.
- test.catan.mlp.CVCatanMlpTest trains one expert and evaluates it with crossvalidation.
- test.catan.mlp.MlpFullDataSetTest trains and evaluates one model on all 6 tasks without crossvalidation.
- test.catan.mlp.CVMlpFullDataSetTest trains and evaluates one model on all 6 tasks with crossvalidation.
- test.transfer.CVMlpTransferTest pre-trains the model on all but the target task, before fine-tunning it on the target task.

Configuration is set in the training_config.txt file (the values given in the project are the default ones used during the experiments):
- Path=(path to parent data folder)
- LearningRate=(learning rate)
- MiniBatchSize=(size of minibatch)
- Epochs=(number of epochs)
- LabelWeight=(weight given to the one hot labels when softening the target labels)
- MetricWeight=(weight given to the similarity metric when softening the target labels)
- DataType=human/synth(what data to use; it adds the name to the path)
- PreTrainTask=0-5(task to be used for pre-training if doing 1to1 transfer learning)
- Task=0-5(choose target task)
- Samples=0-50000 (number of samples to use for training)
- Normalisation=true/false (0 mean and unit variance normalisation)
- PreTrainEpochs=(number of epochs to pre-train when doing transfer learning) 

Note: Extract the data archive that can be found in data folder and provide the path to the extracted data before running any of the experiments. The data is shuffled and grouped based on tasks. Normalisation parameters (means and standard deviation) are also already computed and stored in the corresponding folders. The format of the data is given in utils.NumericalFeatureVectorOffsets.java. The code that used for generating/collecting the data will be released with the code for the STAC project (https://www.irit.fr/STAC/). 

Python script plot.py can be used to plot and vizualise the results.