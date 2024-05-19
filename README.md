# Deep Learning Group Task: Using the StandardSim dataset for change detection

This project relies heavily on [an unofficial StandarSim repo](https://github.com/mohashei/Standard-Sim/blob/0fbfc30a0244700230b4eac5708403f34bba28ad/standard-retail-dataset/README.md) by mokashei.

Here, we train a change detection model (DeepLabV3) on a smaller subset of StandardSim (500 scenes). The first model is trained using a ResNet50 backbone - similar to the authors. The second model introduces a novelty by using a ResNet101 backbone.

In the end, neither of the models perform in a satisfactory manner, suggesting that change detection is a complex task which requires a correspondingly large datset to achieve desired performance. Assuming there were no time constraints, further improvements could be made, such as:
* Training the model on the full dataset with early stopping.
* Testing the models trained on synthetic data on real world data, in order to evaluate to what extent performance would degrade.