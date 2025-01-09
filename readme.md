# WeightsLab UI
This is WeightsLab, an UI designed to allow AI practitioners to have better
control over the training process of Deep Neural Networks training.

This is an early prototype of a solution that aims to DEBUG and FIX potential
problems that occurs during training:
* overfitting
* plauteous
* minority class misses
* problematic data samples analysis
* data set slicing
* weights manipulation (freezing, reinitialisation)

Since the paradigm is about granular statistics and interactivity, this allows
for very useful and interesting flows to be performed:
* model minimization
* data set curation
* root cause analysis
* reduction in non-determinism


## Steps needed to get started
- [ ] Download the framework repo:
```git clone https://github.com/GrayboxTech/weightslab_ui.git``
- [ ] Install the framework:
```pip instal -e .```
- [ ] Download the UI repo(this repo):
```git@github.com:GrayboxTech/weightslab_ui.git```
- [ ] Compile rpc messages:
```python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. experiment_service.proto```
- [ ] Start the trainer process:
```python trainer_worker.py```
- [ ] Launch UI monitoring process:
```python weights_lab.py --root_directory=PATH_TO_ROOT_DIRECTORY_OF_EXPERIMENT```
- [ ] Open the link from terminal:
``` Dash is running on http://127.0.0.1:8050/ ```


### Initial page
![Screenshot of Image 1](screen-shots/hyper_and_plots.png)

### Short Demos
![Short Demo 1](screen-shots/reinits.gif)

![Short Demo 2](screen-shots/data-model-manipulation.gif)


