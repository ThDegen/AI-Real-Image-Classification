# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [x] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [x] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

75

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s256664, s252653, s250247, s243973

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

No, we did not use any open-source frameworks/packages besindes the ones mentioned during the course.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed dependencies using uv, a package manager, with pyproject.toml as the single source of truth for all project dependencies. Dependencies are organized into production requirements and development tools.

For a new team member to get an exact copy of our development environment, they would first need to install uv and clone the repository. Running uv sync in the project root automatically creates a virtual environment and installs all dependencies with exact versions from pyproject.toml. The uv.lock file ensures reproducible installations across all team members' environments.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We used this custom template provided by the course responsibles (https://github.com/SkafteNicki/mlops_template). We used the configs folder to place all our configuration files. We used the data file to keep labels and images. We placed our dockerfiles in the folder docker. When run locally, the models folder is used for model weights. The reports folder was used for this template and the canvas. We put all the code in the "src/ai_real_image_classification" folder. Lastly, in tests we placed our unit testes.
We removed a couple of folders which we didn't need, such as notebooks, as we didn't use jupyter notebooks throughout our project. 
We have added a dvc folder, which points to the remote storage. Additionally, we also added a virtual environment folder, where the setup of our virtual environment is explained. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implemented Ruff for both linting and formatting to ensure a consistent code style and catch common errors early. For typing, we utilized native Python type hints in key modules to improve code clarity and facilitate better IDE support. We added these rules to the pre-commit so that we have a proper code base throughout our project.
These concepts are critical in larger projects because they maintain codebase health and facilitate collaboration. Linting and formatting prevent style-related friction and keep the code readable for all team members regardless of their personal preferences. This enables us to catch potential bugs early and make complex data flows in ML pipelines easier to trace. Finally, clear documentation ensures that the system architecture and API usage are transparent, allowing the project to scale without losing vital knowledge or increasing technical debt.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented two unit tests for our dataset. Since using the full dataset for testing is not practical, we included two sample images under tests/train_data and their corresponding labels in tests/train.csv.
The tests in test_data.py verify, image preprocessing works correctly, dataset length matches the expected number of samples, the types of the image and label are correct (torch.Tensor and int), images have 3 channels and lastly transformations such as resizing are applied correctly.
Tests inside the test_model.py ensures that the model’s training, validation, and testing steps work as expected. It runs a single batch through all stages using synthetic data to verify that all model methods, logging, and optimizers are reachable. Assertions check both the output shape and type, which is essential for the project.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

[Image1](reports/figures/q8.jpeg)
The total code coverage of our project is 96%, calculated across all source and test files. Most modules have 100% coverage, while the model module has slightly lower coverage (90%) due to a few untested lines. Overall, this level of coverage gives us a high degree of confidence that the core functionality behaves as expected under normal conditions.
However, even if our code coverage were 100%, we would not consider the code to be completely error free. Code coverage covers only performance of model and dataset on dummy data, edge cases or real-world scenarios are not handled. Bugs related to numerical stability, unexpected input data, deployment environments, or performance issues may still exist despite full coverage. Therefore, high coverage is an important however it should be complemented with careful reviews.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

During the creation of our project, we used branches in order to try out local changes without affecting the workflow of other team members. When changes were made, we opened a pull request where team members controlled that there isn't anything affecting the performance of the main project. After torough testing and if no problem was found, we merged them back to the main branch. We also included automated unit tests to automatically check the integrity of the code.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did make use of DVC for version controlling our datasets. Specifically, we tracked our training data (79,950 images, ~4.6GB), test data, and CSV metadata files using files, while storing the actual data on Google Cloud Storage.

In the end, it helped us in controlling the data management part of our pipeline. First, it kept our Git repository small and fast by only committing lightweight reference files instead of gigabytes of images. Second, it ensured reproducibility we could always trace back which exact dataset version was used for any experiment, making it easy to verify results or debug issues.

Additionally, DVC made team collaboration straightforward. Instead of manually sharing large files, team members could simply run `dvc pull` to get the exact dataset version needed for any experiment. For our image classification project, where the quality and composition of training data directly affects model performance, having version control over our datasets was crucial for maintaining consistency across experiments and understanding how data changes impacted our results.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have set up our continuous integration (CI) using GitHub Actions and organized it into two separate workflow files, each targeting a specific aspect of code quality and reliability.

The first workflow, codecheck.yaml, focuses on static code quality checks. This workflow is triggered on every push and pull request to ensure that all incoming code adheres to our style and quality standards. It runs on Ubuntu and uses Ruff for both linting and formatting. By enforcing linting and formatting automatically in CI, we ensure consistent code style across the project and catch common issues early. The workflow uses uv to manage the Python environment and dependencies, and dependency caching is enabled via astral-sh/setup-uv, which significantly speeds up repeated runs.

The second workflow, test.yaml, is dedicated to running unit tests. This workflow is also triggered on pushes and pull requests, but it uses a matrix strategy to test across multiple operating systems and Python versions. Specifically, tests are executed on both Ubuntu and Windows, and across Python 3.11 and 3.12, increasing confidence that the code behaves correctly in different environments. Tests are run using pytest in verbose mode, allowing for clear visibility into test execution and failures.

In addition, we added a deployment job that runs only after all tests pass and only on pushes to the main branch. This job authenticates with Google Cloud Platform, sets up the Cloud SDK, and triggers Cloud Build using cloudbuild.yaml to automatically build and deploy the application to Cloud Run. Overall, this CI/CD setup ensures code quality, correctness, and reliable automated deployment.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra and weights and biases to manage all our settings. This allowed us to change and store all configurations. When we ran an experiment, we were able to compare performance based on different parameters such as learning rate, batch size, and more. Moreover, we saved weights and configurations together centrally.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure a full reproducibility and prevent data loss, we integrated Hydra for configuration management and Weights & Biases for real-time experiment tracking. Every run is governed by YAML files stored in a dedicated directory, which archive the exact hyperparameters used. To maintain environment consistency, we utilized pyproject.toml and uv.lock files, allowing anyone to recreate the identical software stack using the uv sync command. This dual approach enables us to make both the execution logic and the computational environment immutable. To reproduce any past experiment, we can simply retrieve the archived Hydra configuration and runs it within the synchronized environment.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

As seen in the first image, we have utilized Weights & Biases to track and compare multiple experimental runs to identify the most effective hyperparameters. The dashboard includes a media panel that logs validation predictions at specific steps. This allows for a asses the model's performance on individual samples with their predicted scores.

The charts in the first and third images track three critical metrics of Loss, Accuracy, and Test Performance.

Tracking training and validation Loss is essential for monitoring convergence. As shown in the validation loss graph, the loss decreases steadily. This indicates the model is effectively learning features without significant divergence. Comparing train_loss against val_loss helps us diagnose overfitting. Since both curves follow a similar downward trend without the validation loss spiking, we can confirm the model generalizes well to unseen data.

Accuracy is our primary success metric for this classification task. The validation accuracy chart shows the model reaches high performance of near 99%.
[Image1](reports/figures/q14_1.jpeg)
[Image2](reports/figures/q14_2.jpeg)
[Image3](reports/figures/q14_3.jpeg)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker played a central role in our project by enabling consistent, reproducible environments for model training across local development and cloud execution. We created a dedicated Dockerfile (located in docker/trainer.dockerfile) that installed PyTorch, our project dependencies, and copied the training script along with necessary utilities. This ensured the same runtime behavior whether running locally or in the cloud.
Initially, we built and tested the image locally using Docker Desktop with the command:
`docker build -f docker/trainer.dockerfile -t ai-real-image-classification-trainer.`
This workflow allowed rapid iteration and debugging before moving to the cloud. Later, to scale experiments on Vertex AI, we shifted to cloud-native builds. We used Cloud Build to construct and push the same image to Artifact Registry with:
`gcloud builds submit . --config=cloudbuild-trainer.yaml`
The resulting container was then referenced in Vertex AI custom training jobs.
link to docker file: [trainer.dockerfile](docker/trainer.dockerfile)

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging strategies were handled individually by each team member based on their specific components. As our code was kept in a modular and simple manner, the local debugging tools integrated within VS Code, such as interactive breakpoints and variable monitoring, were sufficient for identifying and fixing logical errors. We did not conduct extensive profiling, as the straightforward execution flow and minimal overhead ensured the code was already highly efficient for our experimental needs. Additionally, we extensively used the error logs which a generated once a bug is incurred. Lastly, we used Claude Agent for debugging. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Cloud Storage: is used for storing and version controlling our datasets through DVC. Our training images (~4.6GB, 79,950 files) and test data are stored in the GCS bucket, allowing team members to pull specific dataset versions without bloating the Git repository.

Compute Engine: is used for running our model training on a virtual machine. We deployed a VM instance (name: ric-instance, type: e2-medium) in the europe-west1-b zone. This VM provides the computational resources needed to train our image classification model in the cloud, eliminating the need for local GPU hardware and allowing us to run long training jobs remotely.

Cloud Build: We utilize Google Cloud Build as our CI/CD tool to automate the creation of our container artifacts. As defined in our build configuration, the service executes a pipeline that builds the Docker image from our source code directly on Google's infrastructure and pushes the tagged image to our Artifact Registry. This automation ensures that our training and application images are consistently built and immediately available for deployment.

For VertexAI: to execute and manage our machine learning model training in a scalable, serverless environment. This service ran our custom code from a Docker container. The process was automated via Cloud Build, which compiled the container image and submitted the training job to Vertex AI as part of a CI/CD pipeline defined in a `cloudbuild-trainer.yaml` file.

Cloud Run: We use Cloud Run to deploy and host both the frontend and backend of our application. This serverless platform allows us to deploy our containerized services in a fully managed environment, ensuring that the web-facing components of our project are scalable and accessible without requiring manual infrastructure management.

Artifact Registry: We utilize Google Artifact Registry as the central repository for storing and managing our container images. This allows Cloud Run to securely pull the exact versioned images needed for deployment, ensuring consistency between our build environment and the production environment.

All our used services can be found here:
[Image1](reports/figures/q17.jpeg)

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the Compute Engine to run our model training and development tasks. We deployed a VM instance named `ric-instance` with an `e2-medium` machine type in the `europe-west1-b` zone. The e2-medium instance provides 2 vCPUs and 4GB of memory, which was sufficient for our training workload.

The VM runs with both internal IP (10.132.0.2) and external IP (35.240.35.18), allowing us to access it remotely for development and monitoring. We used this instance to execute our training scripts and experiments in a cloud environment, which eliminated dependencies on local hardware and provided a consistent development environment across our team.

The Compute Engine setup integrated with our other GCP services-the VM could directly access our datasets stored in Cloud Storage through DVC, pulling the necessary training and test data on demand. This approach allowed us to run longer training jobs without tying up local machines, and made it easy to scale our compute resources if needed by changing the machine type or deploying additional instances for parallel experiments.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Image1](reports/figures/Q19_1.png)
[Image2](reports/figures/Q19_2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Image1](reports/figures/Q20_1.png)
[Image2](reports/figures/Q20_2.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Image1](reports/figures/Q21.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

Our approach to model training utilized Google Cloud's Vertex AI platform through a CI/CD pipeline. We defined the training infrastructure in a `cloudbuild-trainer.yaml` file, which orchestrated building our pre-developed Docker container directly in the cloud. The workflow was initiated locally by submitting the build configuration with the command `gcloud builds submit . --config=cloudbuild-trainer.yaml.` Once the custom training job was created, we monitored its progress in real-time by streaming logs from the gcloud shell using the `gcloud ai custom-jobs stream-logs` command followed by the specific job path.

However, we encountered instability during execution. Several jobs failed outright, while others halted unexpectedly. These interruptions were most frequently attributed to timeout limitations, where the job duration exceeded the default or configured runtime allowance before completion. 

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We successfully developed a functional API for our model using FastAPI, which provided an interface for real-time inference. By using its built-in support for an user interface, we utilized the /docs endpoint to interactively test our functions and validate data schemas without writing additional client-side code. This approach allowed us to check that our model remained accessible and easily verifiable during development and final testing. 
We implemented this API twice, once for the ONNX implementation (new) and once

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We deployed our API on Google Cloud Run, a fully managed serverless platform that automatically handles scaling, networking and infrastructure. For deployment, we wrapped our ONNX model in a FastAPI application that handles image uploads, preprocessing, model inference and logging of predictions to Google Cloud Storage. The API also exposes Prometheus metrics for monitoring request counts, prediction latency, and error rates.

Cloud Run was chosen because it allows us to deploy containerized applications without managing servers. This is particularly useful for machine learning applications where request volumes may fluctuate. Users can invoke the service by sending a POST request with an image file to the /predict endpoint, which returns a JSON response with the predicted label and probability. Since we also deployed frontend API as well, users also can use UI to upload the pictures to analyse.

Overall, this setup ensures keeping the operational overhead minimal and enabling continuous monitoring and logging for future analysis or retraining.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We have implemented basic monitoring for our deployed model using Prometheus metrics and GCP logging. The API collects metrics such as total number of requests, prediction errors, prediction latency and uploaded image sizes. These metrics are showed via the /metrics endpoint for Prometheus scraping. Moreover we added each prediction, including the filename, probability, predicted label, and timestamp to Google Cloud Storage for historical analysis.

Monitoring helps the longevity of our application by allowing us to detect anomalies such as sudden spikes in errors or unusually long inference times. These may indicate model degradation or infrastructure issues. Tracking image sizes and request volumes also informs scaling decisions. Over time, these insights allow us to maintain model performance, ensure reliability and plan retraining.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

As the model was trained locally, this didn't cause any issues. Additionally, we made sure to use lightweight Python versions and use economically inexpensive cloud hardware.
Student 250247 using the central Google Cloud account for all infrastructure work consumed roughly kr. 45.08 in credits. The most expensive service was Container Registry Vulnerability Scanning at kr. 39.54, an automated security feature that became the main cost driver. Working in the cloud provided essential tools for our MLOps pipeline but underscored the critical need to monitor and understand the cost implications of every enabled service.

++@all report credits)

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

One of the struggles of the project was managing infrastructure and deployment complexity. A significant amount of time was spent on the Docker build process in the cloud, where each image build took approximately 20 minutes. This severely slowed our development cycle, as testing any change to the dependencies or application code required a lengthy wait. We also struggled through running model training on cloud. First we worked on Vertex AI, model slowly trained on CPU but we couldn't manage to use the advantages of GPU.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s250247 was responsible for filling in the MLOps canvas, managing version control using DVC, creating and managing the Google Cloud environment, and developing the Docker containers. 
Student s252653 was responsible for writing the scripts of model and dataset, editing model code suitable for lightning, creating unit tests, deploying backend and frontend, writing frontend, transforming api made by torch to the ONNX.
Student s256664 ++
Student s243973 ++
All team members contributed to the codebase, documentation, and participated in problem-solving discussions. 
We have used generative AI tools, specifically Claude and Gemini, to assist in clarifying technical concepts. Additionally, Claude Agent and Copilot was used for debugging. 


## Project Description

Name: AI vs. Authentic Image Classification
The goal of our project is to develop a classification system able to identify AI-generated images (synthetic) vs. authentic photography (real).

Dataset (https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images):
The dataset consists of authentic images from the Shutterstock platform, categorised into different groups. These categories include one-third of images featuring humans, with the remaining images featuring a balanced selection of other categories. Each authentic image was paired with an AI-generated image. The AI images were created using state-of-the-art generative models, enabling direct comparison between real and synthetic content. Labels for the training and test data are provided in train.csv and test.csv, Binary (0 = Real, 1 = AI-generated). The training set includes 79,950 images, while the test set includes 19,986.

Model:
To separate AI-generated images from real ones, we aim to evaluate the performance of the ResNet model family pretrained on ImageNet.

## Machine Learning Operations Canvas

[Read the canvas](reports/MLOps_Canvas_Group_75.pdf)

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
