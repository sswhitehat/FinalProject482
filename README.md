<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
  <a href="https://github.com/github_username/repo_name">
    <img src="Images/logo.png" alt="Logo" width="240" height="240">
  </a>

<h1 align="center">StrikeMetrics</h1>

  <p align="center">
    StrikeMetrics is a prototype tool for combat sport analysis that utilizes elements of Computer Vision and Machine Learning. 
    <br />
    <a href="https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482">View Demo</a>
    ·
    <a href="https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- VIDEO AND GIF SECTION -->
## Videos and Demonstrations

Here are some videos and GIFs that demonstrate the project:

<div align="center">

### YouTube Videos
  
  [![Video_One](https://img.youtube.com/vi/PHk928ncIkg/0.jpg)](https://www.youtube.com/watch?v=0cK-5Odirgo)
  [![Video Two](http://img.youtube.com/vi/BtxTB55BXr4/0.jpg)](https://www.youtube.com/watch?v=0yUIstA3iCI)
  [![Video Three](http://img.youtube.com/vi/iCUb43KZ0d4/0.jpg)](https://www.youtube.com/watch?v=g-tiUbL8vpw)

### GIF Demonstrations
  
  ![GIF One](Images/gif_one.gif)
  ![GIF Two](Images/gif_two.gif)
  
</div>

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Introduction
Many sports, like football, swimming etc, utilize computer vision approaches to enhance the appeal of the sport and make life easier for referees and officials. However, in boxing, manual counting is required to calculate the total number of punches and kicks (for kickboxing) for each match. Our group wanted to see if there is a way to automate the process by using a computer vision approach. 

### Related Work
There is an existing program which already does something similiar: [Jabbr](https://jabbr.ai/), which is exclusively used for boxing.

Papers that guided our work:
Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).

Thomas, G., Gade, R., Moeslund, T. B., Carr, P., & Hilton, A. (2017). Computer vision for sports: Current applications and research topics. Computer Vision and Image Understanding, 159, 3-18

Sudhakaran, S., & Lanz, O. (2017, August). Learning to detect violent videos using convolutional long short-term memory. In 2017 14th IEEE international conference on advanced video and signal based surveillance (AVSS) (pp. 1-6). IEEE.

Shou, Z., Wang, D., & Chang, S. F. (2016). Temporal action localization in untrimmed videos via multi-stage cnns. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1049-1058)

Kasiri, S., Fookes, C., Sridharan, S., & Morgan, S. (2017). Fine-grained action recognition of boxing punches from depth imagery. Computer Vision and Image Understanding, 159, 143-153.

## Metric Definition

The metric we will be using is the number of punches correctly being detected. At first, we wanted our application to be able to distinguish different types of punches, like a jab vs a hook. However, since differentiating between them is hard for the application, we decided that our metric will just be the number of punches (any kind) and number of kick (any kind) that it accurately predicted.

$$
\text{accuracy} = \frac{\text{number of correct guesses}}{\text{total punches/kicks}}
$$

## Implementation

### Data collection

For our project, we needed to collect a few datasets which we would be training our model on. We used multiple videos from([Glory](https://glorykickboxing.com/) and [One Championship](https://www.onefc.com/)) of kickboxing for our training data. Some of the considerations that went into picking our data was that we didn't want a ring because that had a tendency of interfering with the algorithm and we didn't want any fights that were too fast-paced which would make it harder for the model to learn from.

### Data pre-processing

We performed two steps of pre-processing on the training data. First, we ran the MoveNet algorithm on the video to determine the fighters' keypoints which we would use as the input for our neural network. Second, we annotated the frames with the correct punch type, ie. jab, hook, high kick, etc. We used CVAT to simplify this process.

The process of pre-processing was by far the most time consuming process. Running MoveNet on the training data would take hours for only few minutes of video. Thankfully, MoveNet could run independently without any manual intervention. However, annotating the training data was a manual process, where we would step through each frame and determine if there is a punch or kick.

### Building the neural network

Our KickBoxingLSTM neural network is designed to capture the specific movements in kickboxing using LSTM (Long Short-Term Memory) layers. Here’s a breakdown of the architecture and rationale behind each component:

Input Layer: This layer receives sequences of keypoints representing positions of various body parts over time from Movenet. The input_size parameter specifies the number of features in each input vector, which should match the number of keypoints multiplied by their dimensions (x, y, z).

LSTM Layers: The core of this model is the LSTM layers, which are important for understanding sequences with long-range dependencies(sequences). Each LSTM layer processes the input sequence, maintains hidden states, and passes its outputs to the next layer. This architecture helps in capturing the progression of movements in kickboxing, like the transition from a blocking punches to landing a strike. The num_layers parameter allows for the stacking of multiple LSTM layers to improve the model’s grasp of sequences.

Dropout Layer: To mitigate the risk of overfitting, a dropout layer follows each LSTM layer, except for the last one. It randomly sets input elements to zero during training at a rate specified by dropout_rate, which helps prevent the model from relying too much on any small set of neurons.

Fully Connected Layer: After the LSTM layers, a fully connected layer reduces the dimensionality from the hidden state size to the number of output classes (NUM_CLASSES = 8). This layer is crucial for mapping the learned strikes(punches/kicks) to specific class predictions, like different types of strikes.

Output: The model outputs the logits for each class, which can be passed through a softmax function to obtain probabilities. However, for training stability and efficiency, we often use the raw logits with a loss function that incorporates softmax internally, like cross-entropy which is implemented in the training.

```py
   class KickBoxingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        """
        Initialize the HybridBoxingLSTM model with given parameters.
        Parameters:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int): Number of recurrent layers in the LSTM
            dropout_rate (float): If non-zero, we introduce a Dropout layer on the outputs of each
            individual LSTM layer which is the last layer, with the dropout probability equal to the dropout_rate
        """
        super().__init__()
        # Define the LSTM layer with specified input size, hidden size, and number of layers
        # `batch_first=True` indicates that the input tensors will have a shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # Define a fully connected layer to map the LSTM output to the number of classes
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Parameters:
            x is our Tensor: Input data tensor, expected shape (batch, seq_length, input_size)
        Returns:
            Tensor: Output tensor after passing through LSTM, dropout, and fully connected layer
        """
        # Convert input to float32
        x = x.float()
        # Ensure input tensor is 3D (batch, 1, input_size) if currently 2D
        if x.dim() == 2:
            x = x.unsqueeze(1)
            # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, dtype=torch.float32).to(x.device)
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Apply dropout on the outputs of the LSTM
        out = self.dropout(out)
        # Apply the fully connected layer on the last time step output
        return self.fc(out[:, -1, :])
   ```

### Training the neural network

Training the neural network was relatively simple after we had pre-processed the training data and built the neural network. All we had to do was feed the training data into the model and let it work it's magic including the steps below.

Data Loading and Pre-Processing: Before training, we load the keypoint data from our MoveNet CSV files and parse annotations from corresponding XML files. We also pre-process the data by normalizing the keypoints using a StandardScaler. This normalization is crucial as it ensures that the model is not biased towards features with inherently larger scales such as the category('No Strike').

Cross-Validation Setup: We employed K-fold cross-validation to ensure that our model generalizes well over different subsets of data. This method splits the dataset into k_folds sets which in our case is 5, using one fold for validation and the rest for training in each iteration. It helps in understanding the model's performance variations across different data splits.

Batch Processing: Data is fed into the model in batches using DataLoader. Implemented to optimize the training process with mini-batch gradient descent.

Optimization and Loss Calculation: We use the Adam optimizer for its adaptive learning rate capabilities, which help in converging faster which is esential for sequence learning. The CrossEntropyLoss function is used for calculating the loss as it's great for multi-class classification problems.

Model Evaluation and Saving: After each epoch, we evaluate the model on the validation set and save the model's state if it surpasses a predefined accuracy threshold > 90%. This step ensures that we retain the best version of the model throughout the training process.

Early Stopping: To avoid overfitting and unnecessary computations, we implemented an early stopping mechanism. If the validation loss does not improve for a specified number of epochs (patience), training is halted.

### Running the neural network on test data

Finally, the last step in the process is to run the trained neural network on a completely new test set and see how it performs.

### Adding punches/kicks to original video

For a little bit of polish, our group decided to take the predicted results from the model and add it to the original video, so it is easy to see how well our model is performing.

## Results

We ran the model on a few different test sets to determine how well it performed according to the metric defined earlier. 

The first set we ran the model on was part of the training set (https://www.youtube.com/watch?v=0cK-5Odirgo) and had a good accuracy of 98.83% excluding 'No Strike'

The second set we ran the model on was a normal kickboxing fight (https://www.youtube.com/watch?v=0yUIstA3iCI) and we got an accuracy of 50.00% excluding 'No Strike'

The final set we ran the model on was a muay thai fight (https://www.youtube.com/watch?v=g-tiUbL8vpw). Muay thai is similar to kickboxing in many regards but there are subtle differences. The accuracy on this video was 12.50% excluding 'No Strike'

## Reflection

Our initial plan was to create an application that would be able to detect the different kinds of punches and kicks thrown in a kickboxing match, like jabs, hooks, crosses and upper-cuts, and different kinds of kicks, like leg, body and high kicks. We could then use this data to tally up the final score of the match. However, we encountered many challenges along the way.

Main Challenges:

* Different interpretations of training data - Strikes landing at the same time, Strikes hitting shoulders and arms.
* Training video inconsistencies - Camera angle changes, Ring ropes stopping ideal MoveNet Output.
* Time consuming processes - On our current devices, processes like MoveNet take ~10 hours per fight to run.

Considering the challenges we faced, as a group we feel like we did relatively well on a pretty complicated project in the time that we had.

<!-- ABOUT THE PROJECT -->
## About The Project

CS482 - Final Project - This project will continue past the course.

### Built With

* [![MoveNet][MoveNet.js]][MoveNet-url]
* [![PyTorch][PyTorch.js]][PyTorch-url]
* [![NumPy][NumPy.js]][NumPy-url]
* [![OpenCV][OpenCV.js]][OpenCV-url]

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To set up a local copy of the project, follow these simple example steps.

### Prerequisites

This project requires Python and several Python libraries. Before you begin, ensure you have Python installed on your system. If not, you can download it from [Python](https://www.python.org/downloads/) (Version 3.8+ is recommended). You will also need to install the required Python libraries which are listed in the requirements.txt file after cloning the repository.

### Installation

Follow these instructions to get your development environment running:

1. Clone the repo
   ```sh
   git clone https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482.git
   ```
2. Navigate to the project directory:
   ```sh
   cd StrikeMetrics---FinalProjectCS482
   ```
3. Download Required Libraries
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Training and Inital Test with 5 datasets - 5/5/2024
- [ ] 40 datasets and more features added(Elbows, Different Kicks) - Fall 2024
- [ ] Full 100 datasets - Aiming for 95% plus accuracy. 

See the [open issues](https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Sean Sippel - ssippel@gmu.edu
Chris Mostert - jmostert@gmu.edu
Joseph Cherinet - jcherine@gmu.edu

Project Link: [https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482](https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482)

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Jabbr]: https://jabbr.ai/
[contributors-shield]: https://img.shields.io/github/contributors/sswhitehat/StrikeMetrics---FinalProjectCS482.svg?style=for-the-badge
[contributors-url]: https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sswhitehat/StrikeMetrics---FinalProjectCS482.svg?style=for-the-badge
[forks-url]: https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/network/members
[stars-shield]: https://img.shields.io/github/stars/sswhitehat/StrikeMetrics---FinalProjectCS482.svg?style=for-the-badge
[stars-url]: https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/stargazers
[issues-shield]: https://img.shields.io/github/issues/sswhitehat/StrikeMetrics---FinalProjectCS482.svg?style=for-the-badge
[issues-url]: https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/issues
[license-shield]: https://img.shields.io/github/license/sswhitehat/StrikeMetrics---FinalProjectCS482.svg?style=for-the-badge
[license-url]: https://github.com/sswhitehat/StrikeMetrics---FinalProjectCS482/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[MoveNet.js]: https://img.shields.io/badge/MoveNet-FFDDB3?style=for-the-badge&logo=tensorflow&logoColor=FC6A3F
[MoveNet-url]: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
[PyTorch.js]: https://img.shields.io/badge/PyTorch-66736C?style=for-the-badge&logo=pytorch&logoColor=FC6A3F
[PyTorch-url]: https://pytorch.org/
[NumPy.js]: https://img.shields.io/badge/NumPy-5581E0?style=for-the-badge&logo=numpy&logoColor=9DD4FF
[NumPy-url]: https://numpy.org/
[OpenCV.js]: https://img.shields.io/badge/OpenCV-FFFDFC?style=for-the-badge&logo=opencv&logoColor=FF3E00
[OpenCV-url]: https://opencv.org/
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
