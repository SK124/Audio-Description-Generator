# Media Description Generator
-When AI turns journalist
## Purpose

* Making information more accessible to people with disability.
     *Use case:* Providing descriptive facts from ambient sounds to the hearing-impaired. 
     
* Providing unbaised reporting while it gets more relevant in the trend of [microtargeting](https://en.wikipedia.org/wiki/Microtargeting).
    *Use case:* A standard repository of facts which form the basis of all news outlets.
    
* Richer subtitle generation for non-speech sounds.
    *Use case:* Improved subtitles on streaming services.
     
* Obataining relevant facts from a large swaths of data which is otherwise resource intensive for humans.
    *Use case:* Detecting unusual activity in lengthy surveillance media to strengthen security.
    
* Archival of hefty media in an extremely compact format.(See [WaybackMachine](https://archive.org/web/))
    *Use case:* Keeping a record of current events serving as a time-capsule of humanity.
    
## Approach
<img src="https://github.com/tejasvi/AI_Hackathon/raw/master/genesis/overview.svg?sanitize=true">

* The audio is first converted to WAV format which is the uncompressed form.

* The audio is then converted into a spectrogram which is the visual representation of sound frequencies.

* The spectrograms can then be treated as images where we can benefit from transfer learning using pretrained models (like Resnet).

* After being fed separatly to classification and speech recognition network, appropriate keyword description is generated.

* The obtained keywords can further be used to generate readable sentences using [NLG](https://en.wikipedia.org/wiki/Natural-language_generation). However it proved to be much tedious to prototype.

<img src="https://github.com/tejasvi/AI_Hackathon/raw/master/genesis/planned.svg?sanitize=true">


* The problem scope can be extended to object level reseasoning in images. This will provide more context to the description by using videos along with audio.
 

## Implementation


### Data and preprocessing

The dataset used in this project is [Freesound Audio Tagging](https://arxiv.org/pdf/1906.02975) which contains snippets of audio with sound type labels.

The sounds are further divided into ~11 hr *curated* set and ~80 hr *noisy* set. We used only curated set for training due to resource constraints and for stronger transfer learning demonstration.

For spectrogram generation, widely used librosa library is used. Each audio of length *n* seconds is converted into 128x128n grayscale image.


There are 80 possible labels inlcuding *Gasp,Printer, Gong, Bark, Male singing,etc*. The audio falling in human voice category are fed to speech recognition model.

### Architecture

For classification, ResNet18 architecture pre-trained for ImageNet is used. The evaluation metric used is [LwLRAP](https://www.kaggle.com/pkmahan/understanding-lwlrap) which is believed to be most effective and widely used with spectrograms.

Speech recognition uses [DeepSpeech](https://github.com/mozilla/DeepSpeech) architecture published by Mozilla. 

### Transfer learning

The parameters of ResNet18 are initialized for ImageNet dataset instead of being random. The random initialization demonstrably converged slower than intializion to ImageNet weights.
#### Random Initialization
<img src="https://github.com/tejasvi/AI_Hackathon/raw/master/genesis/random_init.JPG">

#### ImageNet Initialization
<img src="https://github.com/tejasvi/AI_Hackathon/raw/master/genesis/img_init.JPG">

However freezing inner layers did not give any good results. Instead we had to unfreeze all layers to achieve faster convergence. It may be because the images of ImageNet have much different features therefore inner layers also required training.


## Execution

(See [`main.ipynb`](https://github.com/tejasvi/AI_Hackathon/blob/master/main.ipynb)). The trained model was exported after training and is loaded during inference. As a prototype, currently audio files are transcribed in batches. However, it is possible to do classification and speech recognition parallely for real-time processing. For simplicity it currently runs in sequence.

## Resources

* Implementation of LwLARP metric is taken from [Dan Ellis](https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8).
* For data profiling, [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) is used to obtain general overview of data.
* Use of [FastAI](http://fast.ai) library to structure and train the model.
* Other resources include StackOverflow, Medium and the rest.
