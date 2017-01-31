# ElntGen
The project shows how to train LSTM model that discriminates letters.

## Commands
`th trainer.lua` starts training procedure,

`th drawer.lua --run --r` draws an example of the letter,

`th drawer.lua --runProb --r` draws an example of the letter plus trajectories of probabilities for top7 letters.

You can always execute `th trainer.lua --help` or `th drawer.lua --help` to get more details.

## Input

Data used is preprocessed data taken from [CHILI EPFL, The CoWriter Project](https://github.com/chili-epfl/cowriter_logs). Since the data is very limited, this project should be treated as preliminary one.

The data consists handwritting trajectories which were written by children (fixed length - 70). A single input to the LSTM model is a sequence of 69 pairs which are changes in the position of the pen (dX, dY).

## Regularization

The data is very limited so the network has to be regularized in order not to overfit. I used a regularization method suited for this specific task.

For each letter the LSTM model is fed sequentially with 69 pairs. We want model to discriminate given letter at the end of the trajectory. However, the model predicts at the time of every point. Hence, target gradients may be calculated in up to 69 points.

In case of perfect optimizer and model we would use the last target gradient only, since we want model to answer our question at the end. However, we do not use perfect tools. Hence, we want to help and encourage LSTM model to store important information earlier. One can compare three cases:

##### All 69 gradients used

`th trainer.lua --mask 1` - starts nice, but tends to overfit.

##### Just the last one used

`th trainer.lua --mask 70` - struggles at the beginning, but finaly it isn't that bad.

##### Compromise - the method used as a default in the code

`th trainer.lua` or `th trainer.lua --mask 2` - works the best, the difference between training and validation errors are much more smaller.

One can invastigate the code to check how the option `--mask` works in details. In a nutshell this option makes model use only a logatihmic number of gradients. For `--mask 2` target gradients at the positions 34, 17, 8, 4, 2, 1, and 0 are used (counting from the end, so zero is the last one).

##### Extension

In fact this way of regularization may be extended. The positions where target gradients are used may evolve. We can also used weights instead of just using gradients or not. I believe that second model that is trained on partial LSTM learning procedures may be used to obtain superior regularization method.
