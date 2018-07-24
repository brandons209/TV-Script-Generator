# Generating new scripts from learned TV show script
This LSTM RNN model with an embedding layer generates new scripts based on learn scripts from a TV show. I used one episode from [The Simpsons by the Data](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data/home) set of scripts and characters to train on. Specifically, the episode "Moes Tavern".    
While that is what I based this project around, this can be used with any script or text!    

[LICENSE](LICENSE)
____

### Objective
This model was created to generate new, mostly coherent original script text based on learned characters and speech from an episode of The Simpsons using an LSTM RNN neural network that uses an embedding layer for the words.

### Libraries
As with my other projects, I created this model using Keras with the TensorFlow backend. I also used TensorBoard for visualizing the model. Within Keras, I used it's text preprocessing for processing the text for input.

### Performance
Some examples, with 200 words generated:
 ```
 Bart_Simpson: dallas thoughts. goodnight dregs. goodnight dregs. goodnight man make soup out of a hearse... without a yes. the canyonero(" bar)" i wanted to play the drink to my girl!
homer_simpson:(short pipes) ooh, duff, homer.
homer_simpson:(pissed) you could play all cuckoo enough.
moe_szyslak: nah, you were not safe any way. 'cause no one could work on his show are divorced.
carl_carlson: huh?
homer_simpson:(to the tv)... going to make: shut up.
homer_simpson: moe, you got the kind of people uglier than you.
stillwater: i see, president till it looks.
carl_carlson: i took away a pulled duff's people.
moe_szyslak: an action.
homer_simpson: oh, i can't come here.
homer_simpson:(excited) oh, no, no. homer, there's your little time to get along the set.
marge_simpson:(a little man) c'mon,
 ```
```
Marge_Simpson: keep 'em as a hearse. and as burns sixty-nine.(to barflies)(high noises) / ya again!
moe_szyslak:(laughs gasp) we were pretty happy here, you steal a duff, but i am ladder if i can do something important now either..


moe_szyslak: moe's tavern, lisa, people.
moe_szyslak: aw, marriage is my allowance, agent, and too much.
moe_szyslak:(a little girl) aw, no. chicks do not pay a try, eh? a place at her.
homer_simpson:(morose) wait a minute.


homer_simpson: hey moe, you wanna go bowling miles out, but don't get you. help me with the kids out of you.
carl_carlson: what d'ya did this bad?
homer_simpson:(finishing noises) / huh?
marge_simpson:(reading sign) no, no, no, no, no, got no tape that you can watch the money you like me?
```
```
Krusty_the_clown: yep. hold on there and kill yourself, boozy. i wanna have a fiiiiile.
homer_simpson:(looking at cards) oh, no.
moe_szyslak:(sad) no, o'reilly.
homer_simpson: hey, hey, hey, hey. no outside suds!
snake_jailbird:(concerned) hey homer, i had to go too far.
homer_simpson: barney, before i let. you went to that?
moe_szyslak: an action.
homer_simpson:(pissed) wiggum forever, wiggum need my keys to baseball.
moe_szyslak: okay, which is a good again.
barney_gumble:(totally enthused) i all need...
moe_szyslak: homer! thank god, i would count of the cash.
moe_szyslak: hey, wait a. that wraps it to the minute--(beat) patty, can i get my help.
jukebox_record:(sings) life god!
marge_simpson: maybe it does i gotta sell it without the thing so much.

```
With just being trained on one episode, the scripts generated are pretty legible, with use of character's actions in parentheses, and new lines for each character that speaks, and two new lines for a new scene. Although it seems like what they are saying and doing is quite odd, it makes sense and flows a quite well.

I am very pleased with the results, it was much better than I expected. The size of the network is decent, with two 400 unit LSTM layers and 300 nodes for the embedding layer. The model took about 4 hours and 15 minutes to train on a NVIDIA GTX 1060. With these results, I expect that if I had the time and power, that training on a few episodes, or all 600 episodes could yield some very high quality script creation.

With training the model, I used the Adam optimizer and sparse_categorical_crossentropy as the loss. This loss function is nice since it doesn't require the labels to be one-hot encodded, which with a large vocabulary would be very computationally expensive.

### Running the model
Simply set the hyperparameters at the top of simpson_script_gen.py, and also change the path to where the script you want to train on is stored if you want to use your own script. Then run the model! It will save the model after training and also save the dictionaries to use so you can test the model with test_model.py.

If you want to see the output to TensorBoard while training, run tensorboard --logdir=tensorboard_logs

### Testing the model
Simply run python test_model.py /path/to/model and enter the seed and amount of words to be generated. I provide a model for you if you want to try it out yourself!

### TODO
I want to clean up the generated text a bit more, since the way I convert the text to tokens and back is a little clunky.
