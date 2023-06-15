# League-Of-Legends-Roles

---
## Framing the Problem

In this this project, we will be building a classifier, trying to determine whether post-game stats can predict the player's played role. There are 5 roles in this game, which are Top-lane, jungle, support, mid-lane, bot-lane. We want to predict one of these five using post-game stats to see if whether certain post-game stats belong to certain roles. Accordingly, we need to build a classifier that performs multiclass classification. 

For evaluating our model, we will be using accuracy test. Since we have 5 roles on each team every game, we have classes are evenly divided. 

At the time of prediction, we would not know the champion that the player selected since lot of the champions has the designated role. We would only have stats such as kills, death, assist, etc. to predict what role the player played.


#### Assessment of Missingness
When assessing the missingness of our cleaned data with the columns that we need, we are able to see that we have 10 missing values for the columns 'visionscore', 'monsterkills', and 'damagetakenperminute'. These values occurs on the same game. If we look at the match with gameid **8479-8479_game_1**, we are able to see that all of the missing values are from the 10 players of this match. Therefore, without doing any missingness permutation test for this on this data set, we are able to conclude that it is missing at random.

Because there is only 10 rows of missing values out of 124500 rows, we will use mean imputation to fill the missing values. This will preserved the mean of the observed data.

---
## Baseline Model
For the baseline model, the classifier we chose a decision tree classifier with two features.

We first chose 'damagetaken' as our first feature. This feature tells us about the amount of damage a player can take on average depending on their role. Typically 'top-lane', and 'jungle' players play more tankier champions. The other two, 'bot' and 'mid', tend to have less health. We binarized this feature with a threshold of 590. This threshold was chosen in an attempt to classifier 'top-lane' and 'jungle' as the tankier roles and others as less tankier. This was done after evalutaing the mean 'damagetakenperminute' by position.

Then the second feature we chose was wardsplaced. One of many things that support do is to focus on placing wards. We thought that this would help our model to predict supports. We binarized both of these features with their own threshold. Threshold for this was 21 respectively. Similar to the previous feature, this was based on the computed mean by roles, in an attempt to identify support from the data.

Our accuracy on the training set was 53.61%. The training set had a accuracy of 53.29%. For a baseline model with only two features, the baseline model is decent at doing its job. Since the roles are distrubuted evenly, if the model predicts only one role, the accuracy would be 20%. With this model having only two feature that are more focused on three of the roles, a over 50% accuracy is not too bad for a baseline model. The similar accuracy shows that the model is generalizable. 

---
## Final Model
For our final model, we are going to use 8 features, with 4 of then binarized. We will be using a decision tree classifier like before.

We have 8 quantitative post-game data with our cleaned data. It includes kill, death, assists, visionscore, earned gpm, wardsplaced, monsterkills.

### Why choose these features?

**'kill':** the team plays to give certain roles (laners or jungle) more kill as a strategy

**'death':** some roles are more prone to being killed than others

**'assists':** roles such as support might have higher number of assists

**'visionscore':** jungle and support tend to have higher vision score, which may help

**'earned gpm':** certain roles earned more gold than others

**'wardsplaced':** support tends to place more wards. might be good predictor

**'monsterkills':** jungle usually takes all monsters and support almost never takes any

**'damagetakenperminute':** usually top-laners can take the most damage, resulting in higher damage taken per minute, and bot-laners usually take significant less



---
## Fairness Analysis


---





