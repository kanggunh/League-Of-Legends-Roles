# League-Of-Legends-Roles

---

## Framing the Problem

In this this project, we will be building a classifier, trying to determine whether post-game stats can predict the player's played role. There are 5 roles in this game, which are Top-lane, jungle, support, mid-lane, bot-lane. We want to predict one of these five using post-game stats to see if whether certain post-game stats belong to certain roles. Accordingly, we need to build a classifier that performs multiclass classification. 

For evaluating our model, we will be using accuracy test. Since we have 5 roles on each team every game, we have classes are evenly divided. 

At the time of prediction, we would not know the champion that the player selected since lot of the champions has the designated role. We would only have stats such as kills, death, assist, etc. to predict what role the player played.

#### Why choose these features?

**'kill':** the team plays to give certain roles (laners or jungle) more kill as a strategy

**'death':** some roles are more prone to being killed than others

**'assists':** roles such as support might have higher number of assists

**'teamkills' & 'teamdeaths':** might useful to find the percentage of kills and deaths of a given player on a team

**'visionscore':** jungle and support tend to have higher vision score, which may help

**'monsterkills':** jungle usually takes all monsters and support almost never takes any

**'damagetakenperminute':** usually top-laners can take the most damage, resulting in higher damage taken per minute, and bot-laners usually take significant less

side note: We will not include the champion of the player played since a champion is usually played on a certain role and that would be too easy to guess the role of the player

#### Assessment of Missingness
When assessing the missingness of our cleaned data with the columns that we need, we are able to see that we have 10 missing values for the columns 'visionscore', 'monsterkills', and 'damagetakenperminute'. These values occurs on the same game. If we look at the match with gameid **8479-8479_game_1**, we are able to see that all of the missing values are from the 10 players of this match. Therefore, without doing any missingness permutation test for this on this data set, we are able to conclude that it is missing at random.

Because there is only 10 rows of missing values out of 124500 rows, we will use mean imputation to fill the missing values. This will preserved the mean of the observed data.

---
