# **Predicting Roles with post-game stats**

---

## **Framing the Problem**

In this short website, I would like to build a model that predict which role a player plated given their post-game data.

We will be using a csv file provided bt Oracle's Elixir. The dataset we are about to explore contain match data from LCS, LEC, LCK, and more from 2022. Previously, we worked on our exploratory data analysis on this dataset can be found [here](https://kanggun-ucsd.github.io/LeagueOfLegends/).


There are 5 roles in this game, which are Top-lane, jungle, support, mid-lane, bot-lane. We want to predict one of these five using post-game data. Accordingly, we need to build a classifier that performs multiclass classification. 

For evaluating our model, we will be using accuracy test. Since we have 5 roles on each team every game, we have classes are evenly divided. 

At the time of prediction, we would not know the champion that the player selected since lot of the champions has the designated role. We would only have stats such as kills, death, assist, etc. to predict what role the player played.


### Assessment of Missingness
When assessing the missingness of our cleaned data with the columns that we need, we are able to see that we have 10 missing values for the columns 'visionscore', 'monsterkills', and 'damagetakenperminute'. These values occurs on the same game. If we look at the match with gameid **8479-8479_game_1**, we are able to see that all of the missing values are from the 10 players of this match. Therefore, without doing any missingness permutation test for this on this data set, we are able to conclude that it is missing at random.

Because there is only 10 rows of missing values out of 124500 rows, we will use mean imputation to fill the missing values. This will preserved the mean of the observed data.

---
## **Baseline Model**
For the baseline model, the classifier we chose a decision tree classifier with two features.

We first chose 'damagetaken' as our first feature. This feature tells us about the amount of damage a player can take on average depending on their role. Typically 'top-lane', and 'jungle' players play more tankier champions. The other two, 'bot' and 'mid', tend to have less health. We binarized this feature with a threshold of 590. This threshold was chosen in an attempt to classifier 'top-lane' and 'jungle' as the tankier roles and others as less tankier. This was done after evalutaing the mean 'damagetakenperminute' by position.

Then the second feature we chose was wardsplaced. One of many things that support do is to focus on placing wards. We thought that this would help our model to predict supports. We binarized both of these features with their own threshold. Threshold for this was 21 respectively. Similar to the previous feature, this was based on the computed mean by roles, in an attempt to identify support from the data.


Our accuracy on the training set was 53.61%. The training set had a accuracy of 53.29%. For a baseline model with only two features, the baseline model is decent at doing its job. Since the roles are distrubuted evenly, if the model predicts only one role, the accuracy would be 20%. With this model having only two feature that are more focused on three of the roles, a over 50% accuracy is not too bad for a baseline model. The similar accuracy shows that the model is generalizable. 

---
## **Final Model**
For our final model, we are going to use 8 features, with 4 of then binarized. We will be using a decision tree classifier like before.

We have 8 quantitative post-game data with our cleaned data. It includes kill, death, assists, visionscore, earned gpm, wardsplaced, monsterkills.


Here is the general idea for why we chose these features.
>**'kill':** the team plays to give certain roles (laners or jungle) more kill as a strategy

>**'death':** some roles are more prone to being killed than others

>**'assists':** roles such as support might have higher number of assists

>**'visionscore':** jungle and support tend to have higher vision score, which may help

>**'earned gpm':** certain roles earned more gold than others

>**'wardsplaced':** support tends to place more wards. might be good predictor

>**'monsterkills':** jungle usually takes all monsters and support almost never takes any

>**'damagetakenperminute':** usually top-laners can take the most damage, resulting in higher damage taken per minute, and bot-laners usually take significant less

### Modeling Pipeline
We kept the two features from the baseline model binarized. Then we chose to binarize two more features, 'earned gpm' and 'visionscore'. These two was specifcally binarized so that it accounts for different stats that are more relevent to the roles we want to predict. The rest of the features were passed as is. 

For this final model, we wanted to figure out the best hyperparameters of this model. We used GridSearchCV from sklearn to do this. Our hyperparameters were the threshold of each individual Binarizer,'max_depth' of the decision tree, and 'criterion' of the decision tree. We found it taking too long if we put lot of values in this grid search since we have 6 hyperparameters and 5 folds to go through. In order to address this time issue, we chose the list values for threshold based on the mean of each feature. We experimented in increments to allow GridSearch to run on our jupyter notebook within a reasonable amount of time.

This resulted in the following hyperparameter.

```
col_transformer = ColumnTransformer(
    transformers=[
        ('dmgtkn', Binarizer(threshold=568), ['damagetakenperminute']),
        ('wrds', Binarizer(threshold=21), ['wardsplaced']),
        ('earnedgpm', Binarizer(threshold=232), ['earned gpm']),
        ('vscore', Binarizer(threshold=23), ['visionscore'])
        ],
        remainder='passthrough'
        )
pl_final = Pipeline([
    ('col_transformer', col_transformer),
    ('tree', DecisionTreeClassifier(max_depth=8, criterion='gini'))
])
```

---
## **Fairness Analysis**
For our fairness analysis, we decided to look at 'monsterkills'. We manually created these groups by binarizing the 'monsterkills' columns in our dataset, using the Binarizer transformer with a threshold of 100.

The two groups are the following:
- "monster-killer", player who has more than 100 'monsterkills'
- "non-monster-kills", plater who has less than or equal to 100 'monsterkills'

For our evaluation, we will choose accuracy since the number of roles are equal to one another like mentioned previously.

Now, we wil perform a permutation test to see if the difference in accuracy is significant.

- Null Hypothesis: The classifier's accuracy is the same for both monster-killer and non-monster-killer, and any differences are due to chance.

- Alternative Hypothesis: The classifier's accuracy is higher for players who are monster-killer.

- Test statistic: Difference in accuracy (monster-killer minus non-monster-killer)

- Significance level: 0.01

The resulting p-value of 0.0 is less than the significance leven of 0.01.
Therefore, we reject the null hypothesis.

---





