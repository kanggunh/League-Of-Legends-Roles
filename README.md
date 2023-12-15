
---

# **Framing the Problem**

In this short website, I would like to build a classification model that predict which role a player plated given their post-game data.

We will be using a csv file provided bt Oracle's Elixir. The dataset we are about to explore contain match data from LCS, LEC, LCK, and more from 2022. Previously, we worked on our exploratory data analysis on this dataset which can be found [here](https://kanggunh.github.io/LeagueOfLegends/).

There are 5 roles in this game, which are top-lane, jungle, support, mid-lane, bot-lane. We want to predict one of these five using post-game data. Accordingly, we need to build a classifier that performs multiclass classification. 

To evaluate our model, we will be using accuracy test. Accuracy test works as our metric of evaluation since we have evenly distrubuted classes. Every game, each team as 5 roles from the 5 mentioned above. This means that our roles are evenly distrubuted.

At the time of prediction, we would not know the champion that the player selected since lot of the champions has the designated role. We would also not know the player username. We would only have post-game stats such as kills, death, assist, damage, etc. to predict what role the player played.

---

# **Baseline Model**
For the baseline model, the classifier we chose a decision tree classifier with two quantitative features. We do not have any ordinal or nominal features.

We first chose 'damagetaken' as our first feature. This feature tells us about the amount of damage a player can take on average depending on their role. Typically top-lane, and jungle players play more tankier champions. The other two, bot-lane and mid-lane, tend to have less health. We binarized this feature with a threshold of 590. This threshold was chosen in an attempt to classifier top-lane and jungle as the tankier roles and others as less tankier. This threshold was chosen after evalutaing the mean 'damagetakenperminute' by position.

Then the second feature we chose was wardsplaced. One of many things that support do is to focus on placing wards. We thought that this would help our model to predict support. We binarized this feature as well. Threshold for this was 21. Similar to the previous feature, this was based on the computed mean by roles, in an attempt to identify support from the data.

```
baseline_col_transformer = ColumnTransformer(
    transformers=[
        ('damagetaken', Binarizer(threshold=590), ['damagetakenperminute']),
        ('wardsplaced', Binarizer(threshold=21), ['wardsplaced'])
        ],
        remainder='drop'
        )
pl_base = Pipeline([
    ('col_transformer', baseline_col_transformer),
    ('tree', DecisionTreeClassifier(max_depth=8))
])
```

Our accuracy on the training set was about 53.61%. The training set had a accuracy of about 53.29%. First thing we notice is that the similar accuracy shows that the model is not overfitting. For a baseline model with only two features, the model does a decent job predicing the roles. If the model predicts only one role, the accuracy would be 20%. With this model having only two features, an accuracy little over 50% is not too bad for a baseline model.

---
# **Final Model**
For our final model, we are going to use 8 quantitative features and transform 4 of them. We will be using a decision tree classifier like before. We do not have any ordinal or nominal features.

Here is the general idea for why we chose these features.
- **'kill':** the team plays to give certain roles (laners or jungle) more kill as a strategy
- **'death':** some roles are more prone to being killed than others or sacrifice themselves 
- **'assists':** roles such as support might have higher number of assists
- **'visionscore':** jungle and support tend to have higher vision score, which may help
- **'earned gpm':** certain roles earned more gold than others
- **'wardsplaced':** support tends to place more wards
- **'monsterkills':** jungle usually takes all monsters and support almost never takes any
- **'damagetakenperminute':** usually top-laners can take the most damage, resulting in higher damage taken per minute, and bot-laners usually take significant less

With these features introduced, we made sure that there was no missing values. When assessing the missingness of our cleaned data with the columns that we need, we are able to see that we have 10 missing values for the columns 'visionscore', 'monsterkills', 'damagetakenperminute', and 'wardsplaced'. If we look at the match with gameid **8479-8479_game_1**, we are able to see that all of the missing values are from the 10 players of this match. Therefore, without doing any missingness permutation test for this on this data set, we are able to conclude that it is missing at random. Since there is only 10 rows of missing values out of 124500 rows, we will use mean imputation to fill the missing values. This will preserved the mean of the observed data.

### **Modeling Pipeline**
We kept the two features from the baseline model binarized. Then we chose to binarize two more features, 'earned gpm' and 'visionscore'. These two was specifcally binarized so that it accounts for different stats that are more relevent to the roles we want to predict. The rest of the features were passed as is. 

For this final model, we kept our modeling algorithm the same as before. We are going to use decision tree classifier. However this time we wanted to figure out the best hyperparameters of this model. We used GridSearchCV from sklearn to do this. Our hyperparameters were the 'threshold' of each individual Binarizer,'max_depth' of the decision tree, and 'criterion' of the decision tree. 

This process had some limitation for us with computing power on Jupyter Notebook. Since we have 6 hyperparameters and 5 folds to go through, it was taking too long if we put lot of values in this grid search. In order to address this run time issue, we chose the list values for threshold based on the mean of each feature. We experimented in increments (different intervals in increments) to allow GridSearch to run on our jupyter notebook within a reasonable amount of time.

Here is the final model pipeline with the hyperparameter we found through gridsearch.
```
col_transformer = ColumnTransformer(
    transformers=[
        ('dmgtkn', Binarizer(threshold=568), ['damagetakenperminute']),
        ('wrds', Binarizer(threshold=19), ['wardsplaced']),
        ('earnedgpm', Binarizer(threshold=231), ['earned gpm']),
        ('vscore', Binarizer(threshold=24), ['visionscore'])
        ],
        remainder='passthrough'
        )
pl_final = Pipeline([
    ('col_transformer', col_transformer),
    ('tree', DecisionTreeClassifier(max_depth=8, criterion='gini'))
])
```
The accuracy of this model on training set and testing set was about 74.63% and 73.98% respectively. These accuracy of the two sets once again show that our model is not overfitting. We are able to generalize on unseen data. The inclusion of new features in this final model improved the accuracy by about 20%. 

We believe that our new features improved the performance of our model because they included the more characteristics of individual roles. The baseline model only focused on features that was more focused towards top-laners and support. This neglected the other roles. However, this final model includes other features that include all the other roles. For example, the feature 'earned gpm' helped to classify bot-laners since they tend to spend more time collecting gold during the game.

---
# **Fairness Analysis**
For our fairness analysis, we decided to look at 'monsterkills'. We manually created these groups by binarizing the 'monsterkills' columns in our dataset, using the Binarizer transformer with a threshold of 100.

The two groups are the following:
- "monster-killer", player who has more than 100 'monsterkills'
- "non-monster-killer", plater who has less than or equal to 100 'monsterkills'

For our evaluation, we will choose accuracy since the number of roles are equal to one another like mentioned previously. Computing the accuracy in each group, we get that 'monster-killer' gets an accuracy of 99.87% and 'non-monster-killer' get accuracy of 68.47%.

We then performed a permutation test to see if the difference in accuracy is significant.
- Null Hypothesis: The classifier's accuracy is the same for both monster-killer and non-monster-killer, and any differences are due to chance.
- Alternative Hypothesis: The classifier's accuracy is higher for players who are monster-killer.
- Test statistic: Difference in accuracy (monster-killer minus non-monster-killer)
- Significance level: 0.01

The resulting p-value of 0.0 is less than the significance level of 0.01.
Therefore, we reject the null hypothesis.

---
