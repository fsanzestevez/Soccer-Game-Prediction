# Soccer Game Result Prediction

The idea of this project is to predict who wins the game with half'time as the prediction time point. 
Therefore, this is a multiclass classification problem in which we can have Home win, Away win and Draw as results.
This problem is approached as a time-series problem, meaning that the time of ocurrence of the event is meaningful for our prediction (we forecast the future with past results). 

## Data
I created the whole database from different datasets that I have found online: 
a) From https://www.football-data.co.uk/ I gathered the results and betting odds from the most famous betting companies since the 1994/95 season.
b) From https://www.kaggle.com/shubhmamp/english-premier-league-match-data I was able to find the lineups and match stats for every game in between 2014 to 2018.
c) From https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset I found the attributes of all the players from the game FIFA since 2014.

The key idea of my proposed solution is to see whether the players rating on the video game fifa have an impact on the results.

Every dataset I gather had a very different structure, and even different names for the same clubs/players that were all solved with the library FuzzyWuzzy (which uses the Levenshtein distance to match strings). All the data wrangling to create the database is in the file 'createDB.py' 
