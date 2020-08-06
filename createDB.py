# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:53:36 2020

@author: Fran
"""

import pandas as pd
import fuzzywuzzy
import json
import os 

class CreateDB():
    
    def __init__(self, folder):
        self.folder = folder
        self.uniquePlayers = set()
        self.player2id = dict()
        self.lineupsDB = self.loadData()
    
    @staticmethod
    def seasonLineups(df, d, players_Fifa):
        '''
        This static method receives a JSON object (dict) and an empty base 
        DataFrame which it will be filling with the relevant data from the 
        JSON object
    
        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. Empty base DF that will be filled up. It should have 
            the columns needed.
        d : TYPE dict
            DESCRIPTION. JSON object with the raw information about the lineups
            for every game in the season
        players : TYPE pandas.DataFrame
            DESCRIPTION. Fifa Ratings of the players of the specific season
    
        Returns
        -------
        lineups : TYPE pd.DataFrame
            DESCRIPTION. Filled up DataFrame with all the starting players of 
            each game.
    
        '''

        lineups = df.copy()    
        for game_id in d.keys():
            
            game = d[game_id]
            for team_id in game.keys():
                team = game[team_id]
                date = team['team_details']['date']
                team_name = team['team_details']['team_name']
                players_lst = []
                plyr_sts = team['Player_stats']
                for player, stats in plyr_sts.items():
                    place = int(stats['Match_stats']['formation_place'])
                    if place:
                        players_lst.append(player)

                lineups = lineups.append({'Date': date, 'Team': team_name},
                                         ignore_index=True)
    
        return lineups
    
    def loadData(self):
        '''
        
    
        Returns
        -------
        lineups : TYPE
            DESCRIPTION.
    
        '''
        cols = ['Date', 'Team', 'Player_1', 'Player_2', 'Player_3', 'Player_4',
                'Player_5', 'Player_6', 'Player_7', 'Player_8', 'Player_9',
                'Player_10', 'Player_11']
        lineups = pd.DataFrame(columns=cols)
        for (root, dirs, files) in os.walk(self.folder):
            for f in files:
                if f == 'season_stats.json':
                    path = root + '/' + f
                    d = json.load(open(path, 'r', encoding='utf-8'))
                    players = pd.read_csv(root+'/players_'+root[-2:]+'.csv')
                    lineups = self.seasonLineups(lineups, d, players)

        
        return lineups

       
    def downloadResults(self, start = 14, end = 18): 
        league = '/E0' # EPL

        link = 'https://www.football-data.co.uk/mmz4281/'
       
        for year in range(start, end):
            season = str(year) + str(year+1)

            download = pd.read_csv(link + season + league + '.csv')
            download.to_csv(self.folder + '/season' + str(year) + str(year+1) \
                            + '/results.csv')
    
    def playersDB(self):

        playersDB = pd.DataFrame(columns = ['sofifa_id'])
        
        for (root, dirs, files) in os.walk(self.folder):
            for f in files:
                if f.startswith('players_'):
                    path = root + '/' + f
                    players_year = pd.read_csv(path)
    
                    colDict = dict()
                    dont_rename = ['sofifa_id', 'dob']
                    for col in players_year.columns:
                        if col in dont_rename:
                            pass
                        else:
                            colDict[col] = col + year 
                    players_year.rename(columns=colDict, inplace=True)
                    playersDB = pd.merge(playersDB, players_year, how='outer', 
                                         on='sofifa_id')
            
        self.playersDB = playersDB
        return 
                
                
for (root, dirs, files) in os.walk('data'):
    print('root', root)
    print('season', root[-2:])
    print('dirs', dirs)
    print('files', files)                
                
                
                
                
                
                
                
    